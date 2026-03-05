import shutil,time, math, os, types, requests, gc, copy, json, random, re
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_load_checkpoint
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file, save_file
from accelerate import Accelerator, PartialState
from accelerate.utils import broadcast
from datasets import load_dataset
import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    DataCollatorWithPadding,
    TrainerControl,
    Trainer,
)
from accelerate.utils import set_seed
INVALID_LOGPROB = 1.0
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import forward, truncate_response, first_true_indices, batch_generation, OnlineTrainerState, exact_div, disable_dropout_in_model,prepare_deepspeed, pad
from verifiable_lib.reward_func import IFEvalVerifier
from verifiable_lib.gsm8k_reward_func import GSM8KVerifier
from verifiable_lib.ifeval_reward_func import run
ifeval_reward_function = IFEvalVerifier()
gsm8k_reward_function = GSM8KVerifier()

import deepspeed
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from contextlib import nullcontext
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.extras.vllm_client import VLLMClient

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

def correlation_reward_function(content, label):
    return float(any([content.lower().startswith(v.lower()) for v in label]))

def get_json_list2(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

class MixedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(MixedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        list_data_dict=[]
        for data in os.listdir(data_path):
            list_data_dict += get_json_list2(os.path.join(data_path,data))
        for v in list_data_dict:
            if random.random()>0.95:
                v["data_attribute"]="CoT"
        self.sources = [v for v in list_data_dict if len(tokenizer.apply_chat_template(v["prompt"], padding=False, add_generation_prompt=True, enable_thinking=v["data_attribute"]=="CoT",))<=max_len]
        random.shuffle(self.sources)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        ret = self.tokenize(self.sources[i])
        return ret

    def tokenize(self, element):
        input_ids = self.tokenizer.apply_chat_template(
            element["prompt"],
            padding=False,
            add_generation_prompt=True,
            enable_thinking=element["data_attribute"]=="CoT",
        )
        cp_element = copy.deepcopy(element)
        prompt = cp_element.pop("prompt")
        data_type = cp_element.pop("data_type")
        rest_key = cp_element
        return {"input_ids": input_ids, "lengths": len(input_ids), "prompt": prompt, "data_type":data_type, "kwargs":rest_key}

def masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values * mask

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class Batch_prompts(object):
    def __init__(self,prompts, data_types, kwargs):
        self.prompts = prompts
        self.data_types = data_types
        self.kwargs = kwargs

    #asign "to" method to deal with accelerate problem
    def to(self, device=None):
        return self

class DataCollatorWithPadding_with_prompt(DataCollatorWithPadding):
    def __call__(self, features):
        prompts = [v.pop("prompt") for v in features]
        data_types = [v.pop("data_type") for v in features]
        kwargs = [v.pop("kwargs") for v in features]
        input_ids = [v.pop("input_ids") for v in features]
        lengths = [v.pop("lengths") for v in features]
        batch={}
        batch["input_ids"]=input_ids
        batch["lengths"]=lengths
        batch["prompts"]=Batch_prompts(prompts, data_types, kwargs)
        return batch

def strict_format_reward_func(string):


   """Reward function that checks if the completion has a specific format."""

   pattern = r"^<think>\n(.*?)\n</think>\n\n"

   matches = re.match(pattern, string, re.DOTALL)
   #think_content=re.findall(pattern, string, re.DOTALL)
   if not matches:
       return 0.0
   return 1.0 if len(matches.group())>50 else 0.0

def get_reward_from_different_model(model, prompt, data_type, rest_arg, tokenizer):
    rewards=[]
    for p,d,g in zip(prompt,data_type, rest_arg):
        if g["data_attribute"]=="CoT":
            if strict_format_reward_func(p[-1]["content"]):
                s=1.0
                p[-1]["content"]=p[-1]["content"].split("</think>\n\n")[-1]
            else:
                s=0.0
        elif g["data_attribute"]=="NoCoT":
            if "<think>" not in p[-1]["content"] and "</think>" not in p[-1]["content"]:
                s=1.0
            else:
                s=0.0
        else:
            raise ValueError("Not supported data attribute")

        if d=="kbqa_answer":
            if "😓" in p[-1]["content"] :
                rewards.append(0.0*s)
            else:
                payload = {"msg": p}
                response = requests.post(model, json=payload)
                rewards.append(response.json()["score"]*s)
        elif d=="kbqa_reject":
            if "😓" in p[-1]["content"] :
                rewards.append(15.0*s)
            else:
                rewards.append(0.0*s)
        elif d=="rlvr":
            score = ifeval_reward_function(tokenizer.encode(p[-1]["content"]), p[-1]["content"], json.loads(g["ground_truth"]))
            if score:
                payload = {"msg": p}
                response = requests.post(model, json=payload)
                rewards.append(response.json()["score"]*s)
            else:
                rewards.append(0.0*s)
        elif d=="ifeval":
            strict_result, loose_result = run({"prompt":p,**g})
            scale = sum(strict_result+loose_result)/(len(strict_result+loose_result)+1e-5)
            payload = {"msg": p}
            response = requests.post(model, json=payload)
            rewards.append(response.json()["score"]*scale*s)
        elif d=="gsm8k":
            score = gsm8k_reward_function(tokenizer.encode(p[-1]["content"]), p[-1]["content"], json.loads(g["answer_only"]))
            if score:
                payload = {"msg": p}
                response = requests.post(model, json=payload)
                rewards.append(response.json()["score"]*s)
            else:
                rewards.append(0.0*s)
        elif d=="correlation":
            score = correlation_reward_function(p[-1]["content"], g["label"])
            if score:
                payload = {"msg": p}
                response = requests.post(model, json=payload)
                rewards.append(response.json()["score"]*s)
            else:
                rewards.append(0.0*s)
        else:
            payload = {"msg": p}
            succeed=False
            while not succeed:
                try:
                    response = requests.post(model, json=payload)
                    rewards.append(response.json()["score"]*s)
                    succeed=True
                except:
                    time.sleep(5)

    return torch.tensor(rewards)

def get_ref_logp_vllm(model, query_responses, pad_token_id, context_length, temperature):
    payload = {"msg": {"query_responses": [v.cpu().numpy().tolist() for v in query_responses],
                       "pad_token": pad_token_id,
                       "context_length": context_length,
                       "temperature": temperature}}
    succeed=False
    while not succeed:
        try:
            response = requests.post(model, json=payload)
            ref_logp = response.json()["ref_logprob"]
            succeed=True
        except:
            time.sleep(5)
    return [torch.tensor(v) for v in ref_logp]

class CustomizedGrpoTrainer(RLOOTrainer):
    def __init__(
        self,
        config,
        processing_class,
        policy,
        ref_policy,
        reward_model,
        train_dataset,
        data_collator=None,
        eval_dataset=None,
        # less commonly used
        optimizers=(None, None),
        callbacks=None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy
        #added for save problem
        self.tokenizer = self.processing_class

        self.policy.generation_config.eos_token_id = 151645
        self.policy.generation_config.pad_token_id = 151643

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )*args.rloo_k
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding_with_prompt(self.processing_class),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding_with_prompt(self.processing_class),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if accelerator.is_main_process:
            self.vllm_client = VLLMClient(host="0.0.0.0")
        accelerator.wait_for_everyone()

        rank0_print("args.world_size:",args.world_size)
        rank0_print("args.local_batch_size:",args.local_batch_size)
        rank0_print("args.num_mini_batches:",args.num_mini_batches)
        rank0_print("args.batch_size:",args.batch_size)
        rank0_print("args.mini_batch_size:",args.mini_batch_size)
        rank0_print("args.local_mini_batch_size:",args.local_mini_batch_size)
        rank0_print("self.local_dataloader_batch_size:",self.local_dataloader_batch_size)
        rank0_print("self.train_dataset_len:",self.train_dataset_len)

        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        self.gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        self.is_fsdp_xla_v1_enabled, self.is_fsdp_xla_enabled, self.is_fsdp_xla_v2_enabled = None, None, None

    """
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)
    """

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None,):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model, resume_from_checkpoint, load_module_strict=True)
                rank0_print(f"Resume model from {resume_from_checkpoint}")
            else:
                rank0_print("Not supported")

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
            self._load_scaler(resume_from_checkpoint)

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        #self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = OnlineTrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.state.train_batch_size = args.micro_batch_size
            #or use init_training_references
            self.state.is_world_process_zero= self.is_world_process_zero()
            self.state.is_local_process_zero= self.is_local_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        rank0_print("self.state.max_steps:",self.state.max_steps)
        rank0_print("self.state.num_train_epochs:",self.state.num_train_epochs)
        rank0_print("self.state.logging_steps:",self.state.logging_steps)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            data = next(iter_dataloader)
            if resume_from_checkpoint and update<=self.state.global_step:
                continue
            self.state.episode += 1 * args.batch_size
            with torch.no_grad():
                queries = data["input_ids"]*args.rloo_k
                context_lengths = data["lengths"]*args.rloo_k
                prompts = [copy.deepcopy(v) for v in data["prompts"].prompts*args.rloo_k]
                data_types = data["prompts"].data_types*args.rloo_k
                rest_args = data["prompts"].kwargs*args.rloo_k
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                sequence_lengths = []
                response_lengths = []

                accelerator.wait_for_everyone()
                for name, module in accelerator.unwrap_model(model).named_modules():
                    if not list(module.named_children()):
                        if isinstance(module, QuantLinear):
                            with self.gather_if_zero3([module.weight, module.min, module.max, module.zeros, module.scales, module.delta_zeros]):
                                if accelerator.is_main_process:
                                    weight = module.fake_quant(module.weight)
                                    self.vllm_client.update_named_param(name+".weight",weight)
                            if module.bias is not None:
                                with self.gather_if_zero3([module.bias]):
                                    if accelerator.is_main_process:
                                        self.vllm_client.update_named_param(name+".bias",module.bias.data)
                        else:
                            for n,p in module.named_parameters():
                                with self.gather_if_zero3([p]):
                                    if accelerator.is_main_process:
                                        self.vllm_client.update_named_param(name+"."+n, p.data)
                if accelerator.is_main_process:
                    self.vllm_client.reset_prefix_cache()

                # gpu0                           gpu1
                # 1,2,3,4                        5,6,7,8
                # args.n=2
                # 1,2,3,4,1,2,3,4                5,6,7,8,5,6,7,8
                # vllm args.n=2
                # 1,1,2,2,3,3,4,4                5,5,6,6,7,7,8,8
                # index
                # 0,2,4,6,1,3,5,7 index
                vllm_prompts = [processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=rest["data_attribute"]=="CoT",) for prompt, rest  in zip(prompts[:len(data["prompts"].prompts)], rest_args[:len(data["prompts"].prompts)])]
                all_vllm_prompts = gather_object(vllm_prompts)
                if accelerator.is_main_process:
                    completion_ids = self.vllm_client.generate(
                        prompts=all_vllm_prompts,
                        n=args.m,
                        repetition_penalty=1.0,
                        temperature=(args.temperature + 1e-7),
                        top_p = 1.0,
                        top_k = 40,
                        max_tokens=args.response_length,
                    )
                else:
                    completion_ids = [None] * len(all_vllm_prompts) * args.m

                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(accelerator.process_index * len(vllm_prompts)*args.m, (accelerator.process_index + 1) * len(vllm_prompts)*args.m)
                completion_ids = completion_ids[process_slice]

                pre_scores=[]
                for i, (pre_prompt,pre_data_type,pre_rest_arg) in enumerate(zip(prompts[:len(data["prompts"].prompts)], data_types[:len(data["prompts"].prompts)], rest_args[:len(data["prompts"].prompts)])):
                    for j in range(args.m):
                        send_prompt = pre_prompt+[{"role": "assistant", "content": processing_class.decode(completion_ids[i*args.m+j][:-1])}]
                        score = get_reward_from_different_model(reward_model, [send_prompt[1:] if send_prompt[0]["role"]=="system" else send_prompt], [pre_data_type], [pre_rest_arg], processing_class)
                        pre_scores.append(score)
                pre_scores = torch.tensor(pre_scores).reshape(len(vllm_prompts), args.m)

                #Pre Compute grouped-wise rewards from all samples
                mean_grouped_rewards = pre_scores.mean(dim=1)
                std_grouped_rewards = pre_scores.std(dim=1)

                pre_scores_index = torch.argsort(pre_scores, dim=1)
                pre_scores_index = torch.cat((pre_scores_index[:,:args.rloo_k//2],pre_scores_index[:,-args.rloo_k//2:]),dim=1)
                pre_scores_index = (pre_scores_index+(torch.arange(len(vllm_prompts))*args.m).unsqueeze(-1)).flatten()
                completion_ids = [completion_ids[v] for v in pre_scores_index]
                pre_scores = pre_scores.flatten()[pre_scores_index]

                index = np.arange(len(completion_ids)).reshape(-1,args.rloo_k).T.flatten()
                responses_list = [completion_ids[v] for v in index]
                scores = [pre_scores[v] for v in index]
                query_responses = [q+r for q,r in zip(queries, responses_list)]

                for i in range(0, len(queries), args.local_rollout_forward_batch_size):
                    query = [torch.tensor(v, dtype=torch.int64).to(accelerator.device) for v  in queries[i : i + args.local_rollout_forward_batch_size]]
                    prompt = prompts[i : i + args.local_rollout_forward_batch_size]
                    data_type = data_types[i : i + args.local_rollout_forward_batch_size]
                    rest_arg = rest_args[i : i + args.local_rollout_forward_batch_size]
                    query_response = [torch.tensor(v, dtype=torch.int64).to(accelerator.device) for v in query_responses[i : i + args.local_rollout_forward_batch_size]]
                    context_length = context_lengths[i : i + args.local_rollout_forward_batch_size]
                    response = [qr[c:]for qr,c in zip(query_response, context_length)]
                    response_length = [len(v) for v in response]

                    with torch.no_grad():
                        policy_output = forward(accelerator.unwrap_model(model), pad(query_response, padding_value=processing_class.pad_token_id, padding_side="right"), processing_class.pad_token_id)
                        logits = [v[:,l - 1 : -1]/((args.temperature + 1e-7)) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1]/(args.temperature + 1e-7) for v,l in zip(policy_output.logits, context_length)]
                    all_logprob = [F.log_softmax(v[:,:rl], dim=-1) for v,rl in zip(logits, response_length)]
                    logprob = [torch.gather(v, 2, r.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0) for v,r in zip(all_logprob, response)]
                    del policy_output, logits, all_logprob
                    torch.cuda.empty_cache()

                    ref_logprob = get_ref_logp_vllm(ref_policy, query_response, processing_class.pad_token_id, context_length, args.temperature)
                    ref_logprob = [v.squeeze(0).to(dtype=logprob[0].dtype).to(device=logprob[0].device) for v in ref_logprob]

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = [truncate_response(args.stop_token_id, processing_class.pad_token_id, v.unsqueeze(0)).squeeze(0) for v in response]
                    #modify to generate padding token after the first stop token id
                    assert all([torch.all(pr==r)] for pr, r in zip(postprocessed_response, response))

                    # Response Processing 2. run reward model on the truncated responses
                    # Warning: sequence_length+1 must eq response_length, to deal with generate <pad> token error
                    sequence_length = [torch.tensor(v-1, dtype=torch.int64).to(accelerator.device) for v in response_length]
                    #sequence_length = [first_true_indices(v == processing_class.pad_token_id) - 1 for v in postprocessed_response]

                    responses+=response
                    postprocessed_responses+=postprocessed_response
                    logprobs+=logprob
                    ref_logprobs+=ref_logprob
                    sequence_lengths+=sequence_length
                    response_lengths+=response_length

                responses = pad(responses, padding_value=processing_class.pad_token_id, padding_side="right")
                postprocessed_responses = pad(postprocessed_responses, padding_value=processing_class.pad_token_id, padding_side="right")
                query_responses = [torch.tensor(v, dtype=torch.int64).to(accelerator.device) for v in query_responses]
                logprobs = pad(logprobs, padding_value=INVALID_LOGPROB, padding_side="right")
                ref_logprobs = pad(ref_logprobs, padding_value=INVALID_LOGPROB, padding_side="right")
                context_lengths = torch.tensor(context_lengths,dtype=torch.int64).to(accelerator.device)
                response_lengths = torch.tensor(response_lengths,dtype=torch.int64).to(accelerator.device)
                sequence_lengths = torch.cat([v.unsqueeze(0) for v in sequence_lengths], 0)
                scores = torch.tensor(scores).to(dtype=logprobs.dtype).to(device=logprobs.device)
                mean_grouped_rewards = mean_grouped_rewards.to(dtype=logprobs.dtype).to(device=logprobs.device)
                std_grouped_rewards = std_grouped_rewards.to(dtype=logprobs.dtype).to(device=logprobs.device)
                del (logprob, ref_logprob)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                #"""
                # Compute grouped-wise scores
                ori_scores = scores.detach()

                #Pre Compute grouped-wise rewards from all samples
                # Compute grouped-wise rewards
                #mean_grouped_rewards = scores.view(args.rloo_k, -1).mean(dim=0)
                #std_grouped_rewards = scores.view(args.rloo_k, -1).std(dim=0)

                # Normalize the rewards to compute the advantages
                advantages = (scores.reshape(args.rloo_k, -1) - mean_grouped_rewards.unsqueeze(0)) / (std_grouped_rewards.unsqueeze(0) + 1e-4)
                advantages = torch.clamp(advantages, -5.0, 5.0)
                advantages = advantages.flatten()

                """
                # vectorized RLOO advantages implementation
                rlhf_reward = scores.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = torch.clamp(advantages, -5.0, 5.0)
                advantages = advantages.flatten()
                """
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = [query_responses[v] for v in  list(micro_batch_inds)]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_ref_logprobs = ref_logprobs[micro_batch_inds]
                            context_length = context_lengths[micro_batch_inds]
                            response_length = response_lengths[micro_batch_inds]
                            sequence_length = sequence_lengths[micro_batch_inds]
                            output = forward(model, pad(mb_query_responses, padding_value=processing_class.pad_token_id, padding_side="right"), processing_class.pad_token_id)
                            logits = [v[:,l - 1 : -1]/((args.temperature + 1e-7)) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1]/(args.temperature + 1e-7) for v,l in zip(output.logits, context_length)]
                            new_all_logprobs = [F.log_softmax(v[:,:rl], dim=-1) for v,rl in zip(logits, response_length)]
                            new_logprob = [torch.gather(v, 2, r[:,:rl].unsqueeze(-1)).squeeze(-1).squeeze(0) if r.dim()==2 else torch.gather(v, 2, r[:rl].unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0) for v,r,rl in zip(new_all_logprobs, mb_responses, response_length)]
                            new_logprobs = pad(new_logprob, padding_value=INVALID_LOGPROB, padding_side="right")
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds][:,:torch.max(response_length)], INVALID_LOGPROB
                            )

                            negative_ref_logprobs_diff = mb_ref_logprobs[:,:torch.max(response_length)]-new_logprobs
                            negative_ref_logprobs_diff = torch.clamp(negative_ref_logprobs_diff, max=np.log(1e3), min=np.log(1e-3)) # similar to 10
                            per_token_kl = torch.exp(negative_ref_logprobs_diff)-(mb_ref_logprobs[:,:torch.max(response_length)]-new_logprobs)-1
                            #per_token_kl = torch.exp(mb_ref_logprobs[:,:torch.max(response_length)]-new_logprobs)-(mb_ref_logprobs[:,:torch.max(response_length)]-new_logprobs)-1
                            logprobs_diff = new_logprobs - mb_logprobs[:,:torch.max(response_length)]
                            #https://github.com/yfzhang114/r1_reward/blob/3693413abedd4a1efaa9484143acfaa5d78a8d2b/openrlhf/models/loss.py#L74
                            logprobs_diff = torch.clamp(logprobs_diff, max=np.log(1e3), min=np.log(1e-3)) # similar to 10
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage.unsqueeze(-1) * ratio
                            #Clip-Higher
                            pg_losses2 = -mb_advantage.unsqueeze(-1) * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange+0.00)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss_max = masked(pg_loss_max, ~padding_mask[micro_batch_inds][:,:torch.max(response_length)])
                            per_token_kl = masked(per_token_kl, ~padding_mask[micro_batch_inds][:,:torch.max(response_length)])
                            loss = ((pg_loss_max+args.kl_coef*per_token_kl).sum(dim=1)/torch.sum(~padding_mask[micro_batch_inds][:,:torch.max(response_length)],dim=1)).mean()

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(pad(logits, padding_value=0.0, padding_side="right"), dim=-1)
                                entropy = torch.logsumexp(pad(logits, padding_value=0.0, padding_side="right"), dim=-1) - torch.sum(prob_dist * pad(logits, padding_value=0.0, padding_side="right"), dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (pg_loss_max.sum(dim=1)/torch.sum(~padding_mask[micro_batch_inds],dim=1)).mean()
                                if torch.isnan(loss).any():
                                    print("=====================")
                                    print(mb_responses)
                                    print(padding_mask[micro_batch_inds])
                                    print(torch.isnan(mb_advantage).any())
                                    print(torch.isnan(ratio).any())
                                    print(torch.isnan(new_logprobs).any())
                                    print(torch.isnan(mb_logprobs[:,:torch.max(response_length)]).any())
                                    print("=====================")
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss_max, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs, mb_ref_logprobs
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = per_token_kl.sum(1).mean()
                mean_entropy = (-torch.masked_fill(logprobs, padding_mask, 0.0)).sum(1).mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["objective/response_length"] = self.accelerator.gather(response_lengths.float().mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

def convert_layer_to_quantlinear_layer(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, child_name, QuantLinear(child, child_name))
        else:
            convert_layer_to_quantlinear_layer(child)

def round_ste(x: torch.Tensor):
    #straight-through estimator (STE)
    return (x.round() - x).clone().detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    #straight-through estimator (STE)
    return (x.clamp(min,max) - x).clone().detach() + x

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale + zero), 0, maxq)
    return scale * (q - zero)

class QuantLinear(nn.Module):
    def __init__(self, m, name):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.sym=False
        self.mse=False
        self.norm=2.4
        self.grid=100
        self.maxshrink=.8
        self.bits=4
        self.qmin = 0
        self.qmax = 2 ** (self.bits) - 1
        self.group_size=128
        self.register_parameter('weight',m.weight) # trainable
        self.device = self.weight.device
        if m.bias is not None:
            self.register_parameter('bias',m.bias)
        else:
            self.bias = None
        self.d_in = m.in_features
        self.d_out = m.out_features
        self.name = name
        self.gather_if_zero3 = deepspeed.zero.GatheredParameters if is_deepspeed_zero3_enabled() else nullcontext

        with self.gather_if_zero3([self.weight]):
        # init scale and zero point through Q4-0 quantization
            with torch.no_grad():
                if self.weight is not None:
                    x = self.weight.reshape(-1,self.group_size)
                    tmp = torch.zeros(x.shape[0], device=self.device)
                    xmin = torch.minimum(x.min(1)[0], tmp)
                    xmax = torch.maximum(x.max(1)[0], tmp)

                    if self.sym:
                        xmax = torch.maximum(torch.abs(xmin), xmax)
                        tmp = xmin < 0
                        if torch.any(tmp):
                            xmin[tmp] = -xmax[tmp]
                    tmp = (xmin == 0) & (xmax == 0)
                    xmin[tmp] = -1
                    xmax[tmp] = +1
                    scales = (xmax - xmin) / self.qmax
                    if self.sym:
                        zeros = torch.full_like(scales, (self.qmax + 1) / 2)
                    else:
                        zeros = -xmin / scales
                if self.mse:
                    best = torch.full([x.shape[0]], float('inf'), device=x.device)
                    for i in range(int(self.maxshrink * self.grid)):
                        p = 1 - i / self.grid
                        xmin1 = p * xmin
                        xmax1 = p * xmax
                        scale1 = (xmax1 - xmin1) / self.qmax
                        zero1 = -xmin1 / scale1 if not self.sym else zeros
                        q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.qmax)
                        q -= x
                        q.abs_()
                        q.pow_(self.norm)
                        err = torch.sum(q, 1)
                        tmp = err < best
                        if torch.any(tmp):
                            best[tmp] = err[tmp]
                            scales[tmp] = scale1[tmp]
                            zeros[tmp] = zero1[tmp]

            self.register_buffer('min', torch.nn.Parameter(0.5*scales.unsqueeze(1).to(self.weight.dtype)))
            self.register_buffer('max', torch.nn.Parameter(2.0*scales.unsqueeze(1).to(self.weight.dtype)))
            #second step fix scales
            #self.register_buffer('scales', torch.nn.Parameter(scales.unsqueeze(1).to(self.weight.dtype)))
            #self.register_buffer('zeros', torch.nn.Parameter(zeros.unsqueeze(1).to(self.weight.dtype)))
            #first step not fix scales, zeros same as fixed
            self.register_parameter('scales', torch.nn.Parameter(scales.unsqueeze(1).to(self.weight.dtype)))
            self.register_parameter('zeros', torch.nn.Parameter(zeros.unsqueeze(1).to(self.weight.dtype)))
            self.register_parameter('delta_zeros', torch.nn.Parameter(torch.zeros_like(self.zeros).to(self.weight.dtype)))
        deepspeed.zero.register_external_parameter(self, self.scales)
        deepspeed.zero.register_external_parameter(self, self.zeros)
        deepspeed.zero.register_external_parameter(self, self.delta_zeros)

    def fake_quant(self, x):
        scale = clamp_ste(self.scales, self.min, self.max)
        round_zero_point = self.zeros

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x = x / scale
        if round_zero_point is not None:
            x_int = round_ste(x.add(round_zero_point))
        x_int = clamp_ste(x_int, self.qmin, self.qmax)
        #x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
            scale_delta_zeros = clamp_ste(self.delta_zeros,-scale,scale)
        x_dequant = x_dequant.mul(scale)-scale_delta_zeros
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    
    def qat_load_init_quant(self):
        with self.gather_if_zero3([self.weight, self.min, self.max, self.zeros, self.scales, self.delta_zeros]):
            with torch.no_grad():
                self.scales.data = self.scales.clamp(self.min, self.max)
                dim1, dim2 = self.weight.shape
                x= self.weight.reshape(-1, self.group_size)
                x_int = (x / self.scales+self.zeros).round()
                x_int = x_int.clamp(self.qmin, self.qmax)
                x_dequant = x_int.sub(self.zeros)
                self.delta_zeros.data = self.delta_zeros.clamp(-self.scales,self.scales)
                x_dequant = x_dequant.mul(self.scales)-self.delta_zeros
                if self.group_size:
                    self.weight.data = x_dequant.reshape(dim1, dim2)

    def forward(self, *args, **kwargs):
        weight = self.fake_quant(self.weight)
        if self.name=="lm_head":
            #embedding weight and lm_head tie
            self.weight.data = weight
        bias = self.bias
        return  self.fwd_func(args[0], weight, bias,  **self.fwd_kwargs)

def model_qat_load_init(model):
    for child_name, child in model.named_children():
        if isinstance(child, QuantLinear):
            child.qat_load_init_quant()
        else:
            model_qat_load_init(child)

def _load_state_dict_into_zero3_model(model_to_load, state_dict):
    """
    Loads state dict into a model specifically for Zero3, since DeepSpeed does not support the `transformers`
    tensor parallelism API.

    Nearly identical code to PyTorch's `_load_from_state_dict`
    """
    # copy state_dict so `_load_state_dict_into_zero3_model` can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if is_deepspeed_zero3_enabled() and len([key for key in state_dict if key.startswith(prefix)]) > 0:
            import deepspeed

            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, assign_to_params_buffers=False)

    return error_msgs

@dataclass
class RLOOConfig_add(RLOOConfig):
    m: Optional[int] = field(default=False)
    ref_model_path: Optional[str] = field(default=False)
    gptq_path: Optional[str] = field(default=None)
    load_from_gptq: Optional[bool] = field(default=False)
    qat_path: Optional[str] = field(default=None)
    load_from_qat: Optional[bool] = field(default=False)

if __name__ == "__main__":
    global local_rank
    parser = HfArgumentParser((ScriptArguments, RLOOConfig_add, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # remove output_dir if exists
    resume_from_checkpoint=True
    if not resume_from_checkpoint:
        pass
        #shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path, padding_side="left", trust_remote_code=False
    )
    tokenizer.eos_token_id = 151645
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)

    reward_model = f"http://{training_args.reward_model_path}:8000/reward"
    ref_policy = f"http://{training_args.ref_model_path}:8000/ref_model"
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map=None
    )

    convert_layer_to_quantlinear_layer(policy)

    if training_args.load_from_gptq==True:
        state_dict = {}
        for i in range(1, 3):
            file_path = os.path.join(training_args.gptq_path, f"model-0000{i}-of-00002.safetensors")
            data = load_file(file_path)
            state_dict.update(data)
        rank0_print(f"Lack weight {set(list(policy.state_dict().keys()))-set(list(state_dict.keys()))}")
        if is_deepspeed_zero3_enabled():
            _load_state_dict_into_zero3_model(policy, state_dict)
        else:
            policy.load_state_dict(state_dict, strict=len(state_dict.keys())==len(policy.state_dict().keys()))
        model_qat_load_init(policy)
        policy=policy.to(torch.bfloat16)
        rank0_print("Load from gptq model succeed")

    if training_args.load_from_qat==True:
        state_dict = {}
        for i in range(1, 3):
            file_path = os.path.join(training_args.qat_path, f"model-0000{i}-of-00002.safetensors")
            data = load_file(file_path)
            state_dict.update(data)
        unexpected_keys = set(list(state_dict.keys())) - set(list(policy.state_dict().keys()))
        assert len(unexpected_keys)==0

        if is_deepspeed_zero3_enabled():
            _load_state_dict_into_zero3_model(policy, state_dict)
        else:
            policy.load_state_dict(state_dict)

        model_qat_load_init(policy)
        policy=policy.to(torch.bfloat16)
        rank0_print("Load from qat model succeed")

    if training_args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    ################
    # Dataset
    ################
    set_seed(training_args.seed)
    with PartialState().local_main_process_first():
        train_dataset = MixedDataset(script_args.dataset_name, tokenizer, 1024)
    ################
    # Training
    ################
    trainer = CustomizedGrpoTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
    )

    #for name, p in policy.named_parameters():
    #    rank0_print(name, p.dtype, p.requires_grad)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
