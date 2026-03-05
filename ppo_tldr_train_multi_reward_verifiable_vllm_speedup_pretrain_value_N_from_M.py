import shutil,time, math, os, types, requests, gc, copy, json, random, re
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    DataCollatorWithPadding,
    TrainerControl,
    Trainer,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
)
from accelerate.utils import set_seed
from trl.trainer.ppo_trainer import PolicyAndValueWrapper, INVALID_LOGPROB
#from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import forward, truncate_response, first_true_indices, batch_generation, OnlineTrainerState, exact_div, disable_dropout_in_model,prepare_deepspeed, pad
from verifiable_lib.reward_func import IFEvalVerifier
from verifiable_lib.gsm8k_reward_func import GSM8KVerifier
from verifiable_lib.ifeval_reward_func import run
ifeval_reward_function = IFEvalVerifier()
gsm8k_reward_function = GSM8KVerifier()

import deepspeed
from contextlib import nullcontext
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.extras.vllm_client import VLLMClient

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

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            bessel_correction = mask_sum / mask_sum
        else:
            bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

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

def get_values(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    return reward_logits

class CustomizedPPOTrainer(PPOTrainer):
    def __init__(
        self,
        config,
        processing_class,
        policy,
        ref_policy,
        reward_model,
        train_dataset,
        value_model,
        data_collator=None,
        eval_dataset=None,
        # less commonly used
        optimizers=(None, None),
        callbacks=None,):
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
        self.value_model = value_model
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
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )*args.n  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.n, "`local_batch_size` must be a multiple of n"
            )

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, value_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.model.config = policy.config  # needed for pushing to hub
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

        rank0_print("args.train_dataset_len:",self.train_dataset_len)
        rank0_print("args.world_size:",args.world_size)
        rank0_print("args.local_batch_size:",args.local_batch_size)
        rank0_print("args.num_mini_batches:",args.num_mini_batches)
        rank0_print("args.batch_size:",args.batch_size)
        rank0_print("args.mini_batch_size:",args.mini_batch_size)
        rank0_print("args.local_mini_batch_size:",args.local_mini_batch_size)
    
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        self.gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

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
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
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
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"]*args.n
                context_lengths = data["lengths"]*args.n
                prompts = [copy.deepcopy(v) for v in data["prompts"].prompts*args.n]
                data_types = data["prompts"].data_types*args.n
                rest_args = data["prompts"].kwargs*args.n
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                sequence_lengths = []
                response_lengths = []
                values = []

                accelerator.wait_for_everyone()
                for name, param in accelerator.unwrap_model(model).policy.named_parameters():
                    with self.gather_if_zero3([param]):
                        if accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
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
                #sample top N from M completions
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
                pre_scores_index = torch.cat((pre_scores_index[:,:args.n//2],pre_scores_index[:,-args.n//2:]),dim=1)
                pre_scores_index = (pre_scores_index+(torch.arange(len(vllm_prompts))*args.m).unsqueeze(-1)).flatten()
                completion_ids = [completion_ids[v] for v in pre_scores_index]
                pre_scores = pre_scores.flatten()[pre_scores_index]
                
                index = np.arange(len(completion_ids)).reshape(-1,args.n).T.flatten()
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
                        policy_output = forward(accelerator.unwrap_model(model).policy, pad(query_response, padding_value=processing_class.pad_token_id, padding_side="right"), processing_class.pad_token_id)
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
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value = get_values(
                        unwrapped_value_model, pad(query_response, padding_value=processing_class.pad_token_id, padding_side="right"), processing_class.pad_token_id
                    )
                    value = [v[:,l - 1 : -1][:,:s+1].squeeze(-1).squeeze(0) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1][:,:s+1].squeeze(-1).squeeze(0) for v,l,s in zip(full_value, context_length, sequence_length)]

                    responses+=response
                    postprocessed_responses+=postprocessed_response
                    logprobs+=logprob
                    ref_logprobs+=ref_logprob
                    sequence_lengths+=sequence_length
                    response_lengths+=response_length
                    values+=value

                 
                responses = pad(responses, padding_value=processing_class.pad_token_id, padding_side="right")
                postprocessed_responses = pad(postprocessed_responses, padding_value=processing_class.pad_token_id, padding_side="right")
                query_responses = [torch.tensor(v, dtype=torch.int64).to(accelerator.device) for v in query_responses]
                logprobs = pad(logprobs, padding_value=INVALID_LOGPROB, padding_side="right")
                ref_logprobs = pad(ref_logprobs, padding_value=INVALID_LOGPROB, padding_side="right")
                context_lengths = torch.tensor(context_lengths,dtype=torch.int64).to(accelerator.device)
                response_lengths = torch.tensor(response_lengths,dtype=torch.int64).to(accelerator.device)
                sequence_lengths = torch.cat([v.unsqueeze(0) for v in sequence_lengths], 0)
                values = pad(values, padding_value=0.0, padding_side="right")
                scores = torch.tensor(scores).to(dtype=values.dtype).to(device=values.device)
                mean_grouped_rewards = mean_grouped_rewards.to(dtype=logprobs.dtype).to(device=logprobs.device)
                std_grouped_rewards = std_grouped_rewards.to(dtype=logprobs.dtype).to(device=logprobs.device)
                del (logprob, ref_logprob, full_value, value)
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
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)
                #"""
                # Compute grouped-wise scores
                ori_scores = scores.detach()
                # Normalize the rewards to compute the advantages
                scores = (scores.reshape(args.n, -1) - mean_grouped_rewards.unsqueeze(0)) / (std_grouped_rewards.unsqueeze(0) + 1e-4)
                scores = torch.clamp(scores, -5.0, 5.0)
                scores = scores.flatten()

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
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
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            context_length = context_lengths[micro_batch_inds]
                            response_length = response_lengths[micro_batch_inds]
                            sequence_length = sequence_lengths[micro_batch_inds]
                            output, vpred_temp = forward(model, pad(mb_query_responses, padding_value=processing_class.pad_token_id, padding_side="right"), processing_class.pad_token_id)
                            logits = [v[:,l - 1 : -1]/((args.temperature + 1e-7)) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1]/(args.temperature + 1e-7) for v,l in zip(output.logits, context_length)]
                            new_all_logprobs = [F.log_softmax(v[:,:rl], dim=-1) for v,rl in zip(logits, response_length)]
                            new_logprob = [torch.gather(v, 2, r[:,:rl].unsqueeze(-1)).squeeze(-1).squeeze(0) if r.dim()==2 else torch.gather(v, 2, r[:rl].unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0) for v,r,rl in zip(new_all_logprobs, mb_responses, response_length)]
                            new_logprobs = pad(new_logprob, padding_value=INVALID_LOGPROB, padding_side="right")
                            new_logprobs = torch.masked_fill(
                                    new_logprobs, padding_mask[micro_batch_inds][:,:torch.max(response_length)], INVALID_LOGPROB
                            )

                            vpred = [v[:,l - 1 : -1][:,:s+1].squeeze(-1).squeeze(0) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1][:,:s+1].squeeze(-1).squeeze(0) for v,l,s in zip(vpred_temp, context_length, sequence_length)]
                            vpred = pad(vpred, padding_value=0.0, padding_side="right")
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds][:,:torch.max(response_length)], 0)

                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values[:,:torch.max(response_length)] - args.cliprange_value,
                                mb_values[:,:torch.max(response_length)] + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return[:,:torch.max(response_length)])
                            vf_losses2 = torch.square(vpredclipped - mb_return[:,:torch.max(response_length)])
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds][:,:torch.max(response_length)])
                            vf_clipfrac = masked_mean(
                                    (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds][:,:torch.max(response_length)]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs[:,:torch.max(response_length)]
                            #https://github.com/yfzhang114/r1_reward/blob/3693413abedd4a1efaa9484143acfaa5d78a8d2b/openrlhf/models/loss.py#L74
                            logprobs_diff = torch.clamp(logprobs_diff, max=np.log(1e3), min=np.log(1e-3)) # similar to 10

                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage[:,:torch.max(response_length)] * ratio
                            #Clip-Higher
                            pg_losses2 = -mb_advantage[:,:torch.max(response_length)] * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange+0.08)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds][:,:torch.max(response_length)])
                            loss = pg_loss + args.vf_coef * vf_loss
                            
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                        (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds][:,:torch.max(response_length)]
                                )
                                prob_dist = torch.nn.functional.softmax(pad(logits, padding_value=0.0, padding_side="right"), dim=-1)
                                entropy = torch.logsumexp(pad(logits, padding_value=0.0, padding_side="right"), dim=-1) - torch.sum(prob_dist * pad(logits, padding_value=0.0, padding_side="right"), dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                if torch.isnan(loss).any():
                                    print("=====================")
                                    print(mb_responses)
                                    print(padding_mask[micro_batch_inds])
                                    print(torch.isnan(mb_advantage).any())
                                    print(mb_advantage)
                                    print(torch.isnan(ratio).any())
                                    print(torch.isnan(new_logprobs).any())
                                    print(torch.isnan(mb_logprobs[:,:torch.max(response_length)]).any())
                                    print("=====================")
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-torch.masked_fill(logprobs, padding_mask, 0.0)).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(ori_scores.mean()).mean().item()
                metrics["objective/response_length"] = self.accelerator.gather(response_lengths.float().mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

            with self.gather_if_zero3(self.model.value_model.parameters()):
                if accelerator.is_main_process and update%self.args.save_steps==0:
                    state_dict = self.model.value_model.state_dict()
                    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
                    del state_dict
                    torch.save(cpu_state_dict, os.path.join(self.args.output_dir,f"value_{update}.pt"))
                    del cpu_state_dict
                    torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

@dataclass
class PPOConfig_add(PPOConfig):
    n: Optional[int] = field(default=False)
    m: Optional[int] = field(default=False)
    ref_model_path: Optional[str] = field(default=False)

if __name__ == "__main__":
    global local_rank
    parser = HfArgumentParser((ScriptArguments, PPOConfig_add, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

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
    value_model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen3-4B-250426", trust_remote_code=True, num_labels=1, torch_dtype=torch.bfloat16, device_map=None
    )
    if training_args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        value_model.gradient_checkpointing_enable()

    rank0_print("Load value model succeed")

    ################
    # Dataset
    ################
    set_seed(training_args.seed)
    with PartialState().local_main_process_first():
        train_dataset = MixedDataset(script_args.dataset_name, tokenizer, 1024)

    ################
    # Training
    ################
    trainer = CustomizedPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
    )

    #"""
    for name, p in policy.named_parameters():
        rank0_print(name, p.dtype, p.requires_grad)
    #for name, p in value_model.named_parameters():
    #    rank0_print(name, p.dtype, p.requires_grad)
    #"""
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    # Save value model
    with trainer.gather_if_zero3(trainer.model.value_model.parameters()):
        if trainer.args.local_rank == 0:
            state_dict = trainer.model.value_model.state_dict()
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            torch.save(cpu_state_dict, os.path.join(training_args.output_dir,"value.pt"))
