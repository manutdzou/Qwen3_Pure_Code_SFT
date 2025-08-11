import copy
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import io
import json
from torch.utils.data import Dataset
from transformers import Trainer

from utils import get_json_list2
import random

IGNORE_INDEX = -100

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    gptq_path: Optional[str] = field(default=None)
    load_from_gptq: Optional[bool] = field(default=False)
    qat_path: Optional[str] = field(default=None)
    load_from_qat: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    start_token_id: int,
    end_token_id: int,
    begin_think_token_id: int,
    end_think_token_id: int,
) -> Dict:
    input_ids, targets = [], []
    contents = []
    for i, source in enumerate(sources):
        add_generation_prompt=source["data_attribute"]=="CoT"
        source = source["prompt"]
        content = tokenizer.apply_chat_template(
            source,
            padding=False,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=add_generation_prompt
        )
        input_id = tokenizer.apply_chat_template(
            source,
            padding=False,
            add_generation_prompt=False,
            enable_thinking=add_generation_prompt
        )
        prompt_id = tokenizer.apply_chat_template(
            source[:-1],
            padding=False,
            add_generation_prompt=True,
            enable_thinking=add_generation_prompt
        )

        target = [IGNORE_INDEX]*len(prompt_id)+input_id[len(prompt_id):]
        for index, token_id in enumerate(input_id):
            if token_id==start_token_id or token_id==end_token_id:
                target[index]=token_id

        target = target[:-1]
        input_id = input_id[:-1]
        content = content[:-1]

        assert len(input_id) == len(target)
        assert torch.sum(torch.tensor(tokenizer.encode(content))-torch.tensor(input_id))==0 ##some case like assistant\n\n not eq
        input_ids.append(torch.tensor(input_id[:max_len],dtype=torch.int64))
        targets.append(torch.tensor(target[:max_len],dtype=torch.int64))
        contents.append(content)

    attention_mask = [v.ne(tokenizer.pad_token_id) for v in input_ids]
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.start_token_id = 151644
        self.end_token_id = 151645
        self.begin_think_token_id = 151667
        self.end_think_token_id = 151668

        logging.warning("Loading data...")
        list_data_dict = get_json_list2(data_path)
        list_data_dict = [v for v in list_data_dict if len(tokenizer.apply_chat_template(v["prompt"], padding=False, add_generation_prompt=False, enable_thinking=v["data_attribute"]=="CoT"))<=max_len]

        logging.warning("Formatting inputs...")
        self.sources = list_data_dict
        random.shuffle(self.sources)
        logging.warning("Tokenizing inputs... This may take some time...")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        ret = preprocess([self.sources[i]], self.tokenizer, self.max_len, self.start_token_id, self.end_token_id, self.begin_think_token_id, self.end_think_token_id)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        return ret


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_len=training_args.model_max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def convert_layer_to_quantlinear_layer(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, child_name, QuantLinear(child, child_name))
        elif isinstance(child, nn.Embedding):
            setattr(model, child_name, QuantEmbedding(child, child_name))
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
        self.name = name

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

    def forward(self, *args, **kwargs):
        weight = self.fake_quant(self.weight)
        bias = self.bias
        out = self.fwd_func(args[0], weight, bias,  **self.fwd_kwargs)
        return out

    def qat_load_init_quant(self):
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

class QuantEmbedding(nn.Module):
    def __init__(self, m, name):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.embedding
        self.sym=False
        self.mse=False
        self.norm=2.4
        self.grid=100
        self.maxshrink=.8
        self.bits=4
        self.qmin = 0
        self.qmax = 2 ** (self.bits) - 1
        self.group_size=128
        #to deal with bf16 overflow, nan problem
        m.weight.data =torch.where(m.weight.abs()>1e-8, m.weight, torch.zeros_like(m.weight.data, device=m.weight.device).normal_(mean=0.0, std=0.02))
        self.register_parameter('weight',m.weight) # trainable
        self.device = self.weight.device
        self.d_in = m.num_embeddings
        self.d_out = m.embedding_dim

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
                xmin[tmp] = -0.06
                xmax[tmp] = +0.06
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
        self.name = name

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

    def forward(self, *args, **kwargs):
        weight = self.fake_quant(self.weight)
        #weight = self.weight
        out = self.fwd_func(args[0], weight,  **self.fwd_kwargs)
        return out

    def qat_load_init_quant(self):
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

def model_qat_load_init(model):
    for child_name, child in model.named_children():
        if isinstance(child, QuantLinear):
            child.qat_load_init_quant()
        elif isinstance(child, QuantEmbedding):
            child.qat_load_init_quant()
        else:
            model_qat_load_init(child)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #model = transformers.AutoModelForCausalLM.from_pretrained(
    #    model_args.model_name_or_path,
    #    torch_dtype='auto',
    #    cache_dir=training_args.cache_dir,
    #    device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}#if not set, cannot load model
    #)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        device_map=None,
        #low_cpu_mem_usage=True
    )

    convert_layer_to_quantlinear_layer(model)

    if training_args.load_from_gptq==True:
        state_dict = torch.load(training_args.gptq_path, weights_only=True)
        unexpected_keys = set(list(state_dict.keys())) - set(list(model.state_dict().keys()))
        assert len(unexpected_keys)==0
        model.load_state_dict(state_dict)
        model=model.to(torch.bfloat16)
        rank0_print("Load from gptq model succeed")

    if training_args.load_from_qat==True:
        state_dict = {}
        for i in range(1, 3):
            file_path = os.path.join(training_args.qat_path, f"model-0000{i}-of-00002.safetensors")
            data = load_file(file_path)
            state_dict.update(data)
        unexpected_keys = set(list(state_dict.keys())) - set(list(model.state_dict().keys()))
        assert len(unexpected_keys)==0
        model.load_state_dict(state_dict)
        model_qat_load_init(model)
        model=model.to(torch.bfloat16)
        rank0_print("Load from qat model succeed")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side='right'
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    
   # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    #trainer.train(resume_from_checkpoint=True)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) #avoid checkpoint load problem
    #trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
