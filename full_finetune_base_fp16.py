import copy
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
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
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        device_map=None,
        #low_cpu_mem_usage=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side='right'
    )
    tokenizer.eos_token_id = 151645
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    
   # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    trainer.save_state(resume_from_checkpoint=False)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) #avoid checkpoint load problem

if __name__ == "__main__":
    train()
