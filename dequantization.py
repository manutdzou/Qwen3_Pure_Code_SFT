import os
import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file, save_file

#model_path = "Qwen3-4B-QAT-heavy-cold-start"
#model_path = "Qwen3-4B-QAT-lite-cold-start"
#model_path = "/dfs/data/LLM/Qwen3_4B/Qwen3-4B-GPTQ"
#model_path = "/dfs/data/LLM/Qwen3_4B/grpo_tldr_multi_reward_qat_very3_M"
model_path = "/dfs/data/LLM/Qwen3_4B/ppo_tldr_multi_reward_share_policy_value_modify_score_vllm_M"
model = {}
for i in range(1, 3):
    file_path = os.path.join(model_path, f"model-0000{i}-of-00002.safetensors")
    data = load_file(file_path)
    model.update(data)

def fake_quant(x, scales, zeros, min_scale, max_scale, scale_delta_zeros, group_size, bit):
    qmin = 0
    qmax = 2**bit-1
    dim1, dim2 = x.shape
    x = x.reshape(-1, group_size)
    scales = scales.clamp(min_scale, max_scale)
    x_int = (x / scales+ zeros).round()
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = x_int
    if zeros is not None:
        x_dequant = x_dequant.sub(zeros)
    scale_delta_zeros=scale_delta_zeros.clamp(-scales,scales)
    x_dequant = x_dequant.mul(scales)-scale_delta_zeros
    if group_size:
        x_dequant = x_dequant.reshape(dim1, dim2)
    return x_dequant

save_model={}
dtype = torch.bfloat16
for key in model:
    if key.endswith(".weight") and len(model[key].shape)==2 and ".".join(key.split(".")[:-1]+["zeros"]) in model:
        name = ".".join(key.split(".")[:-1])
        zeros = model[name+".zeros"].to(dtype)
        scales = model[name+".scales"].to(dtype)
        weight = model[key].to(dtype)
        min_scale = model[name+".min"].to(dtype)
        max_scale = model[name+".max"].to(dtype)
        delta_zeros = model[name+".delta_zeros"].to(dtype)
        weight=fake_quant(weight, scales, zeros, min_scale, max_scale, delta_zeros, bit=4, group_size=128)
        print("Quant==========", key)
        save_model[key]=weight
    elif key.endswith(".scales") or key.endswith(".zeros") or key.endswith(".min") or key.endswith(".max") or key.endswith(".delta_zeros"):
        pass
    else:
        print(key)
        save_model[key]=model[key].to(dtype)
print((save_model["model.embed_tokens.weight"]-save_model["lm_head.weight"]).abs().max())
save_model["model.embed_tokens.weight"]=save_model["lm_head.weight"].clone()

import json
from safetensors.torch import save_file as safe_save_file
from huggingface_hub import split_torch_state_dict_into_shards

def save_state_dict(state_dict, save_directory):
    state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size= '5GB')
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        safe_save_file(
            shard,
            os.path.join(save_directory, filename),
            metadata={"format": "pt"},
        )
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))


save_state_dict(save_model, "Qwen3-4B-QAT-test")
print("done")
