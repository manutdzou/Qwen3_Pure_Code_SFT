# A purely simply code for Qwen3 Post Training

## SFT

SFT Qwen3 Instruct model with mixture CoT and NonCoT data, Please refer to sft.json to see the training data format. We have tested on 1.7B/4B/8B model with one node 8-A100 server. 

```
sh train_sft.sh
```

## QAT(Quantization Aware training)

Support QAT linear layers and embedding layers use Q4-1 method.

```
sh train_sft_qat.sh
```

## Reinforcement Learning & Quantization

Support PPO and grpo, mix rule based and neural based reward, especially explored the application of reinforcement learning in model quantization.

One gpu for vllm inference, 7 gpus for training and 1 client to get reward scores, 1 client to get ref model logp.

```
vll.sh
python get_reward_27B_score.py
python get_ref_logp_vllm.py
sh train_fp16ppo_multi_reward.sh
sh train_qat_grpo_multi_reward.sh
```