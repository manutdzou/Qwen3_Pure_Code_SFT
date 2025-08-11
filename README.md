# A purely simply code for Qwen3 Post Training

## SFT

SFT Qwen3 Instruct model with mixture CoT and NonCoT data, Please refer to sft.json to see the training data format. We have tested on 1.7B/4B/8B model with one node 8-A100 server. 

```
sh train_sft.sh
```

## QAT(Quantization Aware training)

Support QAT linear layers and embedding layers use Q4-1 method.

## Reinforcement Learning

Coming soon...