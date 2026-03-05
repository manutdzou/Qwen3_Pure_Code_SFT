from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import transformers
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
from trl.trainer.utils import pad
import uvicorn
import traceback
# 初始化 FastAPI
app = FastAPI()

#Warning: must be fp16 or bf16 model not QAT model
MODEL_NAME = "Qwen3-4B-250426"  # 替换为实际模型路径
#MODEL_NAME = "Qwen3-4B-GPTQ-dequant"  # 反量化模型bf16形式
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()
print(model)

print("Model loaded!")

# 定义请求体
class Message(BaseModel):
    query_responses: list[list]
    pad_token: int
    context_length: list[int]
    temperature: float

class Request(BaseModel):
    msg: Message

@app.post("/ref_model")
async def generate_text(request: Request):
    try:
        inputs = request.msg
        query_responses = [torch.tensor(v, dtype=torch.int64).to(0) for v in inputs.query_responses]
        context_length = inputs.context_length
        response = [qr[c:]for qr,c in zip(query_responses, context_length)]
        response_length = [len(v) for v in response]
        query_responses = pad(query_responses, padding_value=tokenizer.pad_token_id, padding_side="right")
        pad_token_id = inputs.pad_token
        temperature = inputs.temperature

        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        with torch.inference_mode():
            ref_output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True, output_hidden_states=True,)
            ref_logits = [v[:,l - 1 : -1]/((temperature + 1e-7)) if v.dim()==3 else v.unsqueeze(0)[:,l - 1 : -1]/(temperature + 1e-7) for v,l in zip(ref_output.logits, context_length)]
            ref_all_logprob = [F.log_softmax(v[:,:rl], dim=-1) for v,rl in zip(ref_logits, response_length)]
            ref_logprob = [torch.gather(v, 2, r.unsqueeze(0).unsqueeze(-1)).squeeze(-1) for v,r in zip(ref_all_logprob, response)]

        ref_logprob = [v.float().cpu().numpy().tolist() for v in ref_logprob]

        return {"ref_logprob": ref_logprob}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
