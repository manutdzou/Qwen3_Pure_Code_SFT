from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
import uvicorn
from lerm_27b import LDLRewardModel27B
# 初始化 FastAPI
app = FastAPI()

# 模型路径
MODEL_NAME = "Lenovo-Reward-v2-Gemma-2-27B"  # 替换为实际模型路径

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 使用 accelerate 在 GPU 0 和 GPU 1 上加载模型
print("Loading model...")
model = LDLRewardModel27B.from_pretrained(MODEL_NAME, device_map='auto')
model.eval()
print(model)

print("Model loaded!")

# 定义请求体
class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str

class Request(BaseModel):
    msg: list[Message]

@app.post("/reward")
async def generate_text(request: Request):
    try:
        inputs = request.msg
        inputs_conv_tokenized =  tokenizer.apply_chat_template(inputs, tokenize=True, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs_conv_tokenized.to(0)


        # 3. 生成文本
        with torch.inference_mode():
            output = model(input_ids=input_ids)

        score = output['score'].float().cpu().numpy().tolist()[0]

        # return {"score": score,
        #         "input": inputs}
        print(inputs, score)
        return {"score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
