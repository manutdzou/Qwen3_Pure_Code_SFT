from trl.extras.vllm_client import VLLMClient
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example usage
if __name__ == "__main__":

    vllm_ip = "0.0.0.0"
    client = VLLMClient(host=vllm_ip)
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B-250424")

    sample_para = SamplingParams(temperature=0.7, top_p=1.0, top_k=40, max_tokens=1024, repetition_penalty=1.0)

    messages = [{"role": "user", "content": "一个直角三角形，两条边分别为6和8，求第三条边，请列出所有可能？"},]
    messages2 = [{"role": "user", "content": "一个平行四边形，两条边分别为6和8，求最大和最小面积，请列出所有可能？"},]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    text2 = tokenizer.apply_chat_template(
        messages2,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    # Generate completions
    generated_ids = client.generate([text,text2], n=1, 
            repetition_penalty=sample_para.repetition_penalty, 
            temperature=sample_para.temperature, 
            top_p=sample_para.top_p, 
            max_tokens=sample_para.max_tokens)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    print("Responses:", response)  # noqa

    # Update model weights
    """
    model = AutoModelForCausalLM.from_pretrained("Qwen3-8B-250424").to("cuda")

    client.update_model_params(model)
    generated_ids = client.generate([text], n=4, 
            repetition_penalty=sample_para.repetition_penalty, 
            temperature=sample_para.temperature, 
            top_p=sample_para.top_p, 
            max_tokens=sample_para.max_tokens)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Responses:", response)  # noqa
    # Generate completions
    generated_ids = client.generate([text,text2], n=4, 
            repetition_penalty=sample_para.repetition_penalty, 
            temperature=sample_para.temperature, 
            top_p=sample_para.top_p, 
            max_tokens=sample_para.max_tokens)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Responses:", response)  # noqa

    # Update model weights

    model = AutoModelForCausalLM.from_pretrained("Qwen3-8B-250424").to("cuda")

    client.update_model_params(model)
    generated_ids = client.generate([text], n=4, 
            repetition_penalty=sample_para.repetition_penalty, 
            temperature=sample_para.temperature, 
            top_p=sample_para.top_p, 
            max_tokens=sample_para.max_tokens)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Responses:", response)  # noqa
    """
