import json,os, random
import re
from typing import List
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from tqdm import tqdm

class VerifierFunction(ABC):
    """
    Abstract base class for verifier functions.

    Each verifier is initialized with a name and a weight (default 1.0).
    The __call__ method must be implemented by subclasses.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def __call__(self, tokenized_prediction: List[int], prediction: str, label: Any) -> float:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.

        Returns:
            int: Reward score. Can be binary (0/1) or continuous.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self) -> None:
        super().__init__("gsm8k", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> float:
        prediction = prediction.split("<|im_end|>")[0].split("</think>\n\n")[-1].strip()
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        return float(str(extracted).lower() == str(label).lower())


def get_json_list2(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    model_name = "/dfs/dataset/770-1740731284808/data/Qwen2.5-7B-QAT-test"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    reward_function = GSM8KVerifier()

    #datas = get_json_list2("gsm8k_en_R1.json")
    datas = get_json_list2("gsm8k_zh_R1.json")
    random.shuffle(datas)
    for data in tqdm(datas):
        messages = data["prompt"]
        label = data["answer_only"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generation_config = GenerationConfig(
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            do_sample=True,
        )

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            generation_config=generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        score = reward_function(generated_ids[0].tolist(), response, json.loads(label))
       
        print(score)
