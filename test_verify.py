model_path="decapoda-research/llama-13b-hf"
lora_path="./lora-Vicuna"


import sys
import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained(model_path)
BASE_MODEL = model_path

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
inputs = "Hello, Where is the capital of the United States?" #"你好,美国的首都在哪里？"
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
print(input_ids)
generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=15,
        )
print(generation_output)
print(tokenizer.decode(generation_output[0]))




model = PeftModel.from_pretrained(
        model,
        lora_path,
        #"/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco/",
        torch_dtype=torch.float16,
        device_map={'': 0}
    )

inputs = "你好,中国的首都在哪里？" #"你好,美国的首都在哪里？"
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
print(input_ids)
generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=15,
        )
print(generation_output)
print(tokenizer.decode(generation_output[0]))
