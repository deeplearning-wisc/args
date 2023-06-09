from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False, padding_side='left')

def run_model(prompt: str) -> str:
    global model, tokenizer
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    res_tokens = model.generate(tokens, max_new_tokens=20)
    res_text = tokenizer.batch_decode(res_tokens, skip_special_tokens=True)
    return res_text[0]

while True:
    print(run_model(input("> ")))