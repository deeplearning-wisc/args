import torch

# import the huggingface transformers libraries
# they internally leverage pytorch, so we have to have that already
from transformers import AutoTokenizer, AutoModelForCausalLM

# just initialize the objects so that it will force it to download the model&tokenizer
print(f"loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
print(f"loading model")
model = AutoModelForCausalLM.from_pretrained("yahma/llama-7b-hf", torch_dtype=torch.float16).cuda()

def run_model(prompt: str) -> str:
    global model, tokenizer
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    res_tokens = model.generate(tokens, max_new_tokens=20)
    res_text = tokenizer.batch_decode(res_tokens, skip_special_tokens=True)
    return res_text[0]

while True:
    print(run_model(input("> ")))