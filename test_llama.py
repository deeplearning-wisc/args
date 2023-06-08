# import the huggingface transformers libraries
# they internally leverage pytorch, so we have to have that already
from transformers import AutoTokenizer, AutoModelForCausalLM

# just initialize the objects so that it will force it to download the model&tokenizer
tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("yahma/llama-7b-hf")

print("done :)")