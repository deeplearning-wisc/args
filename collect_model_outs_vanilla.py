from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification
import torch
from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from search import RBSearch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--run_percent", type=float, default=100.)
parser.add_argument("--llm", type=str)
parser.add_argument("--max_new_token", type=int, default=128)
parser.add_argument("--params", type=str, default="{}")
parser.add_argument("--llm_gpu", type=str, default="cuda:0")

parser.add_argument("--out_file", type=str)

args = parser.parse_args()

additional_dict = eval(args.params)

print(f"{args=}")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

out_path = Path(args.out_file + f".jsonl")
if out_path.exists():
    print("ERROR: out_path already exists!")
    exit(1)

print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
test_ds = load_dataset(args.dataset, split=args.split)

if args.dataset == "Dahoas/full-hh-rlhf":
    # FOR HHRLHF
    test_ds = test_ds["prompt"]
elif args.dataset == "stanfordnlp/SHP":
    # FOR SHP
    unique_prompts = []
    seen_posts = set()
    for post_id, histr in zip(test_ds["post_id"], test_ds['history']):
        if post_id in seen_posts: continue
        model_prompt = " Human: " + histr + " Assistant: "
        unique_prompts.append(model_prompt)
        seen_posts.add(post_id)
    test_ds = unique_prompts

end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

truncated_ds = test_ds[0:end_idx]
print(f"{len(truncated_ds)=}")

print(f"[INFO]: Loading models ({args.llm=})")

from types import SimpleNamespace

llm_data = SimpleNamespace()

llm_path = args.llm
llm_data.llm_dev = args.llm_gpu
print("Loading LLM...")
llm_data.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16).to(llm_data.llm_dev)
llm_data.LLM.eval()

print(f"Loading tokenizer...")
llm_data.tokenizer = AutoTokenizer.from_pretrained(llm_path)

def get_input_ids(prompt):
    tokens = llm_data.tokenizer(prompt, return_tensors="pt").input_ids.to(llm_data.llm_dev)
    return tokens

def tokens_to_text(tokens):
    return llm_data.tokenizer.batch_decode(tokens, skip_special_tokens=True)

print(f"[INFO]: Done")

def runprompt(prompt: str, new_token=24, llm_dev:str="cuda:0", params={"NODEF": True}) -> str:
    tokens = get_input_ids(prompt)

    if tokens.shape[-1] > llm_data.LLM.config.to_dict().get("max_sequence_length", 2048):
        print("The sequence of tokens is too long!!! Returning none!")
        return None, None

    tokens = llm_data.LLM.generate(tokens, max_new_tokens=new_token, **params)   
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = tokens_to_text(tokens)[0]
    del tokens
    tokens_text_np = tokens_text.removeprefix(prompt)
    return tokens_text_np, raw_tokens

data = []
for idx, ds_row in enumerate(tqdm(truncated_ds)):
    print(f"{ds_row=}")
    current_prompt = ds_row #["prompt"]
    start = time.time()
    res, tokens = runprompt(current_prompt, args.max_new_token, llm_dev=args.llm_gpu, params=additional_dict)
    if tokens == None:
        print("Too long, skipped")
        continue

    elapsed = time.time() -start

    data.append({"prompt": current_prompt, "result": res, "response": current_prompt + res, "elapsed":elapsed})
    print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
    with open(Path(args.out_file + f"_{config_num}.json"), "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False)
