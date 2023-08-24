from transformers import LlamaTokenizer, LlamaForSequenceClassification
import argparse
import torch
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)
parser.add_argument("--rm", type=str, default=r"/nobackup-fast/bruh/LMFlow/output_models/llama-7b-rm")
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--tokenizer", type=str, default=r"/nobackup-fast/bruh/LMFlow/output_models/llama-7b-sft")
parser.add_argument("--npout", type=str)

args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

with open(args.out_file, "r") as out_f:
    lines = out_f.readlines()
    if lines[-1] == "": lines = lines[:-1]
    lines = [json.loads(line) for line in lines]

rm_model = LlamaForSequenceClassification.from_pretrained(args.rm, num_labels=1, torch_dtype=torch.float16).to(args.rm_gpu)

def extract_out(output_data):
    output = output_data["result"]
    if output.startswith(": "): output = output[2:]
    # output = re.split("human:", output, flags=re.IGNORECASE)[0]
    return output_data["prompt"] + output

def get_rm(text):
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(args.rm_gpu)
    print(f"{tokens.shape=}")
    # 1966 1819 1813
    if tokens.shape[1] >= 1334: return None
    rm_out = rm_model(tokens)

    rm_val = rm_out.logits.flatten().item()

    del rm_out
    del tokens
    return rm_val

def get_rm_from_tokens(tokens):
    return rm_model(torch.tensor(tokens).unsqueeze(0).to(args.rm_gpu)).logits.flatten().item()

from tqdm import tqdm

rm_scores = []
num_skip = 0
for line in tqdm(lines):
    outp = extract_out(line)
    if len(outp) == 0: rm_scores.append(0.)
    # print(f"{get_rm(outp)}")
    rm_score = get_rm(outp)
    if rm_score == None: 
        print("skipped one")
        num_skip += 1
        continue
    else: rm_scores.append(rm_score)
    # rm_scores.append(get_rm_from_tokens(line["tokens"]))

import numpy as np
np.save(f"{args.npout}", np.array(rm_scores))
print(f"{np.mean(rm_scores)=}")
print(f"{num_skip=}")
