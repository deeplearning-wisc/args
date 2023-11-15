from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification
import argparse
import torch
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)
parser.add_argument("--rm", type=str)
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--experiment", type=str, default="hhrlhf")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

with open(args.out_file, "r") as out_f:
    lines = json.load(out_f)

rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1, torch_dtype=torch.float16).to(args.rm_gpu)

def extract_out(output_data):
    # output = output_data["result"]
    # if output.startswith(": "): output = output[2:]
    # output = re.split("human:", output, flags=re.IGNORECASE)[0]
    # return output_data["prompt"] + output
    if "response" in output_data:
        output = output_data["response"]
    elif "output" in output_data:
        output = output_data["output"]

    if args.experiment == "hhrlhf":
        output_np = output.removeprefix(output_data["prompt"])
        if output_np.startswith(": "): output = output_np[2:]
        output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
        return output_data["prompt"]+output_np
    elif args.experiment == "shp":
        return output

    # return output_data["output"]

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

import numpy as np
if args.npout != "": np.save(f"{args.npout}", np.array(rm_scores))
print(f"{np.mean(rm_scores)=}")
print(f"{num_skip=}")
