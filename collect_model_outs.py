from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
# from simctg.modsimctg import SimCTGLlama
from search import RBSearch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--run_percent", type=float, default=100.)
# parser.add_argument("--rm", type=str, default=r"/nobackup-fast/khanov/orig_full_rm")
parser.add_argument("--rm", type=str, default=r"/nobackup-fast/bruh/LMFlow/output_models/llama-7b-rm")
# parser.add_argument("--llm", type=str, default=r"/nobackup-fast/khanov/orig_full_sft")
parser.add_argument("--llm", type=str, default=r"/nobackup-fast/bruh/LMFlow/output_models/llama-7b-sft")
parser.add_argument("--max_new_token", type=int, default=128)

parser.add_argument("--llm_gpu", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")
parser.add_argument("--recover", action='store_true', default = False)

parser.add_argument("--config", type=str)

parser.add_argument("--out_file", type=str)

args = parser.parse_args()

print(f"{args=}")

if args.recover:
    print("[INFO]: LOOKS LIKE YOU WANT TO RECOVER SOME RESULTS,")
    print("[INFO]: MAKE SURE ALL COMMANDLINE ARGS ARE EXACTLY THE SAME!!!")
    input("PRESS ENTER TO CONTINUE")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: Config doesn't exist!")
    exit(1)
    
out_path = Path(args.out_file + f"_0.jsonl")
if out_path.exists() and (not args.recover):
    print("ERROR: out_path already exists!")
    exit(1)

if not out_path.exists() and args.recover:
    print("ERROR: out_path DOESN'T exist!")
    exit(1)

with open(cfg_path) as f:
    run_configs = [json.loads(line) for line in f.readlines()]
    
# validate configs
for run_config in run_configs:
    if "contrastive" not in run_config:
        print(f"Missing key 'contrastive' in {run_config=}")
        exit(1)
    elif "rm_weight" not in run_config:
        print(f"Missing key 'rm_weight' in {run_config=}")
        exit(1)
    elif "topk" not in run_config:
        print(f"Missing key 'topk' in {run_config=}")
        exit(1)
    elif "mode" not in run_config:
        print(f"Missing key 'mode' in {run_config=}")
        exit(1)
    elif "sampling" not in run_config:
        print(f"Missing key 'sampling' in {run_config=}")
        exit(1)
    elif "sample_temp" not in run_config:
        print(f"Missing key 'sample_temp' in {run_config=}")
        exit(1)
#     elif "new_token" not in run_config:
#         print(f"Missing key 'new_token' in {run_config=}")
#         exit(1)

print(f"[INFO]: Loaded {len(run_configs)} run configs.")
print(f"[DEBUG]: {run_configs=}")
    
print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
test_ds = load_dataset(args.dataset, split=args.split)
test_ds = test_ds["prompt"]

# print(f"[INFO]: Before max length {len(test_ds)=}")
# test_ds = list(filter(lambda x: len(x) < 2**11, test_ds))
# print(f"[INFO]: After max length {len(test_ds)=}")

end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

truncated_ds = test_ds[0:end_idx]
# print(f"{truncated_ds=}")
print(f"{len(truncated_ds)=}")
# print(f"{truncated_ds.keys()=}")

print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
# model = SimCTGLlama(args.llm, args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
search = RBSearch(llm_path=args.llm, rm_path=args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
print(f"[INFO]: Done")

# def runprompt(prompt: str, contrastive=0., rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sampling="greedy", sample_temp=None, llm_dev:str="cuda:0") -> str:
#     input_ids = model.tokenizer([prompt], return_tensors="pt")
#     result_tokens = model.fast_contrastive_search(input_ids["input_ids"].to(llm_dev), topk, contrastive, rm_weight, new_token, mode=mode, sampling=sampling, sample_temp=sample_temp)
#     result_text = model.tokenizer.batch_decode([result_tokens], skip_special_tokens=True)[0]
#     return result_text[len(prompt):]

def runprompt(prompt: str, contrastive=0., rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sampling="greedy", sample_temp=None, llm_dev:str="cuda:0") -> str:
    if mode != "logit_mix": raise ValueError("Invalid mode")
    if sampling != "greedy": raise ValueError("Invalid sampling")
    if contrastive != 0: raise ValueError("Invalid contrastive")
        
#     tokens = search.generate(prompt, method="greedy", topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)
    tokens = search.generate(prompt, method="greedy_large", topk=topk, chunk_size="auto", max_new_token=new_token, weight=rm_weight, debug=False)
    
    # too long seqlen
    if tokens == None: return None, None
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokens_to_text(tokens)[0]
    del tokens
    tokens_text_np = tokens_text.removeprefix(prompt)
    return tokens_text_np, raw_tokens

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("last line not empty??")
            exit(1)
        
        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) -1]:
            print(f"[INFO]: PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        if args.recover and (idx <= len(samples) -1):
            print(f"[INFO]: SKIPPING {idx}")
            continue

        print(f"{ds_row=}")
        current_prompt = ds_row #["prompt"]
        start = time.time()
        res, tokens = runprompt(current_prompt, float(run_config["contrastive"]), float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sampling"], run_config["sample_temp"], llm_dev=args.llm_gpu)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"_{config_num}.jsonl"), "a") as outfile:
            json.dump({**run_config, **{"prompt": current_prompt, "result": res, "tokens": tokens, "elapsed":elapsed, "prompt_len": len(current_prompt)}}, outfile, ensure_ascii=False)
            outfile.write('\n')
