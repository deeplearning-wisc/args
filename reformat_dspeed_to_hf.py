import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_bin", type=str)
parser.add_argument("--out_bin", type=str)
args = parser.parse_args()

PATH = args.in_bin
OUT_PATH = args.out_bin
print("loading state dict")
state_dict = torch.load(PATH)

special_keys = {"v_head.weight": "score.weight"}

new_state_dict = {}
for key in tqdm(state_dict.keys()):
    if key in special_keys:
        new_state_dict[special_keys[key]] = state_dict[key]
    elif key.startswith("rwtranrsformer."):
        new_state_dict["model." + key.removeprefix("rwtranrsformer.")] = state_dict[key]
    else:
        print(f"UNKNOWN KEY {key=}")

torch.save(new_state_dict, OUT_PATH)
