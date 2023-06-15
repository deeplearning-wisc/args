from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

DATASET = "Anthropic/hh-rlhf"
SPLIT = "train"
HAS_SFT = False
POOLING_METHODS = ["pooling_mode_mean_tokens"]
LAYERS = [-1]

pooling_method_str = str(list(map(lambda x: x[len("pooling_mode_"):], POOLING_METHODS))).replace('\'','').replace(' ','')
layer_str = str(LAYERS).replace('\'','').replace(' ','')
res_dir_name = f"{DATASET.replace('/', '_').replace('-', '_')}_{SPLIT}_{'sft' if HAS_SFT else 'nosft'}_{layer_str}_{pooling_method_str}"

current_dir = Path(".")
res_dir = current_dir / "processed" / res_dir_name

print("collecting files")
chunk_nums = [int(file.name.split("_")[-1].split(".")[0]) for file in tqdm(res_dir.glob("chunk_*.pkl"))]

total_dict = {"embeddings": [], "labels": []}

print("reading files")
# make sure it's in order
for chunk_num in tqdm(sorted(chunk_nums)):
    # print(f"grabbing chunk {chunk_num}")

    out_file = res_dir / f"chunk_{chunk_num}.pkl"

    with open(out_file, "rb") as out_file_handle:
        out_dict = pickle.load(out_file_handle)
    
    total_dict["embeddings"].append(out_dict["embeddings"])
    total_dict["labels"].append(out_dict["labels"])

print("merging")
# combine the numpy arrays into one really long one
total_dict["embeddings"] = np.concatenate(total_dict["embeddings"])
total_dict["labels"] = np.concatenate(total_dict["labels"])

print("writing total file")
# make a pkl file with the same name as directory
total_file = current_dir / "processed" / f"{res_dir_name}.pkl"
with open(total_file, "wb") as out_file_handle:
    pickle.dump(total_dict, out_file_handle)
