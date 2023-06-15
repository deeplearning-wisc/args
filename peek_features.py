from pathlib import Path
import pickle
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

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

print("loading dataset")
total_file = current_dir / "processed" / f"{res_dir_name}.pkl"
with open(total_file, "rb") as out_file_handle:
    dataset = pickle.load(out_file_handle)

print(f"cols")

print(f"{dataset['embeddings'].mean(axis=0)=}")
print(f"{dataset['embeddings'].std(axis=0)=}")

print(f"{np.mean(dataset['labels'])=}")


print(f"row")

# print(f"{dataset['embeddings'].mean(axis=1)=}")
# print(f"{dataset['embeddings'].std(axis=1)=}")

print(f"{np.mean(dataset['embeddings'].mean(axis=1))=}")
print(f"{np.std(dataset['embeddings'].mean(axis=1))=}")


print(f"{np.mean(dataset['embeddings'].std(axis=1))=}")
print(f"{np.std(dataset['embeddings'].std(axis=1))=}")