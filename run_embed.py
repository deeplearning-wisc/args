from llm_features import LLaMaEmbeddingGenerator, OPTEmbeddingGenerator
import gc
import torch
import datasets
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run an embedding process')
parser.add_argument('--batchsize', default=32, type=int, required=True)
parser.add_argument('--numworkers', type=int, required=True)
parser.add_argument('--workerid', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)

args = parser.parse_args()

CUDA_DEVICE = args.gpu
BATCH_SIZE = args.batchsize

NUM_WORKERS = args.numworkers
MY_WORKER_NUM = args.workerid

print(f"{BATCH_SIZE=}, {CUDA_DEVICE=}, {NUM_WORKERS=}, {MY_WORKER_NUM=}")

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

print(f"{res_dir=}")
embedding_gen = LLaMaEmbeddingGenerator(POOLING_METHODS, LAYERS, device=f"cuda:{CUDA_DEVICE}")
embedding_gen.llm = embedding_gen.llm.cuda(CUDA_DEVICE)

dataset = datasets.load_dataset(DATASET, data_dir=None)[SPLIT]

def embed(row):
    print("running embed")
    global embedding_gen
    all_text = row["chosen"] + row["rejected"]
    with torch.no_grad():
        embeddings_gpu = embedding_gen.process_batched_text(all_text)

        labels = torch.zeros(len(all_text)).cpu()
        labels[0:len(row["chosen"])] = 1

        print("moving to cpu")
        embeddings = embeddings_gpu.cpu()
    del embeddings_gpu
    print("collecting")
    gc.collect()
    torch.cuda.empty_cache()
    print("done")
    return {"embeddings": embeddings.numpy(), "labels": labels.numpy()}

res_dir.mkdir(parents=True, exist_ok=True)

for num, row in enumerate(tqdm(dataset.iter(batch_size=BATCH_SIZE))):
    if num % NUM_WORKERS != MY_WORKER_NUM: continue

    print(f"Working on chunk {num}....")
    out_file = res_dir / f"chunk_{num}.pkl"

    if out_file.exists():
        print(f"chunk {num} already exists, skipping")
        continue

    out_dict = embed(row)
    del row

    with open(out_file, "wb") as out_file_handle:
        pickle.dump(out_dict, out_file_handle)