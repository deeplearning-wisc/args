from llm_features import LLaMaEmbeddingGenerator, OPTEmbeddingGenerator
import gc
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

CUDA_DEVICE = 0
BATCH_SIZE = 16

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
embedding_gen = LLaMaEmbeddingGenerator(POOLING_METHODS, LAYERS)
embedding_gen.llm = embedding_gen.llm.cuda(CUDA_DEVICE)

dataset = load_dataset(DATASET, data_dir=None)[SPLIT]

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
    return {"embeddings": embeddings, "labels": labels}

res_dir.mkdir(parents=True, exist_ok=True)

# for row in tqdm(dataset.iter(batch_size=BATCH_SIZE)):
#     embed(row)

new_ds = dataset.map(embed, batched=True, batch_size=BATCH_SIZE, remove_columns=["rejected", "chosen"])

new_ds.save_to_disk(res_dir.absolute())