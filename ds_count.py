import datasets

BATCH_SIZE = 8
DATASET = "Anthropic/hh-rlhf"
SPLIT = "train"

print("loading ds")
dataset = datasets.load_dataset(DATASET, data_dir=None)[SPLIT]

print("calculating lengths")
lengths = []
for num, row in enumerate(dataset.iter(batch_size=BATCH_SIZE)):
    lengths.extend([len(cstr) for cstr in row["chosen"]])
    lengths.extend([len(cstr) for cstr in row["rejected"]])

import numpy as np
print(f"{num} batches")
print(f"avg len: {np.mean(lengths)}")
print(f"med len: {np.median(lengths)}")
print(f"max len: {np.max(lengths)}")
print(f"min len: {np.min(lengths)}")