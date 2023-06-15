from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
import torch
from sklearn.utils import shuffle
import wandb

DATASET = "Anthropic/hh-rlhf"
SPLIT = "train"
HAS_SFT = False
POOLING_METHODS = ["pooling_mode_mean_tokens"]
LAYERS = [-1]

BATCH_SIZE = 32*2
PRINT_EVERY = 200
EPOCHS = 80
DEVICE = "cuda:0"
MODEL_NAME = "model.pt"

TEST_SIZE = 128*8*8

LR = 1e-2

#NOTE: UPDATE ME
config = {"lr": LR, "optimizer": "sgd", "batch_size": BATCH_SIZE, "pooling_methods": POOLING_METHODS, "layers": LAYERS, "base_model": "llama7b", "print_every": PRINT_EVERY, "test_size": TEST_SIZE, "activation": "relu", "topo": [4096, 256, 128, 64, 2]}


print(f"{config=}")
wandb.init(project="aux-train", config=config)

pooling_method_str = str(list(map(lambda x: x[len("pooling_mode_"):], POOLING_METHODS))).replace('\'','').replace(' ','')
layer_str = str(LAYERS).replace('\'','').replace(' ','')
res_dir_name = f"{DATASET.replace('/', '_').replace('-', '_')}_{SPLIT}_{'sft' if HAS_SFT else 'nosft'}_{layer_str}_{pooling_method_str}"

current_dir = Path(".")
res_dir = current_dir / "processed" / res_dir_name

MODEL_PATH =  res_dir / MODEL_NAME

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

print("loading dataset")
total_file = current_dir / "processed" / f"{res_dir_name}.pkl"
with open(total_file, "rb") as out_file_handle:
    dataset = pickle.load(out_file_handle)

print("creating model")

# model = nn.Sequential(nn.Linear(4096, 2048), nn.SiLU(),
#                       nn.Linear(2048, 1024), nn.SiLU(),
#                       nn.Linear(1024, 512), nn.SiLU(),
#                       nn.Linear(512, 256), nn.SiLU(),
#                       nn.Linear(256, 2))

model = nn.Sequential(nn.Linear(4096, 256), nn.ReLU(),
                      nn.Linear(256, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 2))

# model = nn.Sequential(nn.Linear(4096, 2))

model = model.to(DEVICE)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

print(f"Total model params: {sum(p.numel() for p in model.parameters())}")

X, y = dataset["embeddings"], dataset["labels"]
X, y = shuffle(X, y, random_state=0)

test_dataset_embeddings = torch.tensor(X[-TEST_SIZE:]).to(DEVICE)
test_dataset_labels = torch.tensor(y[-TEST_SIZE:]).to(DEVICE)

train_dataset_embeddings = X[:-TEST_SIZE]
train_dataset_labels = y[:-TEST_SIZE]

max_acc = 0

STEP = 0

print("training")
for epoch in range(EPOCHS):
    running_loss = 0.0
    wandb.log({"epoch": epoch}, commit=False, step=STEP)
    for i, data in enumerate(zip(chunks(train_dataset_embeddings, BATCH_SIZE), 
                                 chunks(train_dataset_labels, BATCH_SIZE))):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = torch.tensor(inputs).to(DEVICE), torch.tensor(labels).long().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        wandb.log({"batch_loss": batch_loss}, commit=False, step=STEP)

        # print statistics
        running_loss += batch_loss
        if i % PRINT_EVERY == (PRINT_EVERY -1):
            est_loss = running_loss / PRINT_EVERY
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {est_loss:.3f}')
            wandb.log({"running_loss": est_loss}, step=STEP)
            running_loss = 0.0

        STEP += 1
    is_test_correct = model(test_dataset_embeddings).argmax(axis=1) == test_dataset_labels
    accuracy = is_test_correct.float().mean().item()
    max_acc = max(accuracy, max_acc)
    print(f"accuracy: {accuracy}")
    wandb.log({"val_accuracy": accuracy}, step=STEP)

wandb.run.summary["max_val_acc"] = max_acc
wandb.run.summary["final_running_loss"] = est_loss
wandb.run.summary["final_batch_loss"] = batch_loss

print('Finished Training')
torch.save(model.state_dict(), MODEL_PATH)
wandb.finish()