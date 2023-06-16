import setkey
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
import torch
from sklearn.utils import shuffle
import wandb
import argparse
import json
import sys

activation_functions = {"relu": nn.ReLU, "silu": nn.SiLU, "selu": nn.SELU, "tanh": nn.Tanh, "linear": nn.Identity}
optimizers = {"sgd": optim.SGD, "adam": optim.Adam, "nadam": optim.NAdam}

DATASET = "Anthropic/hh-rlhf"
SPLIT = "train"
HAS_SFT = False
POOLING_METHODS = ["pooling_mode_mean_tokens"]
LAYERS = [-1]

parser = argparse.ArgumentParser(description='Run an training run')
parser.add_argument('--batchsize', default=32, type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--optimizer', type=str, required=True, choices=list(optimizers.keys()))
parser.add_argument('--activation', type=str, required=True, choices=list(activation_functions.keys()))
parser.add_argument('--testsize', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--printevery', type=int, required=True)
parser.add_argument('--epochshuffle', action="store_true")
parser.add_argument('--standardscaletrain', action="store_true")

# parser.add_argument('--poolmethod', type=str, nargs="+", required=True)
# parser.add_argument('--poollayers', type=int, nargs="+", required=True)

parser.add_argument('--notes', type=str, default=None, required=False)
parser.add_argument('--topology', type=int, nargs="+", required=True)

args = parser.parse_args()

# if args.epochshuffle:
#     print("NOT IMPLEMENTED!!!")
#     exit()

BATCH_SIZE = args.batchsize # 32*2
PRINT_EVERY = args.printevery # 200
EPOCHS = args.epochs # 80
DEVICE = f"cuda:{args.gpu}"
WANDB_PROJECT = args.experiment

OPTIMIZER = args.optimizer

# TOPOLOGY = [4096, 256, 128, 64, 2]
TOPOLOGY = args.topology
ACTIVATION_FUNC = args.activation

TEST_SIZE = args.testsize #128*8*8

SEED = args.seed

SHUFFLEEPOCH = args.epochshuffle

STANDARD_SCALE = args.standardscaletrain

LR = args.lr # 1e-2

#NOTE: UPDATE ME
config = {"lr": LR, 
          "optimizer": OPTIMIZER, 
          "batch_size": BATCH_SIZE, 
          "pooling_methods": POOLING_METHODS, 
          "layers": LAYERS, 
          "base_model": "llama7b", 
          "print_every": PRINT_EVERY, 
          "test_size": TEST_SIZE, 
          "activation": ACTIVATION_FUNC, 
          "topo": TOPOLOGY,
          "rseed": SEED,
          "shuffle_epoch": SHUFFLEEPOCH,
          "standard_scale_train": STANDARD_SCALE}

io_pairs = [TOPOLOGY[i: i+2] for i in range(len(TOPOLOGY)-1)]

config_hash = hash(json.dumps(config, sort_keys=True)) % ((sys.maxsize + 1) *2)

print(f"{config=}")
print(f"{config_hash=}")
wandb.init(project=WANDB_PROJECT, config=config)

pooling_method_str = str(list(map(lambda x: x[len("pooling_mode_"):], POOLING_METHODS))).replace('\'','').replace(' ','')
layer_str = str(LAYERS).replace('\'','').replace(' ','')
res_dir_name = f"{DATASET.replace('/', '_').replace('-', '_')}_{SPLIT}_{'sft' if HAS_SFT else 'nosft'}_{layer_str}_{pooling_method_str}"

current_dir = Path(".")
res_dir = current_dir / "processed" / res_dir_name

MODEL_PATH =  res_dir / f"model_{config_hash}.pt"

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

print("loading dataset")
total_file = current_dir / "processed" / f"{res_dir_name}.pkl"
with open(total_file, "rb") as out_file_handle:
    dataset = pickle.load(out_file_handle)

print("creating model")

# model = nn.Sequential(nn.Linear(4096, 256), nn.ReLU(),
#                       nn.Linear(256, 128), nn.ReLU(),
#                       nn.Linear(128, 64), nn.ReLU(),
#                       nn.Linear(64, 2))

modules = []
for i, (in_size, out_size) in enumerate(io_pairs):
    modules.append(nn.Linear(in_size, out_size))
    if i != (len(io_pairs)-1):
        modules.append(activation_functions[ACTIVATION_FUNC]())

print(f"{modules=}")

model = nn.Sequential(*modules)

model = model.to(DEVICE)

loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LR)

tot_num_param = sum(p.numel() for p in model.parameters())
print(f"Total model params: {tot_num_param}")

X, y = dataset["embeddings"], dataset["labels"]
X, y = shuffle(X, y, random_state=SEED)

test_dataset_embeddings = torch.tensor(X[-TEST_SIZE:]).to(DEVICE)
test_dataset_labels = torch.tensor(y[-TEST_SIZE:]).to(DEVICE)

train_dataset_embeddings = X[:-TEST_SIZE]
train_dataset_labels = y[:-TEST_SIZE]

wandb.run.summary["num_params"] = tot_num_param
wandb.run.summary["model_hash"] = config_hash
wandb.run.summary["test_true_prob"] = torch.mean(test_dataset_labels.float()).item()
wandb.run.summary["train_true_prob"] = np.mean(train_dataset_labels)

if STANDARD_SCALE:
    COL_MEANS = torch.tensor(train_dataset_embeddings.mean(axis=0)).to(DEVICE)
    COL_STDS = torch.tensor(train_dataset_embeddings.std(axis=0)).to(DEVICE)
else:
    COL_MEANS = torch.tensor(0.).to(DEVICE)
    COL_STDS = torch.tensor(1.).to(DEVICE)

max_acc = 0
STEP = 0

print("training")
for epoch in range(EPOCHS):
    if SHUFFLEEPOCH:
        train_dataset_embeddings, train_dataset_labels = shuffle(train_dataset_embeddings, train_dataset_labels)

    running_loss = 0.0
    wandb.log({"epoch": epoch}, commit=False, step=STEP)
    for i, data in enumerate(zip(chunks(train_dataset_embeddings, BATCH_SIZE), 
                                 chunks(train_dataset_labels, BATCH_SIZE))):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = (torch.tensor(inputs).to(DEVICE) - COL_MEANS) / COL_STDS, torch.tensor(labels).long().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        wandb.log({"batch_loss": batch_loss, "sample_num": STEP*BATCH_SIZE}, commit=False, step=STEP)

        # print statistics
        running_loss += batch_loss
        if i % PRINT_EVERY == (PRINT_EVERY -1):
            est_loss = running_loss / PRINT_EVERY
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {est_loss:.3f}')
            wandb.log({"running_loss": est_loss}, step=STEP)
            running_loss = 0.0

        STEP += 1
    is_test_correct = model((test_dataset_embeddings - COL_MEANS) / COL_STDS).argmax(axis=1) == test_dataset_labels
    accuracy = is_test_correct.float().mean().item()
    max_acc = max(accuracy, max_acc)
    print(f"accuracy: {accuracy}")
    wandb.log({"val_accuracy": accuracy}, step=STEP)

wandb.run.summary["max_val_acc"] = max_acc
# wandb.run.summary["final_running_loss"] = est_loss
# wandb.run.summary["final_batch_loss"] = batch_loss

print('Finished Training')
torch.save(model.state_dict(), MODEL_PATH)
wandb.finish()