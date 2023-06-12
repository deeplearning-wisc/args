from torch import nn
from torch import optim
from llm_features import EmbeddedHHRLHF, LLaMaEmbeddingGenerator
import gc
import torch

ds = EmbeddedHHRLHF(["pooling_mode_mean_tokens"], [16], llm_embedding=LLaMaEmbeddingGenerator)

ds.llm_embedding_generator.llm = ds.llm_embedding_generator.llm.cuda()

# auxmodel = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), 
#                          nn.Linear(2048, 1024), nn.ReLU(), 
#                          nn.Linear(1024, 512), nn.ReLU(), 
#                          nn.Linear(512, 256), nn.ReLU(),
#                          nn.Linear(256, 1))

auxmodel = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), 
                         nn.Linear(1024, 256), nn.ReLU(), 
                         nn.Linear(256, 1)).cuda()

# , nn.Sigmoid()

loss_f = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(auxmodel.parameters())

while True:
    with torch.no_grad():
        batch = ds.next()
    
    if batch is None: break

    optimizer.zero_grad()

    model_out = auxmodel(batch["embeddings"]).cuda()
    expected = batch["labels"].view(-1, 1).cuda()

    loss_value = loss_f(model_out, expected)

    print(f"{loss_value.item()=}")
    loss_value.backward()
    optimizer.step()

    # reclaim memory from batch
    del batch
    gc.collect()
    torch.cuda.empty_cache()