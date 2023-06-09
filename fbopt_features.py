from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from sentence_transformers.models.Pooling import Pooling

def batch_get_hidden_states(text: List[str], layer_idxs: List[int], llm, tokenizer):
    # quick hack to get model device
    model_device = next(llm.model.decoder.parameters()).device

    # attention_mask
    token_data = tokenizer(text, return_tensors="pt", padding=True)
    token_ids = token_data["input_ids"].to(model_device)
    attention_mask = token_data["attention_mask"].to(model_device)

    hidden_states = llm.model.decoder(input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states

    result = []
    for layer_idx in layer_idxs:
        hidden_state = hidden_states[layer_idx]
        result.append(hidden_state)

    return result, attention_mask

# NOTE: THIS IS ONLY SINGLE THREADED RIGHT NOW!!!!!!!
# this also only uses a single cuda device right now
class LLMEmbeddingGenerator:
    def __init__(self, pooling_methods: List[str], layers: List[int], device="cuda:0") -> None:
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False, padding_side='left')
        print("Loading LLM...")
        self.llm = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).to(device)
        
        kwargs = {pooling_method: True for pooling_method in pooling_methods}
        if "pooling_mode_mean_tokens" not in pooling_methods:
            kwargs["pooling_mode_mean_tokens"] = False
        pool_function = Pooling(4096, **kwargs).forward

        self.pool_function = pool_function
        self.layer_idxs = layers

    def process_batched_text(self, text: List[str]) -> torch.Tensor:
        hidden_states, attention_mask = batch_get_hidden_states(text, self.layer_idxs, self.llm, self.tokenizer)

        pool_with_mask = lambda hidden_state: self.pool_function({"token_embeddings": hidden_state, "attention_mask": attention_mask})

        return torch.cat(list(map(pool_with_mask, hidden_states)), 1) #NOTE: check this axis!!!
    
# TODO decouple the batch_size, from the embedding batch size
class EmbeddedHHRLHF(LLMEmbeddingGenerator):
    def __init__(self, pooling_methods: List[str], layers: List[int], batch_size: int = 16, data_dir: str = None, split: str="train", device="cuda:0") -> None:
        super().__init__(pooling_methods, layers, device)
        self.split = split
        self.dataset = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)[split]
        self.data_iter = self.dataset.iter(batch_size=batch_size)
    
    def next(self) -> torch.Tensor:
        row = next(self.data_iter)
        all_text = row["chosen"] + row["rejected"]

        with torch.no_grad():
            embeddings = self.process_batched_text(all_text)

            labels = torch.zeros(len(all_text))
            labels[0:len(row["chosen"])] = 1

        return {"embeddings": embeddings, "labels": labels}