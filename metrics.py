from tqdm import tqdm
import json
import argparse
from nltk import word_tokenize
import os
from simcse import SimCSE
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", default="llama-7b-greedy", type=str)

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    return args


def compute_rep_n(text, n):
    tokens = word_tokenize(text)
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n


def compute_diversity(text):
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n_val / 100
    return diversity


def clean(text, sep="###"):
    return text.split(sep)[0]


def average(entries):
    return sum(entries) / len(entries)


def compute_coherence(prompts, responses):
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    similarities = np.array(model.similarity(prompts, responses))
    return similarities.trace() / len(similarities)


if __name__ == "__main__":
    args = get_args()

    path = os.path.join("outputs", f"{args.run_name}.json")
    generations = json.load(open(path, "r"))

    entries = []
    for generation in tqdm(generations):
        prompt = generation["prompt"]
        response = clean(clean(generation["response"][len(prompt) :], "###Human:"), "\n\nHuman:")
        if len(response) == 0:
            response = " "
        rep_2 = compute_rep_n(response, 2)
        rep_3 = compute_rep_n(response, 3)
        rep_4 = compute_rep_n(response, 4)
        diversity = compute_diversity(response)
        entries.append(
            {
                "prompt": prompt,
                "response": response,
                "original_response": generation["response"][len(prompt) :],
                "rep_2": rep_2,
                "rep_3": rep_3,
                "rep_4": rep_4,
                "diversity": diversity,
                "response_length": len(response),
                "elapsed": generation["elapsed"],
            }
        )

    evaluations = {
        "rep_2": average([entry["rep_2"] for entry in entries]),
        "rep_3": average([entry["rep_3"] for entry in entries]),
        "rep_4": average([entry["rep_4"] for entry in entries]),
        "diversity": average([entry["diversity"] for entry in entries]),
        "coherence": compute_coherence(
            [entry["prompt"] for entry in entries], [entry["response"] for entry in entries]
        ),
        "response_length": average([entry["response_length"] for entry in entries]),
        "elapsed": average([entry["elapsed"] for entry in entries]),
        "entries": entries,
    }

    eval_path = os.path.join("evaluations", f"{args.run_name}.json")
    json.dump(evaluations, open(eval_path, "w"), indent=2)
