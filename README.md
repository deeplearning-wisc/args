# Alignment as Reward-Guided Search

This repository contains the technical details and official implementation of the paper [**Alignment as Reward-Guided Search**](https://openreview.net/forum?id=shgx0eqdw6). In this work, we introduce ARGS, Alignment as Reward-Guided Search, a novel framework that integrates alignment into the decoding process. By adjusting the model’s probabilistic predictions using a reward signal, ARGS generates texts with semantic diversity while being aligned with human preferences.

[ARGS live demo!](https://huggingface.co/spaces/argsearch/Inference)

![](https://hackmd.io/_uploads/HytELkzW6.png)

## Setup
The following packages, and versions were used:

```bash=
git clone https://github.com/deeplearning-wisc/args.git

conda create -n args python=3.9 -y
conda activate args

cd ARGS
pip -r requirements.txt
```

## Text generation with ARGS

To begin generation, simply import the `ARGS` class specifying the path of the language model and reward model path. Note that the reward model and language model must be trained with the same tokenizer. The tokenizer is automatically loaded from the language model path. When running the `ARGS.generate` method, a decoding method can be specified with the `method` keyword argument, the top-k tokens for reward evaluation can be specified with the `topk` keyword argument, and a reward model weight can be specified with the `weight` keyword argument.

```python
from argsearch import ARGS

LLM_PATH = "models/llama-7b/"
RM_PATH = "models/llama-7b-rm/"

searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH, llm_dev="cuda:0", rm_dev="cuda:1")
text = ""

# args-greedy decoding with weight=1.0
output_tokens = searcher.generate(text, topk=10, weight=1.0, method="greedy")
tokens_text = searcher.tokens_to_text(output_tokens)[0]
print(tokens_text)
```

For more details about the ARGS generation methods:
<details open>
  <summary>ARGS Class Description</summary>

```python=
ARGS.generate(prompt: str, weight: float, topk: int, max_new_token: int, method: str, temperature: float, chunk_size: int, debug: bool)
```
* **returns** - resulting token indicies (including the original prompt)
* **prompt** - the string representation of the prompt to the model
* **weight** - the weight that controls the tradeoff between llm text objective and reward (as described in the paper)
* **topk** - the number of candidates to rank with the reward model (as described in the paper)
* **max_new_token** - the number of tokens to generate
* **method** - which decoding method to use ("greedy" for args-greedy and "topk" for args-stochastic)
* **temperature** - sets the temperature used for top-k sampling, used only when method="topk"
* **chunk_size** - sets the batch size for reward model evaluation, used only when method="greedy-large"
* **debug** - used to debug the decoding process

</details>

## Models

### Checkpoints and Training Details

We utilize various codebases for all model trainings. For a comprehensive understanding, we recommend reviewing their detailed instructions. For the Llama 7B model, we employe [LMFlow](https://github.com/OptimalScale/LMFlow/tree/main) for both supervised finetuning and reward modeling on the complete HH-RLHF training set. To replicate our setup, adjust `examples/reward_modeling.py` to load the model in `float16` and ensure you exclude the LoRA optimization. For the OPT models, all training is facilitated through [DeepSpeed Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). We are releasing the checkpoints used in the paper and the associated scripts for reproducible research.

| Base model | Checkpoints and Scripts                                                                                                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 7B   | SFT: [Script](https://pastebin.com/jJUFQwWu) and [Checkpoint](https://huggingface.co/argsearch/llama-7b-sft-float32) <br/> RM: [Script](https://pastebin.com/2ifxRBAb) and [Checkpoint](https://huggingface.co/argsearch/llama-7b-rm-float32)                                                     |
| OPT-125m   | SFT: [Script](https://pastebin.com/WT68UaBs) <br/> RM: [Script](https://pastebin.com/hitmPibN) |
| OPT-350m   | SFT: [Script](https://pastebin.com/H02x2EVS) <br/> RM: [Script](https://pastebin.com/hitmPibN) |
| OPT-1.3b   | SFT: [Script](https://pastebin.com/axSqXU8b) <br/> PPO: [Script](https://pastebin.com/QiMFhVLi) |
| OPT-2.7b   | SFT: [Script](https://pastebin.com/xwaL9WM3)                                                                                                                         |

For example, if you want to train the `OPT-1.3b` PPO model, you can run the following snippet
```bash
# Set up
conda create -n deepspeed python=3.9 -y
conda activate deepspeed

git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt

# SFT
export SFT_PATH="..."
cd training/step1_supervised_finetuning/
wget https://pastebin.com/raw/axSqXU8b -O- | dos2unix > training_scripts/opt/single_node/run_1.3b.sh
bash training_scripts/opt/single_node/run_1.3b.sh $SFT_PATH
cd ../..

# RM
export RM_PATH="..."
cd training/step2_reward_model_finetuning/
wget https://pastebin.com/raw/hitmPibN -O- | dos2unix > training_scripts/opt/single_node/run_350m.sh
bash training_scripts/opt/single_node/run_350m.sh $RM_PATH $SFT_PATH
cd ../..

# PPO
export PPO_PATH="..."
cd training/step3_rlhf_finetuning/
wget https://pastebin.com/raw/SRL5Zh53 -O- | dos2unix > training_scripts/opt/single_node/run_350m.sh
bash training_scripts/opt/single_node/run_1.3b.sh $SFT_PATH $RM_PATH 2 2 $PPO_PATH
```

If you want to train the Llama 7B, you can run the following snippet

```bash
# Set up
git clone -b v0.0.5 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh

# Download dataset
cd data && ./download.sh hh_rlhf && cd -
rm hh_rlhf/sft/._hh_rlhf_sft_data.json

# SFT
wget https://pastebin.com/raw/jJUFQwWu -O- | dos2unix > ./scripts/run_finetune.sh
./scripts/run_finetune.sh --model_name_or_path "<base llama path>" --dataset_path "data/hh_rlhf/sft/"

# RM
wget https://pastebin.com/raw/2ifxRBAb -O- | dos2unix > ./scripts/run_reward_modeling.sh
./scripts/run_reward_modeling.sh
```

## Inference
To run the inference script, first a configuration file must be made:

`echo '{rm_weight": 1.0, "topk": 10, "mode": "greedy", "sample_temp": null}' > example.config`
This example configuration runs `args-greedy` with `k=10` and `w=1.0`.

Additionally, if a OPT model is used, the deepspeed weights must be formatted to work with huggingface. This can be done with the following command:

```bash=
python reformat_dspeed_to_hf.py 
    --in_bin="model.bin"
    --out_bin="new_model.bin"

# optionally save the old model.bin
# note it cannot be in the same directory

mv new_model.bin model.bin
```

Then the following command can be run to start generation:

```bash
python collect_model_outs.py 
    --run_percent 25. 
    --config="example.config" 
    --out_file="run_outs/example_out" 
    --llm_gpu="cuda:0" 
    --rm_gpu="cuda:1" 
    --llm="models/llama-7b/" 
    --rm="models/llama-7b-rm/"
    --dataset="Dahoas/full-hh-rlhf"
```

The final result of the generation will be stored as a jsonl file with the path `run_outs/example_out_0.jsonl` where the number at the end corresponds to the line number in the configuration file.

## Evaluations

To prepare for the evaluations, please extract your models output in the following form and save it as `outputs/your_run_name.json`. Note that the response should contain both prompt and the output generated by the model.

```jsonld
[
    {
        "prompt": "Are you okay? You look",
        "response": "Are you okay? You look a bit tired or stressed. Anything you'd like to talk about?",
        "method": "greedy"
    },
    {
        "prompt": "...",
        "response": "...",
        "method": "..."
    },
    ...
]
```

To run the evaluations, execute the following commands:

```bash
# For metric evaluations: diversity and coherence
python metrics.py --run_name your_run_name

# For GPT-4 evaluation
python gpt4-eval.py \
    --run_name_red your_run_name_red \
    --run_name_blue your_run_name_blue

# for reward score and runtime evaluation
# for OPT:
python measure_reward.py 
    --out_file="run_outs/example_out_0.json"
    --rm_path="<path to reward model>"
    --tokenizer="<path to base LLM>"
    --rm_gpu="cuda:0"
    --experiment="shp"

# for Llama:
python measure_reward.py 
    --out_file="run_outs/example_out_0.json"
    --rm_path="<path to reward model>"
    --tokenizer="<path to base LLM>"
    --rm_gpu="cuda:0"
    --experiment="hhrlhf"
```

## Results
We conducted a performance comparison between ARGS and conventional decoding methods on Supervised Fine-Tuning (SFT) versions of Llama-7b and several OPT models. Remarkably, ARGS consistently outperformed the alternatives, including scenarios where the OPT model was trained using PPO.

![](https://hackmd.io/_uploads/r1Q2ZFlzp.png)
ARGS-Greedy Llama-7b performance

![](https://hackmd.io/_uploads/HyVn-Yefa.png)
ARGS-Greedy OPT Model performance

## Possible Issues
Before filing an issue, we kindly ask that you check here for details:

<details>
  <summary>Issues with PPO and Deepspeed</summary>
  There are known issues with loading the deepspeed Adam CPU operation:
    
Related GitHub issues:
* https://github.com/microsoft/DeepSpeed/issues/2268
* https://github.com/microsoft/DeepSpeed/issues/2682
* https://github.com/microsoft/DeepSpeed/issues/1127

A possible workaround that worked for us is:
commenting out the following lines (105 to 108) in the `assert_no_cuda_mismatch` function, which can be found in the following path `python3.9/site-packages/deepspeed/ops/op_builder/builder.py`.

```python=
    raise Exception(
            f">- DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} does not match the "
            f"version torch was compiled with {torch.version.cuda}, unable to compile "
            "cuda/cpp extensions without a matching cuda version.")
```
    
Then, run:

```bash=
python -c 'import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()'
```
</details>

<details>
  <summary>Faster Decoding</summary>
 
The default greedy decoding strategy is relativly slow, but should work everywhere. There is a faster decoding strategy which can be accessed by setting `method="greedy-large"` in the `ARGS.generate` method. This requires you to either set the `chunk_size` keyword argument too, or it can be automatically determined by setting `chunk_size="auto"`. If `chunk_size` is set to `"auto"`, the `auto_size` method may need to be modified. The method should work for `NVIDIA RTX A6000` cards.
    
</details>

## Citation

If you find this repository useful in your research, please consider citing:

```
@inproceedings{khanov2024args,
    title={ARGS: Alignment as Reward-Guided Search},
    author={Maxim Khanov and Jirayu Burapacheep and Yixuan Li},
    booktitle={Proceedings of the International Conference on Learning Representations},
    year={2024}
}
```
