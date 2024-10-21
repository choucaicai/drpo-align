
# Optimizing Preference Alignment with Differentiable NDCG Ranking

This repository contains the implementation of the paper "Optimizing Preference Alignment with Differentiable NDCG Ranking".

Full code will be available soon. We are currently in the process of organizing and refining all the code. Some portions of the code have already been uploaded.


## Overview

This project introduces a novel approach to optimize preference alignment using a diffNDCG. Our method allows for end-to-end training of ranking models while directly optimizing for NDCG, leading to improved performance in preference alignment tasks.

## Prerequisites
### Installation 

To set up the project environment:

1. Clone the repository:
2. Install request environment
```
transformers                      4.44.0
accelerate                        0.33.0
bitsandbytes                      0.41.0
```


## Usage

### Training

To train the model:
```
bash train.sh
```

trained SFT model can be found in huggingface.
- [Qwen1.5-0.5-HH-SFT](https://huggingface.co/kasoushu/qwen1.5-0.5-hh-sft)
- [Qwen1.5-1.8-HH-SFT](https://huggingface.co/kasoushu/qwen1.5-1.8-hh-sft)
- [Mistral-SFT-beta](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta)

### Reward Model Evaluation

1. To evaluate a trained model, the first step is to generate sample responses.

- using hf sample
```
    accelerate launch --main_process_port 29500 eval.py \
        --config-path="eval/eval_configs/hf_config" \
        ++mode="sample" \
        ++datasets="[hh_sample512]" \
        ++n_samples=512 \
        ++model.eval_batch_size=4 \
        ++samples_dir="samples/" \
        ++exp_name="model_name" \
        ++model.name_or_path=model_path_name
```

- using vllm sample
```
    python eval.py \
        --config-path="eval_configs/vllm_config_t1.0" \
        ++mode="vllm_sample" \
        ++datasets="[hh_sample512]" \
        ++n_samples=512 \
        ++model.eval_batch_size=4 \
        ++samples_dir="samples/" \
        ++exp_name="model_name" \
        ++model.name_or_path=model_path_name
```

2. Calculate the win rate using the Reward model

- download [reward model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) from huggingface

- Execute the command `python tools/compare_reward.py --samples_file "$file" --output_path "$output_result"` to compare sample responses with chosen responses.

- Execute the command `python tools/compare_reward_sft.py --samples_file "$file"  --sft_file "$sft_file" --output_path "$output_result"` to compare sample responses with other responses such as SFT samples.


### GPT Evaluation

1. modify eval/tools/api_client.py
```
api_key=  "Your Openai Key"
api_url=  "base url"
```


2. Execute the command `python tools/compare.py -f "$file"  -mc 512 -bk chosen -ck policy --r "$result_file" --judge "$judge_bot"`. where 
    - `$file` is the path to the file containing sampled responses.
    - `$result_file` specifies the location where the results will be saved.
    - `$judge_bot` determines the model used for judging, with "gpt-4o" as the default option.

## Project Structure

```
drpo-align
│
├── data/ # Training data
├── eval/ # evaluation scripts
├── recipes/ # Training configs
├── scripts/ # Training scripts
├── src/
│ ├── load_data/ # Data loading and preprocessing
│ ├── losses/ # LTR losses
│ ├── rank_utils/ # LTR utils
│ └── scores/ # define scores
│ └── trainer/ # list trainer
```
