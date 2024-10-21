"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_shp, get_oasst, etc.) will return a dict of Example objects, indexed by the prompt for the text.

Each Example object will contain
- the prompt (formatted with config.human_prefix, config.assistant_prefix)
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- for binary feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for unary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the dataset name
- the unformatted prompt (needed for alpaca)
"""
import sys
sys.path.append('./')
import datasets
import torch
import json
from datasets import Dataset

from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import re
import copy
import os
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import torch.distributed as dist


# HH_RANK_BASE ="data/samples"
HH_RANK_BASE = "data/hh-rank-dataset"
ULTRA_BIN = "data/ultrabin"
ULTRA_RANK = "data/ultrafeedback"
SPLIT_ULTRA_RANK = ""

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def get_rank_dataset_file(dataset='hh',split='train',K=8):
    
    dataset_name = f'{dataset}-rank-{split}-K{K}-deberta-v3.json'
    return os.path.join(HH_RANK_BASE,dataset_name)

def read_super_rank_dataset(split,subset='hh',K=8):

    # base_path =  get_rank_dataset_file(subset,split,K=K)
    # data_list = []
    # with open(base_path,'r') as f:
    #     json_data = json.load(f)
    #     data_list.extend(json_data['samples'])
    print('******************* load data from -> ',HH_RANK_BASE)
    origin_dataset = datasets.load_from_disk(HH_RANK_BASE)
    origin_dataset = origin_dataset[f'{split}_K{K}']
    dataset = []
    
    for data in tqdm.tqdm(origin_dataset):
        prompt = data['prompt']
        reward = data['scores']
        rank_response = data['responses']

        # shuffle
        # random.shuffle(reward)
        # random.shuffle(rank_response)
        reward = reward[:K] 
        rank_response = rank_response[:K]
        # print(reward)

        assert len(reward) == len(rank_response)

        sorted_pairs = list(zip(reward, rank_response))
        random.shuffle(sorted_pairs)
        
        new_reward = [pair[0] for pair in sorted_pairs]
        new_rank_response = [pair[1] for pair in sorted_pairs]
        
        dataset.append(dict(
            prompt=prompt,
            reward = new_reward,
            sft_index = 0,
            rank_response = new_rank_response,
        ))

    return dataset

def get_hhrank(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str, only_helpful = False, only_harmless = False,K=8,eval_examples=None) -> Dataset:
    """
    Load the HH RANK DATASET
    """

    dataset = read_super_rank_dataset(split,subset= 'hh_helpful' if only_helpful else 'hh',K=K)

    assert dataset is not None

    data = []

    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing HH RANK')
    for row in dataset:
        prompt = row['prompt']
        rank_response = row['rank_response']
        reward = row['reward']
        # strip trailing spaces to avoid tokenization issues
        chunks = []
        
        for chunk in re.split(r'\s*(Human:|Assistant:)\s+', prompt): 
            if chunk.startswith('Human'):
                chunk = re.sub(r'\s*Human:\s*', human_prefix, chunk) + human_suffix
            elif chunk.startswith('Assistant'):
                chunk = re.sub(r'\s*Assistant:\s*', assistant_prefix, chunk) + assistant_suffix
            else:
                pass
            if chunk != '':
                chunks.append(chunk)

        prompt = ''.join(chunks)

        responses = [res + assistant_suffix for res in rank_response]
        
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        prompt = clean(prompt)
        responses = list(map(clean, responses))
        
        max_value = max(reward)
        sft_index = reward.index(max_value)

        example = dict(
            prompt = prompt,
            generations = responses,
            scores= reward,
            sft_index = sft_index,
            dataset_name = 'hh_rank',
        )
        data.append(example)

    if split == 'test':
            # data = data[:eval_examples]
        data = data[:256]
    # return data
    return Dataset.from_list(data[:256])


def get_hhsuperrank(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix)
def get_hhsuperrank2(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,K=2)
def get_hhsuperrank4(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,K=4)
def get_hhsuperrank6(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,K=6)
def get_hhsuperrank_helpful(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,only_helpful=True)
def get_hhsuperrankeval(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
   return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,eval_examples=None)


def get_splitrankultrafeed(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
    Returns:   
        A Dataset instance.
    """
    rank0_print(f'Loading SPLIT_ULTRA_RANK ({split} split) from Local...')
    local_data = datasets.load_from_disk(SPLIT_ULTRA_RANK)
    dataset = local_data[split]

    
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ranking UltraFeedBack ')

    data = []
    for row in dataset:
        if not (len(row['completions']) == 4) :
            print('error:',row)
            continue
        scores    = [ c["overall_score"] / 10.0 for c in row['completions']]
        prompt = human_prefix + row['instruction'] + human_suffix + assistant_prefix
        responses = [ c["response"] + assistant_suffix for  c in row['completions'] ]
        
        # clean extra space
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        prompt = clean(prompt)
        responses = list(map(clean, responses))
        
        max_value = max(scores)
        sft_index = scores.index(max_value)
        # print(reward)
        # print(sft_index)
        example = dict(
            prompt = prompt,
            scores=scores,
            generations = responses,
            sft_index = sft_index,
            dataset_name = 'rankultrafeed',
        )
        data.append(example)
    

    return Dataset.from_list(data)



def get_rankultrafeed(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
    Returns:   
        A Dataset instance.
    """
    if split == 'test':
        return get_hhrank(split,human_prefix,human_suffix,assistant_prefix,assistant_suffix,K=4)
    
    rank0_print(f'Loading UltraFeedback ({split} split) from Huggingface...')
    dataset = datasets.load_dataset(ULTRA_RANK, split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ranking UltraFeedBack ')

    data = []
    for row in dataset:
        if not (len(row['completions']) == 4) :
            print('error:',row)
            continue
        scores    = [ c["overall_score"] / 10.0 for c in row['completions']]
        prompt = human_prefix + row['instruction'] + human_suffix + assistant_prefix
        responses = [ c["response"] + assistant_suffix for  c in row['completions'] ]
        
        # clean extra space
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        prompt = clean(prompt)
        responses = list(map(clean, responses))
        
        max_value = max(scores)
        sft_index = scores.index(max_value)
        # print(reward)
        # print(sft_index)
        example = dict(
            prompt = prompt,
            scores=scores,
            generations = responses,
            sft_index = sft_index,
            dataset_name = 'rankultrafeed',
        )
        data.append(example)
    

    return Dataset.from_list(data)


def get_ultrabin(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        A Dataset instance.
    """
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    else:
        raise ValueError()
    
    rank0_print(f'Loading Ultra Binarized dataset ({split} split) from Huggingface...')
    
    dataset = datasets.load_dataset(ULTRA_BIN, split=split)
    
    print(dataset[0]['messages'])
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')
    data = []
    for row in dataset:
        prompt = human_prefix + row['prompt'] + human_suffix + assistant_prefix

        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        chosen = clean(row['chosen'][-1]['content'] + assistant_suffix)
        rejected = clean(row['rejected'][-1]['content'] + assistant_suffix)
        data.append(dict(
            prompt = prompt,
            chosen = chosen,
            rejected = rejected,
        ))
    return Dataset.from_list(data)



import os
from typing import Any, List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError



DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task in ["dpo", "orpo"]:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def get_datasets(
    data_config: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`RankDataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    dataset_mixer = data_config.dataset_mixer

    raw_datasets = load_mix_datasets(
        dataset_mixer,
        splits=splits,
        human_prefix = data_config.human_prefix,
        human_suffix = data_config.human_suffix,
        assistant_prefix = data_config.assistant_prefix,
        assistant_suffix = data_config.assistant_suffix,
        shuffle=shuffle,
    )
    return raw_datasets

def load_local_data(data_name,split, human_prefix, human_suffix, assistant_prefix, assistant_suffix):

    return globals()[f"get_{data_name}"](split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)

def load_mix_datasets(
        dataset_mixer: dict,
        splits: Optional[List[str]] = None,
        human_prefix = "",
        human_suffix = "",
        assistant_prefix = "",
        assistant_suffix = "",
        shuffle=True,
    ) -> DatasetDict:
    
    
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) 

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                dataset = load_local_data(ds,split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
            except DatasetGenerationError:
                raise NotImplementedError

            # Remove redundant columns to avoid schema conflicts on load
            # dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )
    return raw_datasets

def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


if __name__ == "__main__":
    split = 'test'
    human_prefix = "<|user|>"
    human_suffix = ""
    assistant_prefix =  "<|assistant|>"
    assistant_suffix = ""
        # - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        # - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        # - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        # - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    # dataset = get_rankultrafeed(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
    dataset = get_splitrankultrafeed(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix).shuffle(seed=42)
    print(dataset)
    # print(dataset[0])
    # all_res = 0
    # 加载数据集
    # if np.sum(score_arr) == 4.0 and  np.sum(np.abs(score_arr-0.5)) < 0.1:
    #     print(scores)
    #     print(np.sum(score_arr))
    # print(len(dataset.data[key_list[0]].generations))
    # print(dataset.data[key_list[0]].generations)
    # for row in dataset.data[key_list[0]].generations:
    #     print('-----')
    #     print(row)
    #     print()
    # for key in key_list:
        # row = dataset.data[key]
        
