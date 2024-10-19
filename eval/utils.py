import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import gc
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
import tensor_parallel as tp
import contextlib
import transformers
import dataloader
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from accelerate import PartialState
from accelerate.utils import gather_object


def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    device = torch.device('cuda', rank)
    all_values = [torch.empty_like(values).to(device) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)



def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)



def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device

def get_batch_samples(policy, tokenizer, batch: Dict[str, torch.LongTensor],
                      max_length,
                      max_new_tokens,
                      top_p,
                      ) -> Tuple[str, str]:
    """Generate samples from the policy."""

    policy_output = policy.generate(
        batch['prompt_input_ids'],
        attention_mask=batch['prompt_attention_mask'],
        # max_length=self.config.model.max_length,
        # max_new_tokens = self.config.model.max_length,
        max_length=max_length,
        max_new_tokens = max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        top_p = top_p,
    )
    # return temp
    policy_output = pad_to_length(policy_output, max_length, tokenizer.pad_token_id)
    policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    return policy_output_decoded


def sample(policy,tokenizer,eval_iterator,
                    max_length= 512,
                    max_new_tokens= 512,
                    top_p=0.95,
           ) -> List[Dict[str, str]]:
    """
    Generate samples from the policy model.

    Returns:
        A list of samples, each of which is of the form:
        {
            'prompt': the input
            'chosen': the generation chosen by the human for the given prompt
            'policy': the generation from the policy model
        }
    """
    all_policy_samples, all_prompts, all_chosen = [], [], []
    samples = []

    if isinstance(policy, nn.Module):
        policy.eval()

    distributed_state = PartialState()
    policy = policy.to(distributed_state.device)

    print('listttttttttttttttttttttttt',len(list(eval_iterator)))
    with distributed_state.split_between_processes(list(eval_iterator),apply_padding=False) as total_batches:
        print('locaaaaaaaaaaal',len(list(total_batches)))
        for local_eval_batch in tqdm.tqdm(list(total_batches)):

            local_eval_batch = {
                k: v.to(distributed_state.device) if isinstance(v,torch.Tensor) else v for k,v in local_eval_batch.items()
            }
            # for k,v in local_eval_batch.items():
            #     if isinstance(v,torch.Tensor):
            #         print('local:',k,v.shape,v.device)
            all_prompts.extend(local_eval_batch['prompt_text'])

            policy_samples = get_batch_samples(policy,tokenizer,local_eval_batch, max_length, max_new_tokens, top_p)
            # policy_samples = ["",""]
            print(local_eval_batch['prompt_text'])
            print(policy_samples)

            all_policy_samples.extend(policy_samples)
            chosen_samples = []
            for x in (local_eval_batch['target_text'] if 'target_text' in local_eval_batch else local_eval_batch['chosen_text']):
                if tokenizer.eos_token in x:
                    x = x[:x.rfind(tokenizer.eos_token)]
                chosen_samples.append(x)
            all_chosen.extend(chosen_samples)

    all_policy_samples = gather_object(all_policy_samples)
    all_prompts = gather_object(all_prompts)
    all_chosen = gather_object(all_chosen)

    if distributed_state.is_main_process:
        print('all_policy_samples',len(all_policy_samples))
        print('all_prompts',len(all_prompts))
        print('all_chosen',len(all_chosen))

        for i in range(len(all_prompts)):
            samples.append({
                'prompt' : all_prompts[i],
                'chosen' : all_chosen[i],
                'policy' : all_policy_samples[i][len(all_prompts[i]):], 
                # 'policy' : all_policy_samples[i], 

            })

    return samples


from vllm import LLM, SamplingParams

def vllm_sample(model_name_or_path,tokenizer,eval_iterator,
                    max_length= 512,
                    max_new_tokens= 512,
                    temperature=0.8,
                    top_p=0.95,
           ) -> List[Dict[str, str]]:
    """
    Generate samples from the policy model.

    Returns:
        A list of samples, each of which is of the form:
        {
            'prompt': the input
            'chosen': the generation chosen by the human for the given prompt
            'policy': the generation from the policy model
        }
    """
    all_policy_samples, all_prompts, all_chosen = [], [], []
    samples = []

    policy = LLM(model=model_name_or_path,dtype='bfloat16',gpu_memory_utilization=0.25)

    sampling_params = SamplingParams(max_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=50)
    print('===========================================================')
    print(sampling_params)

    for local_eval_batch in tqdm.tqdm(list(eval_iterator)):

        # local_eval_batch = {
        #     k: v.to(distributed_state.device) if isinstance(v,torch.Tensor) else v for k,v in local_eval_batch.items()
        # }
        # for k,v in local_eval_batch.items():
        #     if isinstance(v,torch.Tensor):
        #         print('local:',k,v.shape,v.device)
        # outputs = policy.generate(prompt_token_ids=local_eval_batch['prompt_input_ids'].tolist(), sampling_params=sampling_params)
        outputs = policy.generate(local_eval_batch['prompt_text'], sampling_params=sampling_params)
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # print(outputs)
        policy_samples = [output.outputs[0].text for output in outputs]
        # policy_samples = [""]
        # print(local_eval_batch['prompt_text'])
        all_prompts.extend(local_eval_batch['prompt_text'])
        all_policy_samples.extend(policy_samples)
        chosen_samples = []
        for x in (local_eval_batch['target_text'] if 'target_text' in local_eval_batch else local_eval_batch['chosen_text']):
            if tokenizer.eos_token in x:
                x = x[:x.rfind(tokenizer.eos_token)]
            chosen_samples.append(x)
        all_chosen.extend(chosen_samples)

    all_policy_samples = gather_object(all_policy_samples)
    all_prompts = gather_object(all_prompts)
    all_chosen = gather_object(all_chosen)

    print('all_policy_samples',len(all_policy_samples))
    print('all_prompts',len(all_prompts))
    print('all_chosen',len(all_chosen))

    for i in range(len(all_prompts)):
        samples.append({
            'prompt' : all_prompts[i],
            'chosen' : all_chosen[i],
            'policy' : all_policy_samples[i], 
            # 'policy' : all_policy_samples[i], 
        })

    return samples