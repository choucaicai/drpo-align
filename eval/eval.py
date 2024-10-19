# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main script for running evals. This will run an eval according to the specified config, which should be a YAML file generated during training.
You must override the mode from 'train' to one of 'sample', 'eval', or 'alpacaeval'. 
Overriding the other config parameters is optional.

For sampling, do something like:
    python eval.py --config-path=/data/models/archangel/archangel_sft_pythia1-4b ++mode=sample ++n_samples=512 ++model.eval_batch_size=32    

For calculating the batch metrics (e.g., accuracy of predicted preference direction when preference is inferred from DPO rewards) on a held-out set:
    python eval.py --config-path=/data/models/archangel/archangel_sft_pythia1-4b ++mode=eval

To sample from the unaligned model (e.g., the original EleutherAI/pythia1-4b), add ++saved_policy=null to the command.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from transformers import set_seed
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import json
import socket
from typing import Optional, Set
from utils import sample,vllm_sample
import dataloader
from datetime import datetime

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for evaluating. Validates config, loads model(s), and kicks off worker process(es)."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))

    assert config.mode in ['sample','vllm_sample']
    
    set_seed(config.seed)

    print('=' * 80)
    os.makedirs(config.samples_dir, exist_ok=True)
    print(f'Writing to', config.samples_dir)
    print('=' * 80)

    # purely inference, so put as much as possible onto the first gpu
    model_kwargs = {} 

    print(f'Loading tokenizer at {config.model.name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path)
    print('tokenizer pad_token_id',tokenizer.pad_token_id)
    if tokenizer.pad_token_id is None:
        print('tokenizer.pad_token_id is None, setting to eos token id ',tokenizer.eos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f'Loading dataloader')
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        # since the human/asst fields are not in the configs of the already-released models, add defaults
        human_prefix=config['human_prefix'],
        human_suffix=config['human_suffix'],
        assistant_prefix=config['assistant_prefix'],
        assistant_suffix=config['assistant_suffix'],
        seed=config.seed,
        frac_unique_desirable=config.get('frac_unique_desirable', 1.0),
        frac_unique_undesirable=config.get('frac_unique_undesirable', 1.0),
        chosen_control_token=None,
        rejected_control_token=None,
    )
    eval_iterator = dataloader.SFTDataLoader(
        config.datasets, 
        tokenizer,
        split='test',
        batch_size=config.model.eval_batch_size,
        n_epochs=config.n_epochs , 
        n_examples=config.n_samples,
        max_prompt_count=1,
        **data_iterator_kwargs
    )

    if config.mode == 'vllm_sample':
        
        samples = vllm_sample(config.model.name_or_path,tokenizer,eval_iterator,
                                temperature=config.temperature,
                                top_p=config.top_p,
            )

    elif config.mode == 'sample':
        print('building policy')
        policy_dtype = getattr(torch, config.model.policy_dtype)
        model_class = transformers.AutoModelForCausalLM
        policy = model_class.from_pretrained(
            config.model.name_or_path, low_cpu_mem_usage=True, use_flash_attention_2=config.model.use_flash_attention, torch_dtype='auto', **model_kwargs)
        # policy.resize_token_embeddings(len(tokenizer)) # model being loaded should already be trained with additional tokens for this to be valid
        disable_dropout(policy)
        # saved policy can be force set to null to sample from pretrained model
        if config.saved_policy is not None:
            state_dict = torch.load(os.path.join(config.cache_dir, config.saved_policy), map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained weights for policy at step {step} from {config.saved_policy} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state'])

        samples = sample(policy,tokenizer,eval_iterator,
                            max_length= config.model.max_length,
                            max_new_tokens = config.model.max_length,
                            top_p=config.top_p,
                    )
        
    os.makedirs(config.samples_dir, exist_ok=True)
    fn = os.path.join(config.samples_dir, f'{config.exp_name}.json')
    json.dump({
        'sampled_at' : str(datetime.now()),
        'config' : OmegaConf.to_container(config, resolve=True),
        'samples' : samples,
    }, open(fn, 'w'), indent=2)


if __name__ == '__main__':
    main()