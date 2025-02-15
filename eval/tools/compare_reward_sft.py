import os
from collections import defaultdict
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import random
import math
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from transformers import set_seed
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import json
import torch.nn.functional  as F
from torch.nn.utils.rnn import pad_sequence
import click
from api_client import rm_path
# template

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_path(sample_path):

    if os.path.exists(sample_path):
        with open(sample_path,'r') as f:
            return json.load(f)
    else:
        raise NotImplementedError

def load_model(model_name_or_path):
    rank_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    rank_model.eval()
    rank_model.cuda()
    return rank_model,tokenizer


def model_forward(rank_model,tokenizer,prompt,responses,max_length=1024):
    inputs = [ tokenizer(prompt , res , return_tensors='pt') for res in responses]
    inputs_ids = [ kv_dict['input_ids'].cuda().squeeze() for kv_dict in inputs ]
    attention = [ kv_dict['attention_mask'].cuda().squeeze() for kv_dict in inputs ]
    inputs_ids = pad_sequence(inputs_ids,
                batch_first=True,
                padding_value=tokenizer.eos_token_id) 
    attention_mask = pad_sequence(attention,
                batch_first=True,
                padding_value=0) 
    
    inputs_ids = inputs_ids[:,:max_length]
    attention_mask = attention_mask[:,:max_length]
    # print(inputs_ids.shape,attention_mask.shape)

    with torch.no_grad():
        scores = rank_model(inputs_ids,attention_mask=attention_mask).logits
    return scores

def group_winrate(winrates,length=512,size=4):
    group_size = math.ceil(length / size) 
    win_arr = np.array(winrates)
    win_arr = win_arr // group_size
    result = np.unique(win_arr, return_counts=True)[1]
    return result


def evaluate_win_rate(
            samples_list,
            reward_model,
            tokenizer,
            max_length=1024,
            prompt_key = 'prompt',
            candidate_key = 'policy',
            baseline_key = 'chosen',
            apply_template = None,
        ):
    # input sampels_list: List[{prompt,policy,chosen}]
    # return Dict {}, the key is below
    # 1. rewards_dict: {policy_rewards,chosen_rewards}
    # 2. win_dict : {count: 0 ,policy: {ids:[win_id]},chosen: {count: 0 ,ids:[win_id]}}
    # 3. total: e.g. 512
    total = 0
    all_rewards = defaultdict(list)
    candidate_win = []
    baseline_win = []
    candidate_count = 0
    baseline_count = 0
    pbar = tqdm(enumerate(samples_list),dynamic_ncols=True)
    for idx,item in pbar:
        prompt = item[prompt_key]
        candidate_obj = item[candidate_key]
        baseline_obj = item[baseline_key]
        score = model_forward(reward_model,tokenizer,prompt,[candidate_obj,baseline_obj])
        score = score.squeeze()
        # print( score,score.shape)
        # get score
        score_list = score.cpu().tolist()
        candidata_score = score_list[0]
        baseline_score = score_list[1]
        # 
        # print(candidata_score,baseline_score)
        # 
        total += 1
        all_rewards[candidate_key].append(candidata_score)
        all_rewards[baseline_key].append(baseline_score)
        if candidata_score > baseline_score:
            candidate_count+=1
            candidate_win.append(idx)
        else:
            baseline_count+=1
            baseline_win.append(idx)

        pbar.set_postfix({'win_rate': candidate_count / (0.000001 + total)})

    candidate_arr = group_winrate(candidate_win,length=len(samples_list),size=4)
    baseline_arr = group_winrate(baseline_win,length=len(samples_list),size=4)
    score_array = candidate_arr / (candidate_arr + baseline_arr)

    print('final result','win_rate:' , candidate_count / (0.000001 + total),
                    'win_mean:',  score_array.mean(),
                    'win_std:' , score_array.std(),  )

    return dict(
                    rewards = all_rewards,
                    total = total,
                    win_dict = {
                        candidate_key: {'wins': candidate_count,'ids': candidate_win},
                        baseline_key: {'wins': baseline_count,'ids': baseline_win},
                    },
                    win_rate = candidate_count / (0.000001 + total),
                    win_mean = score_array.mean(),
                    win_std  = score_array.std(),  
                )
    # return dict(
    #     rewards = all_rewards,
    #     total = total,
    #     win_dict = {
    #         candidate_key: {'wins': candidate_count,'ids': candidate_win},
    #         baseline_key: {'wins': baseline_count,'ids': baseline_win},
    #     },
    #     win_rate = candidate_count / (0.000001 + total)
    # )

import pathlib
@click.command()
@click.option('--samples_file',default='sampels/',help='file path of samples dict')
@click.option('--sft_file',default='sfts/',help='file path of samples dict')
@click.option('--samples_key',default='samples',  help='smaples_list key')
@click.option('--candidate_key',default='policy',  help='compare obj key')
@click.option('--baseline_key',default='sft', help='compare target obj key')
@click.option('--prompt_key',default='prompt',help='compare target obj key')
@click.option('--output_path',default='reward_win/result.json', help='output path')
def main(samples_file,sft_file, output_path,candidate_key='policy',baseline_key='sft',samples_key='samples',prompt_key='prompt',save_sample=True):

    print('samples_file ->',samples_file)
    print('sft_file ->',sft_file)
    print('output_path ->',output_path)
    print('candidate_key ->',candidate_key)
    print('target_key ->',baseline_key)
    print('samples_key ->',samples_key)
    print('prompt_key ->',prompt_key)

    print('Checking ---->',samples_file)
    print("Checking Complete")
    print('Checking ---->',sft_file)
    print("Checking Complete")
    policy_samples_data =  check_path(samples_file)
    sft_samples_data =  check_path(sft_file)

    policy_samples = policy_samples_data[samples_key]
    sft_samples = sft_samples_data[samples_key]
    # check 
    samples_list = []
    mismatch_example = 0
    reference_prompt_dict = {
        example[prompt_key]: example  for example in sft_samples
    }
    for (policy_item) in tqdm(policy_samples):

        if policy_item[prompt_key] not in reference_prompt_dict:
            raise NotImplementedError
        
        sft_item = reference_prompt_dict[policy_item[prompt_key]]

        if policy_item[prompt_key] != sft_item[prompt_key]:
            mismatch_example+= 1
            continue

        samples_list.append(
            {
                prompt_key : policy_item[prompt_key],
                candidate_key : policy_item['policy'],
                baseline_key  :  sft_item['policy'],
            }
        )
    # v1
    # for (policy_item,sft_item) in tqdm(zip(policy_samples,sft_samples)):
    #     if policy_item[prompt_key] != sft_item[prompt_key]:
    #         mismatch_example+= 1

    #         raise NotImplementedError

    #     samples_list.append(
    #         {
    #             prompt_key : policy_item[prompt_key],
    #             candidate_key : policy_item['policy'],
    #             baseline_key  :  sft_item['policy'],
    #         }

    #     )
    # v2


    if len(samples_list) < 1:
        print('prompt not pair')
        raise NotImplementedError
    
    print('==================== Checking Pass! Beginning Evalutation! ====================')
    rm_model,tokenizer = load_model(rm_path)

    win_results = evaluate_win_rate(samples_list,rm_model,tokenizer,prompt_key=prompt_key,candidate_key=candidate_key,baseline_key=baseline_key)
    win_results['mismatch_example'] = mismatch_example
    result_data = dict(
        policy_samples_config = policy_samples_data,
        sft_samples_config = sft_samples_data,
        rewards_eval = win_results,
        
    )

    folder = os.path.dirname(output_path)
    os.makedirs(folder,exist_ok=True)
    print('*************save to ->',folder,output_path,os.path.basename(output_path))
    
    if os.path.basename(output_path) == '' or os.path.isdir(output_path):
        output_path = os.path.join(output_path,'winrate.json')
        print('add file name',os.path.basename(output_path),output_path)
        
    with open(output_path,'w') as f:
        json.dump(result_data,f)

    if save_sample:
        folder = os.path.dirname(output_path)
        policy_sample_path = os.path.join(folder,'policy_samples.json')
        sft_sample_path = os.path.join(folder,'sft_samples.json')
        win_path = os.path.join(folder,'single_winrate.json')
        with open(policy_sample_path,'w') as f:
            json.dump(policy_samples_data,f)
        with open(sft_sample_path,'w') as f:
            json.dump(sft_samples_data,f)
        with open(win_path,'w') as f:
            json.dump(win_results,f)
              
if __name__ == "__main__":
    main()