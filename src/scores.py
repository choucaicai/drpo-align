
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import math
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length, trl_sanitze_kwargs_for_tagging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed



def _tdpo_get_batch_position_forward_kl(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                          average_log_prob: bool = False):

    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)

    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)

    if average_log_prob:
        return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1)
    
    return (per_position_kl * loss_mask).sum(-1)

class RankScore(nn.Module):
    def print_attributes(self):
        for attr, value in vars(self).items():
            if not attr.startswith('_'):  # 只打印非下划线开头的属性
                print(f"#######  {self.__class__.__name__} Set  {attr}: {value}")

class RMScore(RankScore):
    
    def __init__(self,
                beta = 0.1,
                **kargs
                 ):
        super(RMScore, self).__init__()
        self.beta = beta
        self.print_attributes()


    def forward(self,policy_logps=None,reference_logps=None,policy_logits=None,reference_logits=None,mean_probs=False,**kargs):

        assert policy_logps is not  None
        assert reference_logps is not None
        assert mean_probs == False

        pred_scores = self.beta * (policy_logps - reference_logps)
        
        return pred_scores,{}


class ProScore(RankScore):
    
    def __init__(self,
                    **kargs
                 ):
        super(ProScore, self).__init__()
        self.print_attributes()
        


    def forward(self,policy_logps=None,reference_logps=None,policy_logits=None,reference_logits=None,mean_probs=False,**kargs):
        assert policy_logps is not  None
        assert mean_probs == False

        pred_scores = policy_logps
        return pred_scores,{}

class ProbScore(RankScore):
    
    def __init__(self,
                    **kargs
                 ):
        super(ProbScore, self).__init__()
        self.print_attributes()
        


    def forward(self,policy_logps=None,reference_logps=None,policy_logits=None,reference_logits=None,mean_probs=False,**kargs):
        assert policy_logps is not  None
        assert mean_probs == True

        pred_scores = policy_logps
        return pred_scores,{}
    
class ProbRankScore(RankScore):
    
    def __init__(self,
                 rank_coef=0.2,
                    **kargs
                 ):
        super(ProbRankScore, self).__init__()
        self.rank_coef = rank_coef

        self.print_attributes()


    def forward(self,policy_logps=None,reference_logps=None,label_scores=None,mean_probs=False,**kargs):
        assert policy_logps is not  None
        assert label_scores is not None
        assert mean_probs == True

        pred_scores = policy_logps

        rank_indices = torch.argsort(-label_scores, dim=-1)
        rank_true = torch.argsort(rank_indices, dim=-1)
        pred_scores = pred_scores + rank_true * self.rank_coef

        print('score:',pred_scores,rank_true,self.rank_coef)
        return pred_scores,{}

class ProbKLScore(RankScore):
    
    def __init__(self,
                    beta = 0.1,
                    kl = 0.5,
                    **kargs
                 ):
        super(ProbKLScore, self).__init__()
        self.beta = beta
        self.kl = kl

        self.print_attributes()
        


    def forward(self,policy_logps=None,policy_logits=None,reference_logits=None,label_scores=None,mean_probs=False,**kargs):

        assert policy_logps is not  None
        assert mean_probs == True
        assert policy_logits is not None
        assert reference_logits is not None
        assert label_scores is not None

        N,R = policy_logps.shape
        mean_logps = policy_logps

        forward_kl = _tdpo_get_batch_position_forward_kl(policy_logits,reference_logits,labels=label_scores.reshape(N*R, -1),average_log_prob=self.mean_logps)  
        rank_forward_kl = forward_kl.reshape(N,R).detach()
        pred_scores = self.beta * mean_logps + self.kl * rank_forward_kl

        return pred_scores,{}

class AdScore(RankScore):
    
    def __init__(self,
                    beta = 0.1,
                    **kargs
                 ):
        super(AdScore, self).__init__()
        self.beta = beta

        self.print_attributes()

    def forward(self,policy_logps,reference_logps=None,policy_logits=None,reference_logits=None,label_scores=None,mean_probs=False):

        assert reference_logps is not None
        assert policy_logits is not None
        assert reference_logits is not None
        assert label_scores is not None

        N,R = policy_logps.shape
        logps_margin  = (policy_logps - reference_logps)
        forward_kl = _tdpo_get_batch_position_forward_kl(policy_logits,reference_logits,labels=label_scores.reshape(N*R, -1) ,average_log_prob = mean_probs)
        rank_forward_kl = forward_kl.reshape(N,R).detach()
        pred_scores = self.beta * (logps_margin + rank_forward_kl)

        return pred_scores,{}

class AdScore(RankScore):
    
    def __init__(self,
                    beta = 0.1,
                    **kargs
                 ):
        super(ProbKLScore, self).__init__()
        self.beta = beta

        self.print_attributes()

    def forward(self,policy_logps=None,reference_logps=None,policy_logits=None,reference_logits=None,label_scores=None,mean_probs=False,**kargs):

        assert policy_logps is not  None
        assert reference_logps is not None
        assert policy_logits is not None
        assert reference_logits is not None
        assert label_scores is not None

        N,R = policy_logps.shape
        logps_margin  = (policy_logps - reference_logps)
        forward_kl = _tdpo_get_batch_position_forward_kl(policy_logits,reference_logits,labels=label_scores.reshape(N*R, -1) ,average_log_prob = mean_probs)
        rank_forward_kl = forward_kl.reshape(N,R).detach()
        pred_scores = self.beta * (logps_margin + rank_forward_kl)

        return pred_scores,{}

class AdaMScore(RankScore):
    
    def __init__(self,
                    ada_coef = 1,
                    gamma = 0.9999,
                    size :int = 8,
                    clip_score: float = 1,
                    **kargs
                 ):
        super(AdaMScore, self).__init__()
        
        self.ada_coef = ada_coef
        self.gamma = gamma
        self.rank_ema = torch.zeros(size=(size,))
        self.clip_score = clip_score
        
        self.print_attributes()

    def forward(self,pred_scores,label_scores=None,**kargs):

        assert label_scores is not None

        pred_scores = pred_scores
        # adaptive margin
        rank_indices = torch.argsort(-label_scores, dim=-1)
        rank_true = torch.argsort(rank_indices, dim=-1)
        sorted_scores = pred_scores.gather(-1,rank_indices)
        
        # if self.rank_ema is None:
        #     self.rank_ema = sorted_scores.cpu().mean(0).detach().clone()

        now_rank_ema = self.rank_ema.to(pred_scores.device)
        select_rank_ema = now_rank_ema[rank_true]

        pred_scores = pred_scores - self.ada_coef * select_rank_ema

        # update

        self.rank_ema = self.gamma * self.rank_ema  + ( (1-self.gamma)*sorted_scores.cpu().mean(0) ).detach()

        max_ema = now_rank_ema[0].detach().squeeze()
        min_ema = now_rank_ema[-1].detach().squeeze()
        
        metrics = {}
        metrics[f"ema_score/max"] = max_ema.detach().cpu().mean()
        metrics[f"ema_score/min"] = min_ema.detach().cpu().mean()
        metrics[f"ema_score/margin"] = max_ema.detach().cpu().mean() - min_ema.detach().cpu().mean()


        # 0.1 -> 0.8, [-0.8,0.8] - > mean + r*0.8-0.4
        # print(now_rank_ema.shape,now_rank_ema,max_ema.detach().cpu().mean(),min_ema.detach().cpu().mean())
        # print(label_scores)
        # print(rank_true)
        print('ema -> ',self.gamma,self.rank_ema,)

        return pred_scores,metrics

class AdaMScoreV2(AdaMScore):
    
    def __init__(self,
                    ada_coef = 1,
                    gamma = 0.9999,
                    tau = 0.1,
                    size :int = 8,
                    clip_score: float = 1,
                    **kargs
                 ):
        super(AdaMScoreV2, self).__init__()
        
        self.ada_coef = ada_coef
        self.gamma = gamma
        self.size = size
        self.rank_ema = torch.zeros(size=(size,))
        self.clip_score = clip_score
        self.tau = tau
        self.print_attributes()

    def forward(self,pred_scores,label_scores=None,**kargs):

        assert label_scores is not None

        pred_scores = pred_scores
        # adaptive margin
        rank_indices = torch.argsort(-label_scores, dim=-1)
        rank_true = torch.argsort(rank_indices, dim=-1)
        sorted_scores = pred_scores.gather(-1,rank_indices)
        
        # if self.rank_ema is None:
        #     self.rank_ema = sorted_scores.cpu().mean(0).detach().clone()

        now_rank_ema = self.rank_ema.to(pred_scores.device)
        select_rank_ema = now_rank_ema[rank_true]

        mean = select_rank_ema.mean()
        std = select_rank_ema.std()
        select_rank_ema = (select_rank_ema - mean) / ( std+0.000001)
        select_rank_ema = select_rank_ema * torch.clip(std,0,(self.size * self.tau / 6))


        pred_scores = pred_scores - self.ada_coef * select_rank_ema

        # update

        self.rank_ema = self.gamma * self.rank_ema  + ( (1-self.gamma)*sorted_scores.cpu().mean(0) ).detach()

        max_ema = now_rank_ema[0].detach().squeeze()
        min_ema = now_rank_ema[-1].detach().squeeze()
        
        metrics = {}
        metrics[f"ema_score/max"] = max_ema.detach().cpu().mean()
        metrics[f"ema_score/min"] = min_ema.detach().cpu().mean()
        metrics[f"ema_score/margin"] = max_ema.detach().cpu().mean() - min_ema.detach().cpu().mean()
        metrics[f"ema_score/mean"] = mean.detach().cpu().mean()
        metrics[f"ema_score/std"] = std.detach().cpu().mean()


        # 0.1 -> 0.8, [-0.8,0.8] - > mean + r*0.8-0.4
        # print(now_rank_ema.shape,now_rank_ema,max_ema.detach().cpu().mean(),min_ema.detach().cpu().mean())
        # print(label_scores)
        # print(rank_true)
        print('ema -> ',self.gamma,self.rank_ema,select_rank_ema,mean.detach().cpu().mean(),std.detach().cpu().mean(),(self.size * self.tau / 6))

        return pred_scores,metrics

class CombineAdaMScore(nn.Module):

    def __init__(self,
                    score_fn=None,
                    adam_score:AdaMScore = None,
                    **kargs
                 ):
        super(CombineAdaMScore, self).__init__()
        
        assert isinstance(adam_score,AdaMScore) or adam_score is None, 'adam_score should be ADAMScore or None, please modify your config'
        self.score = score_fn
        self.adam_score = adam_score

    def forward(self,*args,**kargs):
        
        final_metrics = {}
        # print(kargs)
        pred_scores,metrics = self.score(*args,**kargs)
        final_metrics.update(metrics)
        if self.adam_score is not None:
            pred_scores,metrics = self.adam_score(pred_scores,**kargs)
            final_metrics.update(metrics)

        return pred_scores,final_metrics



RANK_SCORE_TYPE = {
    'dpo': RMScore,
    'imscore': RMScore,
    'pro': ProScore,
    'prob': ProbScore,
    'prob+rank': ProbRankScore,
    'prob+kl': ProbKLScore,
    'ad': AdScore,
    'adam': AdaMScore,
    'adam2': AdaMScoreV2,
    'com+adam': CombineAdaMScore,
}

def list_get(lst, index, default={}):
    return lst[index] if len(lst) > index else default

def get_score_fn(score_name=None,score_config=None):

    if score_name is None:
        return None

    assert score_name in RANK_SCORE_TYPE

    score_cls = RANK_SCORE_TYPE[score_name]

    if issubclass(score_cls,CombineAdaMScore):
        assert isinstance(score_config,list), "score config should be a list"
        assert len(score_config) >= 1 or len(score_config)<=2, "combine scores not just support list size 2"

        score_fn = get_score_fn( **list_get(score_config,0) )
        adam_score = get_score_fn( **list_get(score_config,1) )
        
        return score_cls(score_fn=score_fn,adam_score=adam_score)

    assert isinstance(score_config,dict), "score config should be a dict"
    return score_cls(**score_config)

if __name__ == "__main__":

    # s = RMScore(beta=0.01)
    # ada = AdaMScore()
    # com = CombineAdaMScore(
    #     s,ada
    # )
    s = get_score_fn(
        score_name = 'com+adam',
        score_config = [{            
                "score_name" : 'dpo',
                "score_config" : {} ,
            },
        ]
    )
    
    test_scores = torch.arange(8,).reshape(1,8)
    zero_scores = torch.zeros(8).reshape(1,8)
    label_scores = torch.zeros(8).reshape(1,8)
    print(test_scores.shape,zero_scores.shape)
    res = s(policy_logps=test_scores,reference_logps=zero_scores,label_scores=label_scores)





