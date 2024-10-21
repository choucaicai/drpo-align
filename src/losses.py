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

from .rank_utils import (
    DiffSortNet,
    diffNDCG,
    lambdaLoss,
    rankNetLoss,
    listMLE, 
    listNet,bce,
    approxNDCGLoss,
    NeuralSort,
    neuralNDCGLoss,
    deterministic_neural_sort,
    PADDED_Y_VALUE,
    fast_sort_loss,
    diffNDCG_discounts,
)


class SimpleDiffNDCGLoss(nn.Module):
    
    def __init__(self,
                    size = 8,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    art_lambda: float = 0.25,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                 ):
        super(SimpleDiffNDCGLoss, self).__init__()
        # print('set steepness->',steepness,size)
        self.sorter = DiffSortNet(
            sorting_network_type=sorting_network_type,
            size=size,
            steepness=steepness,
            art_lambda=art_lambda,
            interpolation_type = interpolation_type,
            distribution = distribution,
        )
        self.size = size

    def forward(self,pred_scores,labels):

        _, P = self.sort(-pred_scores)
        ndcg_loss = 1 - diffNDCG(P,pred_scores, labels)
        losses = ndcg_loss 
        return losses
    
    def sort(self,outputs):
        x_hat , perm_prediction = self.sorter(outputs)
        return x_hat , perm_prediction

class SimpleDiffNDCGLossWithDiscounts(nn.Module):
    
    def __init__(self,
                    size = 8,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    art_lambda: float = 0.25,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                    discounts_type: str = 'log'
                 ):
        super(SimpleDiffNDCGLossWithDiscounts, self).__init__()
        print('set steepness->',steepness,size)
        self.sorter = DiffSortNet(
            sorting_network_type=sorting_network_type,
            size=size,
            steepness=steepness,
            art_lambda=art_lambda,
            interpolation_type = interpolation_type,
            distribution = distribution,
        )
        self.size = size
        self.discounts_type=  discounts_type

    def forward(self,pred_scores,labels):
        dev = pred_scores.device
        _, P = self.sort(-pred_scores)
        if self.discounts_type == 'log':
            discounts  = None  # 1 / log(1+r) ,
        elif self.discounts_type == 'line':
            discounts = (torch.tensor(1.) / (torch.arange(labels.shape[-1], dtype=torch.float) + 1.)  ).to(dev)    # 1 / r
        elif self.discounts_type == 'sqrt':
            discounts = (torch.tensor(1.) / ( ( torch.arange(labels.shape[-1], dtype=torch.float) + 1.) ** 0.5 ) ) .to(dev)    # 1 / sqrt(r)
        elif self.discounts_type == 'square':
            discounts = (torch.tensor(1.) / ( ( torch.arange(labels.shape[-1], dtype=torch.float) + 1.) ** 2 ) ) .to(dev)    # 1 / r^2
        elif self.discounts_type == 'four':
            discounts = (torch.tensor(1.) / ( ( torch.arange(labels.shape[-1], dtype=torch.float) + 1.) ** 4 ) ) .to(dev)    # 1 / r^4
        elif self.discounts_type == 'eight':
            discounts = (torch.tensor(1.) / ( ( torch.arange(labels.shape[-1], dtype=torch.float) + 1.) ** 8 ) ) .to(dev)    # 1 / r^8
        else:
            raise NotImplementedError
        ndcg_loss = 1 - diffNDCG_discounts(P,pred_scores, labels,discounts=discounts)
        losses = ndcg_loss 
        return losses
    
    def sort(self,outputs):
        x_hat , perm_prediction = self.sorter(outputs)
        return x_hat , perm_prediction


class PureDiffSortLoss(nn.Module):
    def __init__(self,
                    size = 8,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                 ):
        super(PureDiffSortLoss, self).__init__()
        print('set steepness->',steepness,size)
        self.sorter = DiffSortNet(
            sorting_network_type=sorting_network_type,
            size=size,
            steepness=steepness,
            interpolation_type = interpolation_type,
            distribution = distribution,
        )
        self.size = size
        self.bce = torch.nn.BCELoss()
    #
    def forward(self,pred_scores,labels):
        # Permutation
        _, perm_prediction = self.sort(-pred_scores)
        perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(-labels, dim=-1)).transpose(-2, -1).float()
        ce_loss = self.bce(perm_prediction.float(), perm_ground_truth)
        losses = ce_loss
        return losses
    
    def sort(self,outputs):
        x_hat , perm_prediction = self.sorter(outputs)
        return x_hat , perm_prediction

class NeuralSortLoss(nn.Module):
    def __init__(self,
                    tau = 1.0,
                    hard_loss = True,
                    all_rank_neuralsort = False,
                    *args, **kwargs
                 ):
        super(NeuralSortLoss, self).__init__()
        self.tau = tau
        self.hard_loss = hard_loss
        self.all_rank_neuralsort = all_rank_neuralsort
        self.sorter = deterministic_neural_sort if all_rank_neuralsort else NeuralSort(tau=self.tau)
        self.bce = torch.nn.BCELoss()

    
    def forward(self,pred_scores,labels):
        # Permutation
        if self.all_rank_neuralsort:
            mask = (labels == PADDED_Y_VALUE)
            P_hat = deterministic_neural_sort(pred_scores.unsqueeze(-1), tau=self.tau, mask=mask)
        else:
            P_hat = self.sorter(pred_scores)

        if self.hard_loss:
            perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(labels, dim=-1)).transpose(-2, -1).float()
        else:
            if self.all_rank_neuralsort:
                mask = (labels == PADDED_Y_VALUE)
                perm_ground_truth = deterministic_neural_sort(labels.unsqueeze(-1), tau=self.tau, mask=mask)
            else:
                perm_ground_truth = self.sorter(labels)
        losses = self.bce(P_hat.float(), perm_ground_truth)

        # print('neural sort',losses,self.hard_loss)

        return losses
    
class PiRankLoss(nn.Module):
    def __init__(self,
                    tau = 1.0,
                    all_rank_neuralsort=False,
                    *args, **kwargs
                 ):
        super(PiRankLoss, self).__init__()
        self.all_rank_neuralsort = all_rank_neuralsort
        self.tau = tau
        self.sorter = deterministic_neural_sort if all_rank_neuralsort else NeuralSort(tau=self.tau)
        print(f'##### init PiRankLoss, tau = {self.tau}')

    def forward(self,pred_scores,labels):
        # We reimplemented the original Pirank TensorFlow implementation in PyTorch.
        
        list_size = labels.shape[1]

        # P_hat = self.neural_sort(pred_scores)
        if self.all_rank_neuralsort:
            mask = (labels == PADDED_Y_VALUE)
            P_hat = deterministic_neural_sort(pred_scores.unsqueeze(-1), tau=self.tau, mask=mask)
        else:
            P_hat = self.sorter(pred_scores)

        
        label_powers = torch.pow(2.0, labels.float()) - 1.0
        # print(P_hat.shape,label_powers.shape,labels.shape)
        sorted_powers = torch.matmul(P_hat.float(), label_powers.unsqueeze(-1))

        numerator = torch.sum(sorted_powers, dim=-1)
            
        position = torch.arange(1, list_size + 1, dtype=torch.float32, device=sorted_powers.device)
        #  dcg_denominator
        denominator = torch.log2(position + 1)
        dcg = numerator / denominator
        # dcg = dcg[:, :ndcg_k]
        dcg = torch.sum(dcg, dim=1, keepdim=True)

        P_true = self.sorter(labels)
        ideal_sorted_labels = torch.matmul(P_true.float(), labels.unsqueeze(-1))
        ideal_sorted_labels = torch.sum(ideal_sorted_labels, dim=-1)

        ideal_numerator = torch.pow(2.0, ideal_sorted_labels.float()) - 1.0
        ideal_dcg = ideal_numerator / (1e-10 + denominator)
        # ideal_dcg = ideal_dcg[:, :ndcg_k]
        ideal_dcg = torch.sum(ideal_dcg, dim=1, keepdim=True)

        ndcg = torch.sum(dcg) / (1e-10 + torch.sum(ideal_dcg))

        # print('pirank ndcg:',ndcg)
        return 1.0 - ndcg

class LambdaLoss(nn.Module):
    def __init__(self,
                    weighing_scheme='lambdaRank_scheme',
                    *args, **kwargs
                 ):
        super(LambdaLoss, self).__init__()
        self.weighing_scheme = weighing_scheme

    def forward(self,pred_scores,labels):
        losses = lambdaLoss(pred_scores, labels,weighing_scheme=self.weighing_scheme);
        return losses
    
class BTDPOLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(BTDPOLoss, self).__init__()

    def forward(self,pred_scores,labels):
        bt_dpo_loss = rankNetLoss(pred_scores, labels)
        return bt_dpo_loss
class NeuralNDCGLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(NeuralNDCGLoss, self).__init__()

    def forward(self,pred_scores,labels):
        neural_ndcg_loss = neuralNDCGLoss(pred_scores, labels)
        # print('neural ndcg->',-neural_ndcg_loss)
        return neural_ndcg_loss
    
class PLDPOLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(PLDPOLoss, self).__init__()

    def forward(self,pred_scores,labels):
        pl_dpo_loss = listMLE(pred_scores, labels)
        return pl_dpo_loss
    
class ListNetLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(ListNetLoss, self).__init__()

    def forward(self,pred_scores,labels):
        listnet_loss = listNet(pred_scores, labels)
        return listnet_loss
    
class BceLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(BceLoss, self).__init__()

    def forward(self,pred_scores,labels):
        bce_loss = bce(pred_scores, labels)
        return bce_loss


class PROLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(PROLoss, self).__init__()

    def forward(self,pred_scores,labels):
        B,L = labels.shape
        # print('labels',labels.shape)
        # print('pred_scores',pred_scores.shape)
        y_true_sorted, indices = labels.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(pred_scores, dim=1, index=indices)

        total_loss = 0

        for time in range(L - 1):
            neg_reward = y_true_sorted[:, time+1:] # [batch, training_stage-time-1]
            pos_reward = y_true_sorted[:, time] # [batch]
            
            eps = 1e-10
            neg_temperatures = pos_reward.view(-1, 1) - neg_reward # [batch, training_stage-time-1]
            pos_temperature = torch.max(neg_temperatures, dim=1).values # [batch]
            loss = torch.log(eps + torch.exp(preds_sorted_by_true[:, time] * pos_temperature) + torch.sum(torch.exp(preds_sorted_by_true[:, time+1:] * neg_temperatures), dim=1)) - preds_sorted_by_true[:, time] * pos_temperature # [batch]
            loss = torch.mean(loss)
            total_loss += loss

        return total_loss


class ApproxNDCGLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(ApproxNDCGLoss, self).__init__()

    def forward(self,pred_scores,labels):
        approx_loss = approxNDCGLoss(pred_scores, labels)
        return approx_loss
    
class FastSortLoss(nn.Module):
    def __init__(self,
                    regularization_strength = 1.0,
                    *args, **kwargs
                 ):
        super(FastSortLoss, self).__init__()
        self.regularization_strength = regularization_strength

    def forward(self,pred_scores,labels):
        return fast_sort_loss(pred_scores,labels, regularization_strength= self.regularization_strength)
    
RANK_LOSS_TYPE = {
    'diffsort': PureDiffSortLoss, # No ndcg , just cross entroy
    'neuralsort': NeuralSortLoss, 
    'neuralndcg': NeuralNDCGLoss,
    'simplediffndcg': SimpleDiffNDCGLoss, 
    'simplediffndcg_discounts': SimpleDiffNDCGLossWithDiscounts,
    'btdpo': BTDPOLoss,
    'pldpo': PLDPOLoss,
    'lambda': LambdaLoss,
    'listnet': ListNetLoss,
    'bce': BceLoss,
    'approx': ApproxNDCGLoss, 
    'pro': PROLoss,
    'pirank': PiRankLoss,
    'fastsort': FastSortLoss,
}
