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
          
    

class DiffNDCGLoss(nn.Module):
    
    def __init__(self,
                    gamma = 0.9999,
                    size = 8,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    art_lambda: float = 0.25,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                 ):
        super(DiffNDCGLoss, self).__init__()
        print('set steepness->',steepness,size,gamma)
        self.sorter = DiffSortNet(
            sorting_network_type=sorting_network_type,
            size=size,
            steepness=steepness,
            art_lambda=art_lambda,
            interpolation_type = interpolation_type,
            distribution = distribution,
        )
        self.size = size
        self.gamma = gamma
        self.rank_ema = torch.zeros(size)

    def forward(self,pred_scores,labels):
        # descending order
        # ema
        self.rank_ema = self.rank_ema.to(pred_scores.device)
        rank_indices = torch.argsort(-labels, dim=-1)
        rank_true = torch.argsort(rank_indices, dim=-1)
        select_rank_ema =  self.rank_ema[rank_true] 
        sorted_scores = pred_scores.gather(-1,rank_indices)
        # pred scores
        pred_scores = pred_scores - select_rank_ema
        # update ema 
        self.rank_ema = self.gamma * self.rank_ema  + ( (1-self.gamma)*sorted_scores.mean(0) ).detach()
        
        # print('ema',self.gamma,self.rank_ema)
        # Permutation
        _ , P = self.sort(-pred_scores)
        
        ndcg_loss = 1 - diffNDCG(P,pred_scores, labels)
        
        losses = ndcg_loss 
        return losses
    
    def sort(self,outputs):
        x_hat , perm_prediction = self.sorter(outputs)
        return x_hat , perm_prediction

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
            # position r -> [1..N]
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

class SimpleDiffSortLoss(nn.Module):
    def __init__(self,
                    size = 8,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    art_lambda: float = 0.25,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                 ):
        super(SimpleDiffSortLoss, self).__init__()
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
        self.rank_ema = torch.zeros(size)
    #
    def forward(self,pred_scores,labels):
        _, P = self.sort(-pred_scores)
        ndcg_loss = 1 - diffNDCG(P,pred_scores, labels)
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
        # print('labels',labels)
        # print('P_hard',perm_ground_truth)
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
        print(f'##### init neural sort, tau = {self.tau}, hard loss:{self.hard_loss},{self.all_rank_neuralsort}')

    
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

        # print('P_hard',perm_ground_truth)
        losses = self.bce(P_hat.float(), perm_ground_truth)

        print('neural sort',losses,self.hard_loss)

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

    # @ex.capture
    # def pirank_simple_loss(labels, logits, features, tau, taustar, ndcg_k, ste):
    #     '''
    #     Modeled after tensorflow_ranking/python/losses.py _loss_fn
    #     :param labels: True scores
    #     :param logits: Scores from the NN
    #     :param tau: Temperature parameter
    #     :return:
    #     '''
    #     with tf.name_scope("pirank_scope"):
    #         false_tensor = tf.convert_to_tensor(False)
    #         evaluation = tf.placeholder_with_default(false_tensor, ())

    #         temperature = tf.cond(evaluation,
    #                             false_fn=lambda: tf.convert_to_tensor(
    #                                 tau, dtype=tf.float32),
    #                             true_fn=lambda: tf.convert_to_tensor(
    #                                 1e-10, dtype=tf.float32)  # simulate hard sort
    #                             )

    #         is_label_valid = tfr.utils.is_label_valid(labels)
    #         labels = tf.where(is_label_valid, labels, tf.zeros_like(labels))
    #         logits = tf.where(is_label_valid, logits, -1e-6 * tf.ones_like(logits) +
    #                         tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
    #         logits = tf.expand_dims(logits, 2, name="logits")
    #         labels = tf.expand_dims(labels, 2, name="labels")
    #         list_size = tf.shape(input=labels)[1]

    #         if ste:
    #             P_hat_backward = util.neuralsort(logits, temperature)
    #             P_hat_forward = util.neuralsort(logits, taustar)
    #             P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
    #         else:
    #             P_hat = util.neuralsort(logits, temperature)
    #         P_hat = tf.identity(P_hat, name="P_hat")
    #         label_powers = tf.pow(2.0, tf.cast(labels, dtype=tf.float32), name="label_powers") - 1.0
    #         sorted_powers = tf.linalg.matmul(P_hat, label_powers)

    #         numerator = tf.reduce_sum(sorted_powers, axis=-1, name="dcg_numerator")
    #         position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32, name="dcg_position")
    #         denominator = tf.math.log(position + 1, name="dcg_denominator")
    #         dcg = numerator / denominator
    #         dcg = dcg[:, :ndcg_k]
    #         dcg = tf.reduce_sum(input_tensor=dcg, axis=1, keepdims=True, name="dcg")

    #         P_true = util.neuralsort(labels, 1e-10)
    #         ideal_sorted_labels = tf.linalg.matmul(P_true, labels)
    #         ideal_sorted_labels = tf.reduce_sum(ideal_sorted_labels, axis=-1,
    #                                             name="ideal_sorted_labels")
    #         numerator = tf.pow(2.0, tf.cast(ideal_sorted_labels, dtype=tf.float32),
    #                         name="ideal_dcg_numerator") - 1.0
    #         ideal_dcg = numerator / (1e-10 + denominator)
    #         ideal_dcg = ideal_dcg[:, :ndcg_k]
    #         ideal_dcg = tf.reduce_sum(ideal_dcg, axis=1, keepdims=True, name="dcg")
    #         ndcg = tf.reduce_sum(dcg) / (1e-10 + tf.reduce_sum(ideal_dcg))
    #         return 1. - ndcg

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
        print(P_hat.shape,label_powers.shape,labels.shape)
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

        print('pirank ndcg:',ndcg)
        return 1.0 - ndcg


class AdDiffSortLoss(nn.Module):

    def __init__(self,
                    size = 8,
                    gamma = 0.9999,
                    sorting_network_type : str = 'odd_even',
                    steepness: float = 10.0,
                    interpolation_type: str = None,
                    distribution: str = 'cauchy',
                 ):
        super(AdDiffSortLoss, self).__init__()
        print('set steepness->',steepness,size,gamma)
        self.sorter = DiffSortNet(
            sorting_network_type=sorting_network_type,
            size=size,
            steepness=steepness,
            interpolation_type = interpolation_type,
            distribution = distribution,
        )
        self.gamma = gamma
        self.size = size
        self.bce = torch.nn.BCELoss()
        self.rank_ema = torch.zeros(size)
    def forward(self,pred_scores,labels):
        # Permutation
        self.rank_ema = self.rank_ema.to(pred_scores.device)
        rank_indices = torch.argsort(-labels, dim=-1)
        rank_true = torch.argsort(rank_indices, dim=-1)
        select_rank_ema =  self.rank_ema[rank_true] 
        sorted_scores = pred_scores.gather(-1,rank_indices)
        # pred scores
        pred_scores = pred_scores - select_rank_ema
        # update ema 
        self.rank_ema = self.gamma * self.rank_ema  + ( (1-self.gamma)*sorted_scores.mean(0) ).detach()

        _, perm_prediction = self.sort(-pred_scores)
        perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(-labels, dim=-1)).transpose(-2, -1).float()
        # print('labels',labels)
        # print('P_hard',perm_ground_truth)
        print(f'ema {self.gamma} :->',self.rank_ema)
        ce_loss = self.bce(perm_prediction.float(), perm_ground_truth)
        losses = ce_loss
        return losses
    
    def sort(self,outputs):
        x_hat , perm_prediction = self.sorter(outputs)
        return x_hat , perm_prediction  


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
        print('neural ndcg->',-neural_ndcg_loss)
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

class DPOLoss(nn.Module):
    def __init__(self,
                    *args, **kwargs
                 ):
        super(DPOLoss, self).__init__()

    def forward(self,pred_scores,labels):
        # bce_loss = bce(pred_scores, labels)
        # return bce_loss
        pass
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

        # print(labels)
        # print(y_true_sorted)
        # print(pred_scores)
        # print(preds_sorted_by_true)

        total_loss = 0

        for time in range(L - 1):
            neg_reward = y_true_sorted[:, time+1:] # [batch, training_stage-time-1]
            pos_reward = y_true_sorted[:, time] # [batch]
            
            eps = 1e-10
            neg_temperatures = pos_reward.view(-1, 1) - neg_reward # [batch, training_stage-time-1]
            pos_temperature = torch.max(neg_temperatures, dim=1).values # [batch]
            loss = torch.log(eps + torch.exp(preds_sorted_by_true[:, time] * pos_temperature) + torch.sum(torch.exp(preds_sorted_by_true[:, time+1:] * neg_temperatures), dim=1)) - preds_sorted_by_true[:, time] * pos_temperature # [batch]
            loss = torch.mean(loss)
            # print_loss[time].append(loss.item())
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
    # 'diffndcg': DiffNDCGLoss,   # ndcg+advantage
    
    'diffsort': PureDiffSortLoss, # No ndcg , just cross entroy
    'neuralsort': NeuralSortLoss, # No ndcg , just cross entroy
    'neuralndcg': NeuralNDCGLoss, # No ndcg , just cross entroy
    'addiffsort': AdDiffSortLoss, # No ndcg , just cross entroy, + ad
    # 'simplediffndcg':SimpleDiffSortLoss, # no advantage
    'simplediffndcg': SimpleDiffNDCGLoss, # no advantage
    'simplediffndcg_discounts': SimpleDiffNDCGLossWithDiscounts, # no advantage
    # 'diffndcgva': DiffSortLossVa, # null
    # 'diffndcgvb': DiffSortLossVb, # null
    # 'diffndcgvc': DiffSortLossVc, # null
    'dpo': DPOLoss,
    'btdpo': BTDPOLoss, # DPO BT
    'pldpo': PLDPOLoss, # DPO PL
    'lambda': LambdaLoss, # DPO Lipo
    'listnet': ListNetLoss, # DPO Lipo
    'bce': BceLoss, # DPO Lipo
    'approx': ApproxNDCGLoss, # DPO Lipo
    'pro': PROLoss, # DPO Lipo
    'pirank': PiRankLoss, # DPO Lipo
    'fastsort': FastSortLoss, # DPO Lipo
}
