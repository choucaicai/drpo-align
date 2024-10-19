from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import product
from torch.nn import BCELoss,BCEWithLogitsLoss
import torch
from torch import Tensor

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -100

SORTING_NETWORK_TYPE = List[torch.tensor]


def s_best(x):
    return torch.clamp(x, -0.25, 0.25) + .5 + \
        ((x > 0.25).float() - (x < -0.25).float()) * (0.25 - 1/16/(x.abs()+1e-10))


class NormalCDF(torch.autograd.Function):
    def forward(ctx, x, sigma):
        ctx.save_for_backward(x, torch.tensor(sigma))
        return 0.5 + 0.5 * torch.erf(x / sigma / math.sqrt(2))

    def backward(ctx, grad_y):
        x, sigma = ctx.saved_tensors
        return grad_y * 1 / sigma / math.sqrt(math.pi * 2) * torch.exp(-0.5 * (x/sigma).pow(2)), None


def execute_sort(
        sorting_network,
        vectors,
        steepness=10.,
        art_lambda=0.25,
        distribution='cauchy'
):
    x = vectors
    X = torch.eye(vectors.shape[1], dtype=x.dtype, device=x.device).repeat(x.shape[0], 1, 1)
    for split_a, split_b, combine_min, combine_max in sorting_network:

        split_a = split_a.type(x.dtype)
        split_b = split_b.type(x.dtype)
        combine_min = combine_min.type(x.dtype)
        combine_max = combine_max.type(x.dtype)

        a, b = x @ split_a.T, x @ split_b.T

        # float conversion necessary as PyTorch doesn't support Half for sigmoid as of 25. August 2021
        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == 'logistic':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness).type(x.dtype)

        elif distribution == 'logistic_phi':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness / ((a-b).type(new_type).abs() + 1.e-10).pow(art_lambda)).type(x.dtype)

        elif distribution == 'gaussian':
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == 'reciprocal':
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == 'cauchy':
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + .5
            alpha = alpha.type(x.dtype)
        elif distribution == 'optimal':
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        aX = X @ split_a.T
        bX = X @ split_b.T
        w_min = alpha.unsqueeze(-2) * aX + (1-alpha).unsqueeze(-2) * bX
        w_max = (1-alpha).unsqueeze(-2) * aX + alpha.unsqueeze(-2) * bX

        X = (w_max @ combine_max.T.unsqueeze(-3)) + (w_min @ combine_min.T.unsqueeze(-3))
        
        x = (alpha * a + (1-alpha) * b) @ combine_min.T + ((1-alpha) * a + alpha * b) @ combine_max.T

    return x, X    

def bitonic_network(n):
    IDENTITY_MAP_FACTOR = .5
    num_blocks = math.ceil(np.log2(n))
    assert n <= 2 ** num_blocks
    print('bitonic_network blocks',num_blocks)
    network = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)

            split_a, split_b = np.zeros((n, 2**num_blocks)), np.zeros((n, 2**num_blocks))
            combine_min, combine_max = np.zeros((2**num_blocks, n)), np.zeros((2**num_blocks, n))
            count = 0

            for i in range(0, 2**num_blocks, 2*m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m

                    # Cases to handle n \neq 2^k: The top wires are discarded and if a comparator considers them, the
                    # comparator is ignored.
                    if a >= 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, a], split_b[count, b] = 1, 1
                        if (ix // 2**(block_idx + 1)) % 2 == 1:
                            a, b = b, a
                        combine_min[a, count], combine_max[b, count] = 1, 1
                        count += 1
                    elif a < 2**num_blocks-n and b < 2**num_blocks-n:
                        pass
                    elif a >= 2**num_blocks-n and b < 2**num_blocks-n:
                        split_a[count, a], split_b[count, a] = 1, 1
                        combine_min[a, count], combine_max[a, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    elif a < 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, b], split_b[count, b] = 1, 1
                        combine_min[b, count], combine_max[b, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    else:
                        assert False

            split_a = split_a[:count, 2 ** num_blocks - n:]
            split_b = split_b[:count, 2 ** num_blocks - n:]
            combine_min = combine_min[2**num_blocks-n:, :count]
            combine_max = combine_max[2**num_blocks-n:, :count]
            network.append((split_a, split_b, combine_min, combine_max))

    return network


def odd_even_network(n):
    layers = n

    network = []

    shifted: bool = False
    even: bool = n % 2 == 0

    for layer in range(layers):

        if even:
            k = n // 2 + shifted
        else:
            k = n // 2 + 1
        # print(k,even)
        split_a, split_b = np.zeros((k, n)), np.zeros((k, n))
        combine_min, combine_max = np.zeros((n, k)), np.zeros((n, k))
        count = 0

        # for i in range(n // 2 if not (even and shifted) else n // 2 - 1):
        for i in range(int(shifted), n-1, 2):
            a, b = i, i + 1
            # print('a , b , i ->',a,b,i)

            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 1, 1
            count += 1
        # print(split_a)
        # print(combine_min)
        if even and shifted:
            # print('even and shift')
            # Make sure that the corner values stay where they are/were:
            a, b = 0, 0
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1
            a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1
            # print(even,shifted,split_a)
            # print(even,shifted,combine_min)

        elif not even:
            # print('not even')
            if shifted:
                a, b = 0, 0
            else:
                a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1
        assert count == k
        network.append((split_a, split_b, combine_min, combine_max))
        shifted = not shifted

    return network
def get_sorting_network(type, n, device):
    def matrix_to_torch(m):
        return [[torch.from_numpy(matrix).float().to(device) for matrix in matrix_set] for matrix_set in m]
    if type == 'bitonic':
        return matrix_to_torch(bitonic_network(n))
    elif type == 'odd_even':
        return matrix_to_torch(odd_even_network(n))
    else:
        raise NotImplementedError('Sorting network `{}` unknown.'.format(type))

def sort(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

    Positional arguments:
    sorting_network
    vectors -- the matrix to sort along axis 1; sorted in-place

    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for logistic_phi interpolation (default 0.25)
    distribution -- how to interpolate when swapping two numbers; (default 'cauchy')
    """
    assert sorting_network[0][0].device == vectors.device, (
        f"The sorting network is on device {sorting_network[0][0].device} while the vectors are on device"
        f" {vectors.device}, but they both need to be on the same device."
    )
    return execute_sort(
        sorting_network=sorting_network,
        vectors=vectors,
        steepness=steepness,
        art_lambda=art_lambda,
        distribution=distribution
    )

class DiffSortNet(torch.nn.Module):
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

    Positional arguments:
    sorting_network_type -- which sorting network to use for sorting.
    vectors -- the matrix to sort along axis 1; sorted in-place

    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for sigmoid_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
                 (default 'logistic_phi')
    """
    def __init__(
        self,
        sorting_network_type: str,
        size: int,
        device: str = 'cpu',
        steepness: float = 100,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = 'cauchy',
    ):
        super(DiffSortNet, self).__init__()
        self.sorting_network_type = sorting_network_type
        self.size = size

        self.sorting_network = get_sorting_network(sorting_network_type, size, device)

        if interpolation_type is not None:
            assert distribution is None or distribution == 'cauchy' or distribution == interpolation_type, (
                'Two different distributions have been set (distribution={} and interpolation_type={}); however, '
                'they have the same interpretation and interpolation_type is a deprecated argument'.format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution
    def shard_device(self,m,device):
        return [[matrix.float().to(device) for matrix in matrix_set] for matrix_set in m]

    def forward(self, vectors):
        assert len(vectors.shape) == 2
        
        # print(vectors.shape,vectors.shape[1],self.size)
        assert vectors.shape[1] == self.size
        sorting_network = self.shard_device(self.sorting_network,vectors.device)
        return sort(
            sorting_network, vectors, self.steepness, self.art_lambda, self.distribution
        )


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0
    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)
    
    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains*discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)
    # print('ats',ats_tensor)
    dcg = cum_dcg[:, ats_tensor]
    
    # print('dcg',dcg.shape)
    return dcg

def dcg_bydiscounts(y_pred, y_true,discounts = None, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    if discounts is None:
        discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
            device=true_sorted_by_preds.device)
        
    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains*discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)
    # print('ats',ats_tensor)
    dcg = cum_dcg[:, ats_tensor]
    
    # print('dcg',dcg.shape)
    return dcg

def diffNDCG(P,y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, log_scores=True):
    """
    """
    dev = y_pred.device

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # print('mask',mask)
    # print("=============P    =============")
    # print(P_hat.shape)
    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices

    # Mask P_hat and apply to true labels, ie approximately sort them
    # print("=============P end=============")
    
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1)
    
    # print(y_true_masked)
    # modify to fit llm reward reward [-B,B] 2^r to make gain positive
    y_true_masked = torch.pow(2., y_true_masked) - 1.

    # print(P)
    # print(P.shape,y_true_masked.shape)
    # print('y_true_masked:',y_true_masked)
    ground_truth = torch.matmul(P.transpose(-2,-1).float(), y_true_masked.float() ).squeeze(-1)
    # print('y_true',y_true_masked,y_true_masked.shape)
    # print('res',ground_truth,ground_truth.shape)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    # print('discounts',discounts.shape)
    discounted_gains = ground_truth * discounts
    
    # print('discounted_gains',discounted_gains.shape,discounted_gains)
    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        # idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: torch.pow(2, x) -1 ).permute(1, 0)
    
    # print('idcg',idcg.shape)
    # print('discounted_gains',discounted_gains.shape)
    discounted_gains = discounted_gains[:, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    # print('y_true',y_true)
    # print('y_pred',y_pred)
    # print('per',ground_truth)
    # print('dis gains',discounted_gains)
    # print('ndcg->',ndcg,discounted_gains.sum(dim=-1),idcg)
    idcg_mask = idcg == 0.
    # print('ndcg ->',discounted_gains.sum(dim=-1).shape, idcg.shape,ndcg.shape)
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)
    # print('nnnn',discoun/ted_gains.sum(dim=-1),idcg,ndcg)
    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    
    if idcg_mask.all():
        print('y_pred',y_pred)
        print('y_true',y_true)
        print('ndcg',ndcg)
        print(idcg_mask)
        return torch.tensor(0.).to(y_pred.device)

    # print('loss num mean',(~idcg_mask).sum(),ndcg.shape[0])
    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return mean_ndcg





def diffNDCG_discounts(P,y_pred, y_true,discounts=None, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, log_scores=True):
    """
    """
    dev = y_pred.device

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # print('mask',mask)
    # print("=============P    =============")
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1)
    
    # print(y_true_masked)
    # modify to fit llm reward reward [-B,B] 2^r to make gain positive
    y_true_masked = torch.pow(2., y_true_masked) - 1.


    ground_truth = torch.matmul(P.transpose(-2,-1).float(), y_true_masked.float() ).squeeze(-1)
    # print('y_true',y_true_masked,y_true_masked.shape)
    # print('res',ground_truth,ground_truth.shape)
    if discounts is None:
        discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)

    # print('discounts',discounts.shape)
    discounted_gains = ground_truth * discounts
    
    if powered_relevancies:
        idcg = dcg_bydiscounts(y_true, y_true,discounts=discounts, ats=[k]).permute(1, 0)
    else:
        # idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)
        idcg = dcg_bydiscounts(y_true, y_true,discounts=discounts, ats=[k], gain_function=lambda x: torch.pow(2, x) -1 ).permute(1, 0)
    
    discounted_gains = discounted_gains[:, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)

    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)
    # print('nnnn',discoun/ted_gains.sum(dim=-1),idcg,ndcg)
    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    
    if idcg_mask.all():
        print('y_pred',y_pred)
        print('y_true',y_true)
        print('ndcg',ndcg)
        print(idcg_mask)
        return torch.tensor(0.).to(y_pred.device)

    # print('loss num mean',(~idcg_mask).sum(),ndcg.shape[0])
    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return mean_ndcg

def lambdaLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    
    # print('y_pred ',y_pred)
    # print('y_true ',y_true)
    padded_mask = y_true == padded_value_indicator
    # print('padded_mask ',padded_value_indicator,padded_mask)
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)


    # print('y_pred_sorted ', y_pred_sorted)
    # print('y_true_sorted' , y_true_sorted)
    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # print('maxDCGs',maxDCGs);
    
    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = ( y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :] ).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss

def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNetLoss(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # print(pairs_true)
    # print(selected_pred.shape)
    # print(selected_pred)
    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]
    loss = - F.logsigmoid(pred_diffs)
    # print('loss',loss.shape,loss)
    return loss.mean()


def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")
    
    print('preds_sorted_by_true',preds_sorted_by_true.shape)


    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    print('max_pred_values',max_pred_values.shape)


    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    print('preds_sorted_by_true_minus_max',preds_sorted_by_true_minus_max.shape)

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    print('cumsums',cumsums.shape)

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):

    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)


def bce(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Binary Cross-Entropy loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    # print('pred true')
    # print(y_pred,y_pred.device)
    # print(y_true,y_true.device)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
    # print('sum_valid_ls',ls)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=y_true.device)
    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)
    return loss_output




def deterministic_neural_sort(s, tau, mask):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """
    # print('dsort------------')
    dev = s.device
    s = s.type(torch.float32)
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    # print(s)
    s = s.masked_fill(mask[:, :, None], -1e8)
    # print('masked',s)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)
    # print(A_s.shape)
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    # print('B',B.shape)

    # print(mask.squeeze(-1).sum(dim=1).shape)
    temp = [ n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore
    # print('scaleing',scaling.shape)
    s = s.masked_fill(mask[:, :, None], 0.0)
    # print('before c',s.shape,scaling.unsqueeze(-2).shape)
    C = torch.matmul(s, scaling.unsqueeze(-2))
    # print('C:',C.shape)
    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    # print('P_max',P_max.shape)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    # print('dsort------------')
    return P_hat
    


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat
def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat

def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0
    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)

def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)
    # print('ats',ats_tensor)
    dcg = cum_dcg[:, ats_tensor]
    
    # print('dcg',dcg.shape)
    return dcg

def neuralNDCGLoss(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    
    :return: loss value, a torch.Tensor
    """
    dev = y_pred.device
    # print(y_pred.shape, y_true.shape)

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # print('mask',mask)
    if stochastic:
        raise NotImplementedError
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    # print('y_true',y_true)

    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.
    else:
        y_true_masked = torch.pow(2., y_true_masked) - 1.
        
    # print('y_true_masked',y_true_masked,P_hat.shape)

    ground_truth = torch.matmul(P_hat, y_true_masked.float()).squeeze(-1)

    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts
    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: torch.pow(2, x) -1 ).permute(1, 0)
    
    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    
    if idcg_mask.all():
        return torch.tensor(0.)
    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore

    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG



class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.FloatTensor(dim, 1).fill_(1).to(scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.FloatTensor).to(scores.device)
        
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat,).to(P_hat.device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.LongTensor).to(scores.device)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.LongTensor).to(scores.device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat
            
        return P_hat


import torch
from src.alignment.fast_soft_sort.pytorch_ops import soft_rank, soft_sort
# from fast_soft_sort.pytorch_ops import soft_rank, soft_sort

def fast_sort_loss(pred_scores,labels,regularization_strength=1.0):

    # descending
    res= soft_rank(-pred_scores, regularization_strength = regularization_strength)
    sorted_indices = torch.argsort(-labels)
    label_ranks = torch.argsort(sorted_indices) + 1

    loss = torch.mean(  0.5*( (res - label_ranks) ** 2 )  ) 
    
    # print(sorted_indices)

    return loss

# def calculate_pairwise_accuracy(tensor):
#     # 确保输入是 2D 张量
#     if tensor.dim() == 1:
#         tensor = tensor.unsqueeze(0)
    
#     # 创建比较矩阵
#     comparison_matrix = tensor.unsqueeze(2) > tensor.unsqueeze(1)
#     # 创建理想的比较矩阵（上三角矩阵）
#     # print(comparison_matrix)
#     ideal_matrix = torch.triu(torch.ones_like(comparison_matrix), diagonal=1)
#     # print(ideal_matrix.shape)
#     # 计算匹配的数量
#     matches = ( comparison_matrix * ideal_matrix == True ).float()
#     # 计算上三角矩阵的元素数量（即需要比较的对数）
#     num_comparisons = ideal_matrix.sum()
#     # print(matches.sum(),num_comparisons)
#     accuracy = matches.sum() / num_comparisons
#     return accuracy

def calculate_pairwise_accuracy(predicted_scores, true_rankings):
    # 确保输入是 2D 张量
    if predicted_scores.dim() == 1:
        predicted_scores = predicted_scores.unsqueeze(0)
    if true_rankings.dim() == 1:
        true_rankings = true_rankings.unsqueeze(0)
    
    # 创建预测的比较矩阵
    predicted_comparison = predicted_scores.unsqueeze(2) > predicted_scores.unsqueeze(1)
    # print(predicted_comparison)
    # 创建真实的比较矩阵
    true_comparison = true_rankings.unsqueeze(2) > true_rankings.unsqueeze(1)
    
    # 计算匹配的数量
    matches = (predicted_comparison == true_comparison).float()
    # print(true_comparison)
    # 创建上三角矩阵作为mask
    mask = torch.triu(torch.ones_like(matches), diagonal=1)
    
    # 应用mask并计算准确率
    accuracy = (matches * mask).sum() / mask.sum()
    
    return accuracy

def compute_ndcg(pred,labels):
    ddcg = dcg(pred,labels)
    idcg = dcg(labels,labels)

    return ddcg/idcg


if __name__ == "__main__":
    
    
    y1 = torch.tensor([[  3.1, 1.4, 1.38, 1.4, 1.4 ]])
    label = torch.tensor([ [ 0.8,0.6,0.5,0.4,0.35 ]])

    ddcg = dcg(y1,label)
    idcg = dcg(label,label)
    
    print(calculate_pairwise_accuracy(y1,label))
    print(ddcg,idcg,ddcg/idcg)
