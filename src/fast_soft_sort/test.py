

import torch
from pytorch_ops import soft_rank, soft_sort
values = torch.tensor([[5., 1., 3.,4]], dtype=torch.float64)
labels = torch.tensor([[1.0, 0.7, 0.75,0.8] ], dtype=torch.float64)

res= soft_sort(-values, regularization_strength=1.0)
print(res)
res= soft_sort(-values, regularization_strength=0.1)
print(res)
res= soft_rank(-values, regularization_strength=2.0)
print(res)
res= soft_rank(-values, regularization_strength=2.0)
# label_res= soft_rank(-labels, regularization_strength=1.0)
sorted_indices = torch.argsort(-labels,)
ranks = torch.argsort(sorted_indices) + 1

# sorted_values, sorted_indices = torch.sort(x, dim=1, descending=True)
loss = torch.mean(  0.5*((res - ranks) ** 2)  )
print(res)
print('ranks',ranks)
print('loss',loss,)
# print(sorted_indices)