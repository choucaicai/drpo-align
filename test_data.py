import datasets
import json
from datasets import Dataset, DatasetDict,load_from_disk,load_dataset
import os


dataset = load_from_disk('data/hh-rank-dataset')
dataset = dataset['train_K2']
# dataset = load_dataset('data/hh-rank-dataset')
print(dataset)
# dataset.push_to_hub('kasoushu/hh-rank-dataset',token='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')