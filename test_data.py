import datasets
import json
from datasets import Dataset, DatasetDict
import os


dataset = datasets.load_dataset('data/hh-rank-dataset',split='train_K2')

print(dataset)