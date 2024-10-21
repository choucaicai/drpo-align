import datasets
import json
from datasets import Dataset, DatasetDict
import os

# dataset_name = 
# return os.path.join(HH_RANK_BASE,dataset_name)

# split =  'train'
# K=2
# base_path =  f'data/samples/hh-rank-{split}-K{K}-deberta-v3.json'

# data_list = []
# with open(base_path,'r') as f:
#     json_data = json.load(f)
#     samples_data = json_data['samples']

def load_data(split, K):
    base_path = f'data/samples/hh-rank-{split}-K{K}-deberta-v3.json'
    print(f'loading {base_path} ....')
    with open(base_path, 'r') as f:
        json_data = json.load(f)
        samples_data = json_data['samples']
    return samples_data

splits = ['train', 'test']
K_values = [2, 4, 6, 8]
# 创建一个空的DatasetDict
dataset_dict = DatasetDict()
shard_size = 1024*40
# print(samples_data[0])
# print(json_data['config'],json_data['json_data'])


for split in splits:
    for K in K_values:
        # 加载数据
        data = load_data(split, K)
        
        # 创建Dataset对象
        dataset = Dataset.from_list(data)
        # 将Dataset添加到DatasetDict中，使用适当的键名
        key = f"{split}_K{K}"
        dataset._split = key
        dataset_dict[key] = dataset

        # parquet_path = os.path.join('data/hh-rank-dataset', f"{key}")
        # num_shards = (len(dataset) + shard_size - 1) // shard_size
        # os.makedirs(parquet_path, exist_ok=True)
        # dataset.to_parquet(
        #     os.path.join(parquet_path, f"{key}.parquet"),
        # )
# 打印DatasetDict的结构
# print(dataset_dict)
# # 示例：访问特定的数据集
# print(dataset_dict['train_K2'][0])
# print(dataset_dict['test_K4'][0])


dataset_dict.push_to_hub("kasoushu/hh-rank-dataset")
# 遍历所有split和K值的组合
#         将Dataset添加到DatasetDict中，使用适当的键名
#         key = f"{split}_K{K}"
#         dataset_dict[key] = dataset
        
#         # 保存为Parquet文件
#         parquet_path = os.path.join(output_dir, f"{key}.parquet")
#         dataset.to_parquet(parquet_path)
#         print(f"Saved {key} to {parquet_path}")