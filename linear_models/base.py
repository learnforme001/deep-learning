import torch
import random
from basic import FashionMnist

def data_iter(batch_size, features, labels):
    """构造一个迭代器来遍历数据集"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]) # 花式索引
        yield features[batch_indices], labels[batch_indices]
