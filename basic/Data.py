from torch.utils import data
import torch

class Data:
    @classmethod
    def load_array(cls, data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
    @classmethod
    def synthetic_data(cls, w, b, num_examples):
        """生成 y = X w + b + 噪声"""
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))