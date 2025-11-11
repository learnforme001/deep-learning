import torch

class ActivateFunction:
    @classmethod
    def sigmoid(cls, x):
        pass

    @classmethod
    def relu(cls, x):
        a = torch.zeros_like(x)
        return torch.max(x, a)

