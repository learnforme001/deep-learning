from torch import nn
import torch

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 1. 准备一个大矩阵 P，用来装位置编码
        # max_len 是假设的最大句子长度（比如1000），num_hiddens 是词向量维度（比如512）
        self.P = torch.zeros((1, max_len, num_hiddens))

        # 2. 生成分母
        # 这里的 10000^(2j/d)就是公式中的分母
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # 输入 X 是 [batch_size, 句子长度, 维度]
        # self.P 是预先算好的位置编码表
        # 直接做加法！
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)