from torch import nn
import torch
from basic import DotProductionAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias = False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = DotProductionAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens 形状: (batch_size，)或(batch_size，查询的个数)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
        # output_concat的形状(batch_size，查询的个数，num_hiddens)
    



def transpose_qkv(X, num_heads):
    """为了多头注意力计算，将查询、键和值的形状变换为(batch_size*num_heads, num_queries or kv pairs, num_hiddens/num_heads)"""
    # 输入X的形状(batch_size，查询或者“键－值”对的个数，num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X的形状(batch_size，查询或者“键－值”对的个数，num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # X的形状(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
    # 最终输出的形状(batch_size * num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)

def transpose_output(X, num_heads):
    """逆转多头注意力的维度变换"""
    # 输入X的形状(batch_size * num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # X的形状(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # X的形状(batch_size，查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
    return X.reshape(X.shape[0], X.shape[1], -1)
    # 最终输出的形状(batch_size，查询或者“键－值”对的个数, num_hiddens)