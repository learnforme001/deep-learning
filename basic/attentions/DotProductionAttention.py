from torch import nn
import torch
from basic.AttentionHelper import AttentionHelper
import math

class DotProductionAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductionAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size, num_queries, d)
        # keys: (batch_size, num_kv, d)
        # values: (batch_size, num_kv, value_size)
        # valid_lens: (batch_size,) or (batch_size, num_queries)
        d = queries.shape[-1]
        # scores: (batch_size, num_queries, num_kv)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = AttentionHelper.masked_softmax(scores, valid_lens)
        # attention_weights: (batch_size, num_queries, num_kv)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # output: (batch_size, num_queries, value_size)
