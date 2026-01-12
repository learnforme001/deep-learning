from torch import nn
import torch
from basic.AttentionHelper import AttentionHelper

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size, num_queries, query_size)
        # keys: (batch_size, num_kv, key_size)
        # values: (batch_size, num_kv, value_size)
        # valid_lens: (batch_size,) or (batch_size, num_queries)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries: (batch_size, num_queries, num_hiddens)
        # keys: (batch_size, num_kv, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # features: (batch_size, num_queries, num_kv, num_hiddens)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)  # 经过线性层后，最后一维变为1，去掉该维度
        # scores: (batch_size, num_queries, num_kv)
        self.attention_weights = AttentionHelper.masked_softmax(scores, valid_lens)
        # attention_weights: (batch_size, num_queries, num_kv)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # output: (batch_size, num_queries, value_size)
