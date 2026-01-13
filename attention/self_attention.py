from basic import MultiHeadAttention
import torch

def self_attention_main():
    num_hiddens, num_heads = 100, 5
    # 这里并不必要QKV和num_hiddens的大小相同，但是为了之后的残差连接，那么最好是四者相同
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    # print(attention.eval())
    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    print(attention(X, X, X, valid_lens).shape)
        