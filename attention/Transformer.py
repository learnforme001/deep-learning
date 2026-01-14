import math
import pandas as pd
import torch
from torch import nn
from basic import MultiHeadAttention, Encoder, PositionalEncoding, AttentionDecoder, TryGPU, NMT, EncoderDecoder, AttentionHelper
from RNN import train_seq2seq, predict_seq2seq, bleu

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 通常中间的 ffn_num_hiddens 比较大（是输入维度的 4 倍，比如输入 512，中间层就是 2048）。
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接&层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y) + X)
    
class EncoderBlock(nn.Module):
    """Transformer 编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hidden, num_heads, dropout, use_bias=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 1. 自注意力层 + 残差连接 + 归一化
        # 注意这里传入了三个 X，分别代表 Query, Key, Value。
        # 因为是“自”注意力，所以 Q=K=V=X。
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))

        # 2. 前馈网络层 + 残差连接 + 归一化
        return self.addnorm2(Y, self.ffn(Y))
    
class TransformerEncoder(Encoder):
    """Transformer 编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hidden, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hidden, num_heads, dropout, use_bias))
        
    def forward(self, X, valid_lens, *args):
        # 1. 嵌入与缩放
        # X 的形状: (batch_size, num_steps, num_hiddens)
        # 为什么要乘以 sqrt(num_hiddens)?
        # 解释：Embedding 的值通常比较小（初始化时方差为 1），而位置编码的值在 [-1, 1] 之间。
        # 为了防止位置编码的信息“淹没”了原来的语义信息，我们将 Embedding 放大。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 初始化一个列表来存储每一层的注意力权重（用于可视化或调试）
        self.attention_weights = [None] * len(self.blks)
        
        # 2. 逐层传递
        for i, blk in enumerate(self.blks):
            # 输入 X 进入第 i 个块，输出更新后的 X
            X = blk(X, valid_lens)
            # 记录第 i 个块的注意力权重
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    """解码器中的第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hidden, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，因此state[2][self.i]为None
        # 预测阶段，输出序列是逐个词元生成的，因此state[2][self.i]包含直到当前时间步的所有词元
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头（batch_size, num_steps）
            # 构造一个形状为 (batch_size, num_steps) 的矩阵
            # 第1行值是1，第2行值是2... 第n行值是n-
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 1. 自注意力 (Masked Self-Attention)
        # Query 是 X (当前输入)
        # Key/Value 是 key_values (包含历史信息)
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 2. 编码器-解码器注意力 (Cross Attention) -> 最关键的一步
        # Query 是 Y (来自解码器，代表"我想在这个上下文中找什么")
        # Key/Value 是 enc_outputs (来自编码器，代表"源句子的内容")
        # 这里使用的是 enc_valid_lens，因为我们要屏蔽掉源句子中的 Padding。
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 返回列表结构：
        # 0. 编码器的输出 (供 Cross Attention 使用)
        # 1. 编码器的有效长度 (供 Cross Attention 避免关注 Padding)
        # 2. 每一层的 KV Cache (初始化为 None，供 Masked Self-Attention 存储历史)
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
    
def Transformer_main():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, TryGPU.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = NMT.load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
            f'bleu {bleu(translation, fra, k=2):.3f}')
        
    
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
    -1, num_steps))
    print("Encoder attention weights shape:", enc_attention_weights.shape)
    AttentionHelper.show_heatmaps(
        enc_attention_weights.cpu(), xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))
    

    dec_attention_weights_2d = [head[0].tolist()
                                for step in dec_attention_weight_seq
                                for attn in step for blk in attn for head in blk]
    dec_attention_weights_filled = torch.tensor(
        pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
    dec_self_attention_weights, dec_inter_attention_weights = \
        dec_attention_weights.permute(1, 2, 3, 0, 4)
    dec_self_attention_weights.shape, dec_inter_attention_weights.shape
    # Plusonetoincludethebeginning-of-sequencetoken
    AttentionHelper.show_heatmaps(
        dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
        xlabel='Key positions', ylabel='Query positions',
        titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
    
    AttentionHelper.show_heatmaps(
        dec_inter_attention_weights, xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))

