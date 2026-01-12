import torch
import matplotlib.pyplot as plt
from .Figure import Figure
from .EncoderDecoder import Decoder
from torch import nn

class AttentionHelper:
    @classmethod
    def show_heatmaps(cls, matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                    cmap='Reds'):
        """显示矩阵热图"""
        Figure.use_svg_display()
        # 1. 确定布局
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]

        # 2. 创建画布
        # sharex, sharey 意味着所有子图共享坐标轴，看起来更整洁
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
        
        # 3. 双重循环：遍历每一个子图
        for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
            for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                # === 核心绘图步骤 ===
                # imshow 是画热力图的标准函数
                # matrix.detach().numpy() 是因为 PyTorch 张量带梯度，画图前要剥离梯度并转成 numpy 数组
                pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)

                # 4. 美化坐标轴
                # 只有最后一行才显示 X轴标签（防止乱）
                if i == num_rows - 1:
                    ax.set_xlabel(xlabel)

                # 只有第一列才显示 Y轴标签
                if j == 0:
                    ax.set_ylabel(ylabel)
                if titles:
                    ax.set_title(titles[j])
        # 5. 添加右侧的颜色条 (Colorbar)，告诉你颜色深浅对应的数值
        fig.colorbar(pcm, ax=axes, shrink=0.6)


    @classmethod
    def sequence_mask(cls, X, valid_len, value = 0):
        """序列中屏蔽不相关的项，比如填充词元"""
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    @classmethod
    def masked_softmax(cls, X, valid_lens):
        """通过在最后一个轴上掩蔽元素来执行softmax操作"""
        # X:3D张量，valid_lens:1D或2D张量
        # X: (batch_size, query个数, key个数)
        # valid_lens: (batch_size,) 
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                # 由于后续我们需要对query进行遮掩（并且拍成(batch * query, keys)），因此其valid_lens需要改变形状
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
            X = AttentionHelper.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                                value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)
        
class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    
    @property
    def attention_weights(self):
        raise NotImplementedError