import torch
import matplotlib.pyplot as plt
from .Figure import Figure

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