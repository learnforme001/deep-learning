from torch import nn
import torch

class MLP(nn.Module):
    # 用模型参数声明层，这里声明两个全连接层
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义前向传播
    def forward(self, X):
        # 以x为输入，计算带有激活函数的隐藏层，然后输出未规范化的输出值
        return self.out(nn.functional.relu(self.hidden(X)))

net = MLP()
X = torch.rand(2, 20)
print(net(X))