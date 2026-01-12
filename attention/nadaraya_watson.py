
import torch
from matplotlib import pyplot as plt
from basic import Animator

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数

def plot_kernel_reg(y_hat):
    plt.figure(figsize=(6, 3))
    plt.plot(x_test, y_truth, label='Truth')
    plt.plot(x_test, y_hat, label='Pred')
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)
