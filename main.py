import os
import sys
import shutil
import matplotlib

# 对于远程环境（如VSCode Remote Tunnel），使用非交互式后端
# matplotlib.use('Agg')  # 注释掉以在本地显示图形窗口

from linear_models import soft_max_head, soft_max_torch
from multilayer_perceptron import mlp_head, mlp_torch, train_3d_poly, train_linear_poly, train_high_degree_poly, weight_decay_head, weight_decay_torch, dropout_head, dropout_torch, kaggle_house_price
from CNN import LeNet_main, AlexNet_main, vgg_main, NiN_main, GoogLeNet_main, BatchNorm_LeNet_main, Simple_BatchNorm_LeNet_main, ResNet_main, DenseNet_main
from RNN import sequence_model_main, RNN_head, RNN_torch
from attention import main as attention_main
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 清空outputs目录
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
        print("已清空outputs目录")
    os.makedirs('outputs', exist_ok=True)

    attention_main()

    
    # weight_decay_head(0.3)
    # 保持窗口打开
    plt.ioff()  # 关闭交互式模式
    plt.show()  # 阻塞直到窗口关闭
