import os
import sys
import shutil
import matplotlib

# 对于远程环境（如VSCode Remote Tunnel），使用非交互式后端
matplotlib.use('Agg')

from linear_models import soft_max_head, soft_max_torch
from multilayer_perceptron import mlp_head, mlp_torch, train_3d_poly, train_linear_poly, train_high_degree_poly, weight_decay_head, weight_decay_torch, dropout_head, dropout_torch, kaggle_house_price
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 清空outputs目录
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
        print("已清空outputs目录")
    os.makedirs('outputs', exist_ok=True)
    
    weight_decay_head(0.3)
    # plt.show()
