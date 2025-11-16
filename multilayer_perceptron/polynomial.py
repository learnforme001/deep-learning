import torch
import numpy as np
import math
from torch import nn
from basic import Accumulator, Data, Animator, Train

max_degree = 20
n_train, n_test = 100, 100

def get_poly_data():
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # 多项式计算
    # 标准化，防止特征过大
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)
    labels = np.dot(poly_features, true_w) # 多项式乘系数求和
    labels += np.random.normal(scale=0.1, size=labels.shape) # 添加噪声

    # 转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                               for x in [true_w, features,
                                                         poly_features, labels]]
    return true_w, features, poly_features, labels

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = Data.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = Data.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)

    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        Train.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (Train.evaluate_loss(net, train_iter, loss),
                                    Train.evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

def train_3d_poly():
    true_w, features, poly_features, labels = get_poly_data()
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:])

def train_linear_poly():
    true_w, features, poly_features, labels = get_poly_data()
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
          labels[:n_train], labels[n_train:])
    
def train_high_degree_poly():
    true_w, features, poly_features, labels = get_poly_data()
    train(poly_features[:n_train, :], poly_features[n_train:, :],
          labels[:n_train], labels[n_train:], num_epochs=1500)