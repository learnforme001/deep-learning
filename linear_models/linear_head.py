import torch
from base import data_iter
from basic import Data



def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = Data.synthetic_data(true_w, true_b, 1000)
    # for X, y in data_iter(10, features, labels):
    #     print(X, '\n', y)
    #     break

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 超参数
    lr = 0.03
    num_epochs = 3
    loss = squared_loss
    net = linreg

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size=10, features=features, labels=labels):
            l = loss(net(X, w, b), y) # 小批量损失
            l.sum().backward()
            sgd([w, b], lr, batch_size=10)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')