from .base import sgd
import torch
from basic import Train, FashionMnist
    
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    # 这里y是正确标签的index，而对于真实标签来说，概率为1，因此不需要再乘以y
    return -torch.log(y_hat[range(len(y_hat)), y])

def soft_max_head():
    batch_size = 256
    trans_iter, test_iter = FashionMnist.load_data_fashion_mnist(batch_size)
    
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    lr = 0.1
    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    num_epochs = 10
    Train.train_ch3(net, trans_iter, test_iter, cross_entropy, num_epochs, updater)
    Train.predict_ch3(net, test_iter)




