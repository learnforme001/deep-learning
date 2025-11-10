from .base import load_data_fashion_mnist, sgd, show_images, get_fashion_mnist_labels
import torch
from IPython import display
from basic import Accumulator, Animator
    
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    # 这里y是正确标签的index，而对于真实标签来说，概率为1，因此不需要再乘以y
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    acc = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            acc.add(accuracy(y_hat, y), y.numel())
    return acc[0] / acc[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        # Detach to avoid autograd warning when converting to Python float
        metric.add(float(l.sum().detach()), float(accuracy(y_hat, y)), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def soft_max_head():
    batch_size = 256
    trans_iter, test_iter = load_data_fashion_mnist(batch_size)
    
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
    train_ch3(net, trans_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)




