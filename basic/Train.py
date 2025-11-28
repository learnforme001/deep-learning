from .Accumulator import Accumulator
from .Animator import Animator
import torch
from .FashionMnist import FashionMnist

class Train:
    @classmethod
    def sgd(cls, params, lr, batch_size):
        """小批量随机梯度下降"""
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    @classmethod
    def evaluate_loss(cls, net, data_iter, loss):
        acc = Accumulator(2)
        for X, y in data_iter:
            out = net(X)
            l = loss(out, y)
            acc.add(l.sum(), l.numel())
        return acc[0] / acc[1]

    @classmethod
    def accuracy(cls, y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    @classmethod
    def evaluate_accuracy(cls, net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()
        acc = Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                y_hat = net(X)
                acc.add(cls.accuracy(y_hat, y), y.numel())
        return acc[0] / acc[1]

    @classmethod
    def train_epoch_ch3(cls, net, train_iter, loss, updater):
        """训练模型一轮"""
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
            metric.add(float(l.sum().detach()), float(cls.accuracy(y_hat, y)), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]

    @classmethod
    def train_ch3(cls, net, train_iter, test_iter, loss, num_epochs, updater, save_path=None):
        """训练模型"""
        import os
        if save_path is None:
            os.makedirs('outputs', exist_ok=True)
            save_path = 'outputs/train_ch3.png'
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'],
                            save_path=save_path)
        for epoch in range(num_epochs):
            train_metrics = cls.train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = cls.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    @classmethod
    def predict_ch3(cls, net, test_iter, n=6, save_path=None):
        import os
        if save_path is None:
            os.makedirs('outputs', exist_ok=True)
            save_path = 'outputs/predictions.png'
        for X, y in test_iter:
            break
        trues = FashionMnist.get_fashion_mnist_labels(y)
        preds = FashionMnist.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
        FashionMnist.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n], save_path=save_path)