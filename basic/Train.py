import math
from timeit import Timer
from .Accumulator import Accumulator
from .Animator import Animator
import torch
from .GPU import TryGPU
from torch import nn
from .FashionMnist import FashionMnist
from .Timer import Timer

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

    @classmethod
    def evaluate_accuracy_gpu(cls, net, data_iter, device=None): #@save
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(net, nn.Module):
            net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(Train.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    @classmethod
    def train_ch6(cls, net, train_iter, test_iter, num_epochs, lr, device):
        """用GPU训练模型(在第六章定义)"""
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'test acc'])
        timer, num_batches = Timer(), len(train_iter)
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], Train.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                (train_l, train_acc, None))
            test_acc = cls.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')
        # 显示最终图形
        animator.show()

    @classmethod
    def predict_ch8(self, prefix, num_preds, net, vocab, device):  #@save
        """在prefix后面生成新字符"""
        state = net.begin_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
        for y in prefix[1:]:  # 预热期
            # 目的是计算隐状态
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):  # 预测num_preds步
            y, state = net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
    @classmethod
    def grad_clipping(cls, net, theta):
        """裁剪梯度"""
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    @classmethod
    def train_epoch_ch8(cls, net, train_iter, loss, updater, device, use_random_iter):
        """训练网络一个迭代周期（定义见第8章）"""
        state, timer = None, Timer()
        metric = Accumulator(2)  # 训练损失之和,词元数量
        for X, Y in train_iter:
            # 处理隐状态
            if state is None or use_random_iter:
                # 在第一次迭代或使用随机抽样时初始化state
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                # detach从计算图分离隐状态，不再进行回传
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()
            # 前向传播
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)
            # 计算loss
            l = loss(y_hat, y.long()).mean()
            # 反向传播和梯度裁剪
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                cls.grad_clipping(net, 1)
                # 更新参数
                updater.step()
            else:
                l.backward()
                cls.grad_clipping(net, 1)
                # 因为已经调用了mean函数
                # 更新参数
                updater(batch_size=1)
            metric.add(l * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
    
    @classmethod
    def train_ch8(cls, net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
        """训练模型（定义见第8章）"""
        loss = nn.CrossEntropyLoss()
        animator = Animator(xlabel='epoch', ylabel='perplexity',
                                legend=['train'], xlim=[10, num_epochs])
        # 初始化
        if isinstance(net, nn.Module):
            updater = torch.optim.SGD(net.parameters(), lr)
        else:
            updater = lambda batch_size: cls.sgd(net.params, lr, batch_size)
        predict = lambda prefix: cls.predict_ch8(prefix, 50, net, vocab, device)
        # 训练和预测
        for epoch in range(num_epochs):
            ppl, speed = cls.train_epoch_ch8(
                net, train_iter, loss, updater, device, use_random_iter)
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))