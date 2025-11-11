import torch
from torch import nn
from basic import Train, FashionMnist

def soft_max_torch():
    batch_size = 256
    trans_iter, test_iter = FashionMnist.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction="none")
    lr = 0.1
    updater = torch.optim.SGD(net.parameters(), lr)

    num_epochs = 10
    Train.train_ch3(net, trans_iter, test_iter, loss, num_epochs, updater)
    Train.predict_ch3(net, test_iter)


