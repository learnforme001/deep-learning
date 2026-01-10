import torch
from torch import nn
from basic import TextProcess, TryGPU, Train
from .RNN_simple import RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = TextProcess.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = TryGPU.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

def Deep_RNN_torch(use_random_iter):
    num_epochs, lr = 500, 2
    Train.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device, use_random_iter=use_random_iter)