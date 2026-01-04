import random
import torch
import re
from .DownloadHelper import DownloadHelper
from .Vocab import Vocab

class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = SeqDataLoader.seq_data_iter_random
        else:
            self.data_iter_fn = SeqDataLoader.seq_data_iter_sequential
        self.corpus, self.vocab = SeqDataLoader.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
    @classmethod
    def tokenize(cls, lines, token='word'):  #@save
        """将文本行拆分为单词或字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('错误：未知词元类型：' + token)

    @classmethod
    def read_time_machine(cls):
        """读取《时光机器》数据集"""
        DATA_HUB = DownloadHelper.DATA_HUB
        DATA_HUB['time_machine'] = (DownloadHelper.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

        with open(DownloadHelper.download('time_machine'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    @classmethod
    def load_corpus_time_machine(cls, max_tokens=-1):  #@save
        """返回时光机器数据集的词元索引列表和词表"""
        lines = cls.read_time_machine()
        tokens = cls.tokenize(lines, 'char')
        vocab = Vocab(tokens)
        # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
        # 所以将所有文本行展平到一个列表中
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab
    
    @classmethod
    def seq_data_iter_random(cls, corpus, batch_size, num_steps):  #@save
        """使用随机抽样生成一个小批量序列数据"""
        # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
        corpus = corpus[random.randint(0, num_steps - 1):]
        # 减去1是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中，子序列的起始索引是随机的
        random.shuffle(initial_indices)
        # print(initial_indices)
        # print(corpus)

        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos:pos + num_steps]

        num_batches = len(initial_indices) // batch_size
        for i in range(0, num_batches * batch_size, batch_size):
            # print(i)
            batch_indices = initial_indices[i:i + batch_size]
            X = [data(j) for j in batch_indices]
            Y = [data(j + 1) for j in batch_indices]
            yield torch.tensor(X), torch.tensor(Y)

    @classmethod
    def seq_data_iter_sequential(cls, corpus, batch_size, num_steps):  #@save
        """使用顺序分区生成一个小批量子序列"""
        # 从随机偏移量开始划分序列
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y