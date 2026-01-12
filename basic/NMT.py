from .DownloadHelper import DownloadHelper
import os
import torch
from .Vocab import Vocab
from .Data import Data

class NMT:
    @classmethod
    def read_data_nmt(cls):
        """读取《时光机器》数据集"""
        DATA_HUB = DownloadHelper.DATA_HUB
        DATA_HUB['fra-eng'] = (DownloadHelper.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

        data_dir = DownloadHelper.download_extract('fra-eng')
        with open(os.path.join(data_dir, 'fra.txt'), 'r',
                encoding='utf-8') as f:
            return f.read()
        
    @classmethod
    def preprocess_nmt(cls, text):
        """预处理“英语－法语”数据集"""
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # 使用空格替换不间断空格
        # 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # 在单词和标点符号之间插入空格
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text)]
        return ''.join(out)

    @classmethod
    def tokenize_nmt(cls, text, num_examples=None):
        """词元化“英语－法语”数据数据集"""
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target
    

    @classmethod
    def truncate_pad(cls, line, num_steps, padding_token):
        """截断或填充文本序列"""
        if len(line) > num_steps:
            return line[:num_steps]  # 截断
        return line + [padding_token] * (num_steps - len(line))  # 填充
    
    @classmethod
    def build_array_nmt(cls, lines, vocab, num_steps):
        """将机器翻译的文本序列转换成小批量"""
        lines = [vocab[l] for l in lines]
        lines = [l + [vocab['<eos>']] for l in lines]
        array = torch.tensor([cls.truncate_pad(
            l, num_steps, vocab['<pad>']) for l in lines])
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, valid_len
    
    @classmethod
    def load_data_nmt(cls, batch_size, num_steps, num_examples=600):
        """返回翻译数据集的迭代器和词表"""
        text = cls.preprocess_nmt(cls.read_data_nmt())
        source, target = cls.tokenize_nmt(text, num_examples)
        src_vocab = Vocab(source, min_freq=2,
                            reserved_tokens=['<pad>', '<bos>', '<eos>'])
        tgt_vocab = Vocab(target, min_freq=2,
                            reserved_tokens=['<pad>', '<bos>', '<eos>'])
        src_array, src_valid_len = cls.build_array_nmt(source, src_vocab, num_steps)
        tgt_array, tgt_valid_len = cls.build_array_nmt(target, tgt_vocab, num_steps)
        data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
        data_iter = Data.load_array(data_arrays, batch_size)
        return data_iter, src_vocab, tgt_vocab