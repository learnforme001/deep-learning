import collections
import re
from basic.DownloadHelper import DownloadHelper
from basic.TextProcess import TextProcess, Vocab



def text_processing_main():
    lines = TextProcess.read_time_machine()
    # print(f'文本总行数: {len(lines)}')
    # print(lines[0])
    # print(lines[10])

    # 分词
    tokens = TextProcess.tokenize(lines)
    # for i in range(11):
    #     print(tokens[i])
    
    # 文本词表
    vocab = Vocab(tokens)
    # print(list(vocab.token_to_idx.items())[:10])

    # for i in [0, 10]:
    #     print('文本:', tokens[i])
    #     print('索引:', vocab[tokens[i]])

    corpus, vocab = TextProcess.load_corpus_time_machine()
    print(f'词元索引列表长度: {len(corpus)}'
          f'\n词表大小: {len(vocab)}')

