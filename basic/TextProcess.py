from .SeqDataLoader import SeqDataLoader

class TextProcess:
    

    @classmethod
    def load_data_time_machine(cls, batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
        """返回时光机器数据集的迭代器和词表"""
        data_iter = SeqDataLoader(
            batch_size, num_steps, use_random_iter, max_tokens)
        return data_iter, data_iter.vocab
