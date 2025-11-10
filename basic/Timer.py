import time
import numpy as np

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.timers = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.timers.append(time.time() - self.tik)
        return self.timers[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.timers)/len(self.timers)
    
    def sum(self):
        """返回时间总和"""
        return sum(self.timers)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.timers).cumsum().tolist()