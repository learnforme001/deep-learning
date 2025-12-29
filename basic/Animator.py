from .Figure import Figure
import matplotlib.pyplot as plt
from IPython import display as ipy_display
import os


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=('-', '--', '-.', ':'),
                 nrows=1, ncols=1, figsize=(3.5, 2.5), save_path=None):
        if legend is None:
            legend = []
        self._in_notebook = _in_notebook()
        
        # 只在notebook环境下使用SVG显示
        if self._in_notebook:
            Figure.use_svg_display()
        else:
            # 在非notebook环境下启用交互式模式
            plt.ion()  # 开启交互式模式
        
        # Track if a window has been shown in script mode
        self._shown_once = False
        self.save_path = save_path
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self._set_axes(
            self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
        # 显示初始窗口
        if not self._in_notebook:
            plt.show(block=False)
            plt.pause(0.001)

    def _set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if legend:
                ax.legend(legend)
            ax.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (x_, y_) in enumerate(zip(x, y)):
            if x_ is not None and y_ is not None:
                self.X[i].append(x_)
                self.Y[i].append(y_)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        if self._in_notebook:
            ipy_display.display(self.fig)
            ipy_display.clear_output(wait=True)
        else:
            # 在非notebook环境下实时更新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # 短暂暂停以允许图形更新
            
            # 保存图片（如果指定了路径）
            if self.save_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
                self.fig.savefig(self.save_path, dpi=200, bbox_inches='tight')
                if not self._shown_once:
                    print(f"图片已保存到: {self.save_path}")
                    self._shown_once = True
    
    @classmethod
    def plot(cls, x, y, xlabel=None, ylabel=None, xlim=None, ylim=None, 
             xscale='linear', yscale='linear', figsize=(6, 3), legend=None, grid=True):
        """
        便捷绘图方法，类似 d2l.plot 的使用方式
        
        参数:
            x: x轴数据，可以是单个数组或数组列表（与y对应）
            y: y轴数据，可以是单个数组或数组列表
            xlabel: x轴标签
            ylabel: y轴标签
            xlim: x轴范围
            ylim: y轴范围
            xscale: x轴刻度类型
            yscale: y轴刻度类型
            figsize: 图形大小
            legend: 图例
            grid: 是否显示网格，默认True
        
        示例:
            Animator.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
            Animator.plot([time1, time2], [y1, y2], 'time', 'y', legend=['a', 'b'])
        """
        plt.figure(figsize=figsize)
        
        # 处理y数据
        if not isinstance(y, list):
            y = [y]
        
        # 处理x数据：如果x是列表，则为每个y提供对应的x；否则所有y共用一个x
        if isinstance(x, list):
            x_list = x
        else:
            x_list = [x] * len(y)
        
        # 绘制所有曲线
        for x_data, y_data in zip(x_list, y):
            plt.plot(x_data, y_data)
        
        # 设置坐标轴
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.xscale(xscale)
        plt.yscale(yscale)
        
        if legend:
            plt.legend(legend)
        
        if grid:
            plt.grid(True)
        plt.show()
