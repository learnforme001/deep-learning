from .Figure import Figure
import matplotlib.pyplot as plt
from IPython import display as ipy_display
import os


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        # 检测Jupyter Notebook或Google Colab
        shell_name = ip.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'Shell')
    except Exception:
        return False

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=('-', '--', '-.', ':'),
                 nrows=1, ncols=1, figsize=(3.5, 2.5), save_path=None):
        if legend is None:
            legend = []
        self._in_notebook = _in_notebook()
        self._display_handle = None  # 用于更新display
        
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
        
        # 打印当前数据点（用于调试和查看训练进度）
        if self._in_notebook:
            # 只打印最新的数据点
            latest_values = []
            for i, y_vals in enumerate(self.Y):
                if y_vals:
                    latest_values.append(f"{y_vals[-1]:.4f}")
                else:
                    latest_values.append("N/A")
            
            # 获取当前epoch值
            if self.X and self.X[0]:
                current_epoch = self.X[0][-1]
            else:
                current_epoch = 0
            
            # 获取legend名称
            legend = self.axes[0].get_legend()
            if legend:
                labels = [t.get_text() for t in legend.get_texts()]
                print(f"Epoch {current_epoch:.2f}: " + 
                      ", ".join([f"{label}={val}" for label, val in zip(labels, latest_values)]))
            else:
                print(f"Epoch {current_epoch:.2f}: " + 
                      ", ".join(latest_values))
        
        if self._in_notebook:
            # 在notebook环境下使用清除+重新显示的方式（兼容性最好）
            ipy_display.clear_output(wait=True)
            ipy_display.display(self.fig)
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
    
    def show(self):
        """显示最终图形（主要用于notebook环境保持最后的图形可见）"""
        if self._in_notebook:
            # 在notebook中最后一次显示，保持可见（不使用clear_output）
            ipy_display.display(self.fig)
        else:
            # 在脚本环境中保持窗口打开
            plt.show()
