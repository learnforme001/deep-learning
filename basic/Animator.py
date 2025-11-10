from .Figure import Figure
import matplotlib.pyplot as plt
from IPython import display as ipy_display


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
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        Figure.use_svg_display()
        self._in_notebook = _in_notebook()
        # Track if a window has been shown in script mode
        self._shown_once = False
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self._set_axes(
            self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

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
            # Ensure a window is opened once in script mode
            if not self._shown_once:
                try:
                    # Matplotlib >= 3.0
                    plt.show(block=False)
                except TypeError:
                    # Fallback for older versions
                    self.fig.show()
                self._shown_once = True
            # Update the GUI figure window
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
            # Process GUI events; keep UI responsive
            plt.pause(0.001)
