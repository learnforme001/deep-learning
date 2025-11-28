import torch
import random
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
import torchvision


class FashionMnist:
    @classmethod
    def get_fashion_mnist_labels(cls, labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    @classmethod
    def _in_notebook(cls):
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
        except Exception:
            return False

    @classmethod
    def show_images(cls, imgs, num_rows, num_cols, titles=None, scale=1.5, save_path=None):
        """显示图像，imgs是1~4维的Tensor或者是一个图片（ndarray）组成的list或数组"""
        import os
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                ax.imshow(img.numpy())
            else:
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        plt.tight_layout()
        if cls._in_notebook():
            # Let notebook display handle rendering
            pass
        else:
            # 在非notebook环境下保存图片
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f'图片已保存到: {save_path}')
            else:
                plt.pause(0.001)
        return axes

    @classmethod
    def get_dataloader_workers(cls):
        return 4

    @classmethod
    def load_data_fashion_mnist(cls, batch_size, resize=None):
        """下载Fashion-MNIST数据集，然后加载到内存中"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=cls.get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=cls.get_dataloader_workers()))