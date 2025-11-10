from matplotlib import pyplot as plt


class Figure:
    @staticmethod
    def use_svg_display():
        _backend = plt.get_backend()
        if _backend in ["MacOSX", "Qt4Agg", "Qt5Agg"]:
            print("[INFO] Changing Matplotlib backend from '{}' to 'SVG'".format(_backend))
            plt.switch_backend("SVG")

    @staticmethod
    def set_figsize(figsize=(3.5, 2.5)):
        Figure.use_svg_display()
        plt.rcParams["figure.figsize"] = figsize
