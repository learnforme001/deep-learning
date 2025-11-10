from matplotlib import pyplot as plt


class Figure:
    @staticmethod
    def use_svg_display():
        """Prefer SVG rendering in notebooks without changing GUI backend.

        In Jupyter, set the inline format to SVG for clarity. In regular
        Python scripts, do nothing so interactive backends (TkAgg/QtAgg)
        remain available for on-screen windows.
        """
        try:
            # Only available in notebook/inline contexts
            from matplotlib_inline.backend_inline import set_matplotlib_formats
            set_matplotlib_formats("svg")
        except Exception:
            # Not in a notebook (or module not available) — keep current backend
            pass

    @staticmethod
    def set_figsize(figsize=(3.5, 2.5)):
        Figure.use_svg_display()
        plt.rcParams["figure.figsize"] = figsize
