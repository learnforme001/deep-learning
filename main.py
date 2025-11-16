import os
import sys
import matplotlib

# Ensure an interactive backend before importing pyplot or code using it
# Preference: TkAgg (if tkinter exists) -> QtAgg (if Qt bindings exist) -> WebAgg (browser)
def _choose_backend():
    env_backend = os.environ.get("MPLBACKEND", "").strip()
    if env_backend:
        return  # Respect explicit choice

    # If already a GUI backend, keep it
    current = str(matplotlib.get_backend()).lower()
    non_gui = ("agg", "pdf", "ps", "svg", "cairo", "template", "module://matplotlib_inline")
    if not any(k in current for k in non_gui):
        return

    # Try TkAgg if tkinter is available
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        return
    except Exception:
        pass

    # Try QtAgg only if a Qt binding is importable
    for mod in ("PyQt6", "PySide6", "PyQt5", "PySide2"):
        try:
            __import__(mod)
            matplotlib.use("QtAgg", force=True)
            return
        except Exception:
            continue

    # Fallback to WebAgg (opens in a browser)
    try:
        matplotlib.use("WebAgg", force=True)
        print("[INFO] Using WebAgg backend; a browser window will display figures.")
    except Exception:
        # Last resort: leave as-is; user must install a GUI backend
        print("[WARN] No GUI backend available. Install Tk (tkinter) or Qt (PyQt/PySide).", file=sys.stderr)


_choose_backend()

from linear_models import soft_max_head, soft_max_torch
from multilayer_perceptron import mlp_head, mlp_torch, train_3d_poly, train_linear_poly, train_high_degree_poly, weight_decay_head, weight_decay_torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    weight_decay_torch(10)
    # Keep figures open when running as a script
    plt.show()
