
import matrix as _matrix
from bg import bg as _bg


def figure(title=None, xlabel=None, ylabel=None, grid=False, figsize=(8, 5)):
    plt = _bg.plt
    plt.figure(figsize=figsize)
    plt.grid(not not grid)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)
    if title is not None:
        plt.title(title)

def plot(x, y, label=None, color=None, linestyle=None, linewidth=None,
            marker=None, markersize=None, markerfacecolor=None,
            markeredgecolor=None, alpha=None):
    plt = _bg.plt
    if isinstance(x, _matrix.Matrix):
        x = x.numpyvec(float)
    if isinstance(y, _matrix.Matrix):
        y = y.numpyvec(float)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, marker=marker, markersize=markersize,
            markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor,
            alpha=alpha)

    # Enable legend if anything has been labelled.
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(lbl for lbl in labels):
        plt.legend()
