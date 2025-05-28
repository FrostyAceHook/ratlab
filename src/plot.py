
# Avoid super expensive import on start-up, only do it when used.
plt = None
def init_plotting():
    global plt
    if plt is None:
        import matplotlib.pyplot as plt
        plt.ion() # get it


def figure(xlabel=None, ylabel=None, title=None, grid=False):
    init_plotting()
    plt.figure(figsize=(8, 5))
    plt.grid(grid)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)
    if title is not None:
        plt.title(title)

def plot(x, y, label=None, color=None, linestyle=None, linewidth=None,
            marker=None, markersize=None, markerfacecolor=None,
            markeredgecolor=None, alpha=None):
    init_plotting()
    plt.plot(x, y, label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, marker=marker, markersize=markersize,
            markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor,
            alpha=alpha)

    # Enable legend if anything has been labelled.
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(lbl for lbl in labels):
        plt.legend()
