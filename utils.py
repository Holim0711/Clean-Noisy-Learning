import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def plot_confusion_matrix(cm, labs=None, ylabs=None, xlabs=None, cmap='Blues'):
    """
    Reference: scikit-learn/sklearn/metrics/_plot/confusion_matrix.py
    Args:
        - cm: confusion matrix (N x M)
        - lab: label texts (only when N == M)
        - ylab: true label texts
        - xlab: pred label texts
        - cmap: color theme (see 'colormaps' in matplotlib)
    """
    fig, ax = plt.subplots()

    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0., vmax=1.)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap_max if cm[i, j] < 0.5 else cmap_min
        text = f'{cm[i, j]:.2f}'
        text = '1.0' if text[0] == '1' else text[1:]
        ax.text(j, i, text, ha="center", va="center", color=color)

    if ylabs is None:
        ylabs = np.arange(cm.shape[0]) if labs is None else labs

    if xlabs is None:
        xlabs = np.arange(cm.shape[1]) if labs is None else labs

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=xlabs,
           yticklabels=ylabs,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((cm.shape[0] - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    return fig
