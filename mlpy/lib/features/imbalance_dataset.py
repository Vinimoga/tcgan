"""
There are some statistics always used to describe a dataset.

References:
- \cite{huang2019deep}, Deep prototypical networks for imbalanced time series classification under data scarcity.
- \cite{ma2021joint}, Joint-Label Learning by Dual Augmentation for Time Series Classification.

"""
import numpy as np

from mlpy.lib.data.utils import distribute_y


def imbalance_scores(y, n_classes, **kwargs):
    """ pack all the following scores
    """
    res = {
        'imbalance_degree': imbalance_degree(y, n_classes),
        'scarcity_degree': scarcity_degree(y, n_classes),
        'imbalance_ratios': imbalance_ratios(y)
    }
    return res


def imbalance_degree(y, n_classes):
    # \cite{huang2019deep}, called beta, 'We measure the data balance in the training set by Shannon Entropy'.
    n = len(y)
    y_distr = distribute_y(y)
    beta = 0
    for _, ni in y_distr.items():
        r = ni / n
        beta += r * np.log(r)
    beta = -1 * beta / np.log(n_classes)
    return beta


def scarcity_degree(y, n_classes):
    # \cite{huang2019deep}: called alpha, averaging instances per class
    return len(y) / n_classes


def imbalance_ratios(y):
    # \cite{ma2021joint}, 'the ratio between the numbers of samples of most and least frequent classes.'
    y_distr = distribute_y(y)
    n_max = 0
    n_min = np.inf
    for _, ni in y_distr.items():
        n_max = max(n_max, ni)
        n_min = min(n_min, ni)
    im_r = n_max / n_min
    return im_r

