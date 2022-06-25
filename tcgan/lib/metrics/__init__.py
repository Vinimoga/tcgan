from .base import gan_loss
from .base import gan_accuracy
from .nnd import nnd_score, nnc_score
from .mmd import mmd_score
from .vis import tsne, pca


def similarity_by_classes(func, x1, x2, y, with_details=False):
    from mlpy.lib.data.utils import distribute_dataset

    x1_distr = distribute_dataset(x1, y)
    x2_distr = distribute_dataset(x2, y)
    score_distr = {}
    score = 0
    count = 0
    for c in x1_distr.keys():
        x1_c = x1_distr[c]
        x2_c = x2_distr[c]
        # most of the time, input of similarity function (func) is 2D array.
        _score = func(x1_c, x2_c)
        score_distr[c] = _score
        score += _score
        count += 1
    score = score / count  # average on all classes
    if with_details:
        return score, score_distr
    else:
        return score
