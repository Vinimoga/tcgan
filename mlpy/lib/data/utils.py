import numpy as np
from sklearn.utils.validation import column_or_1d
import json
import random

from sklearn import preprocessing
from sklearn import model_selection

train_test_split = model_selection.train_test_split


def z_normalize(data, axis=1):
    """
    
    :param data: 
    :param axis:  int (1 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.
        Note: time series should use axis=1.
    :return: 
    """
    return preprocessing.scale(data, axis=axis)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors.

    :param labels_dense: 1-array like. Each element correspond a class label and it's type must be int
        and value should be in [0, num_classes-1].
    :param num_classes: 
    :return: 
    """

    # check input
    labels_dense = column_or_1d(labels_dense, warn=True)
    if min(labels_dense) < 0 or max(labels_dense) >= num_classes:
        raise ValueError(
            "The value of label range in [{}, {}], which out of the valid range [0, {}-1]".format(
                min(labels_dense), max(labels_dense), num_classes)
        )
    # dense to one hot
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def one_hot_to_dense(y):
    return np.argmax(y, axis=1)


def distribute_dataset(X, y):
    y = column_or_1d(y, warn=True)
    res = {}
    for id in np.unique(y):
        res[id] = X[y==id]
    return res


def distribute_y(y):
    y = column_or_1d(y)
    res = {}
    for id in np.unique(y):
        res[id] = np.sum(y==id)
    return res


def distribute_y_json(y):
    y = column_or_1d(y)
    res = {}
    for id in np.unique(y):
        res[str(id)] = str(np.sum(y==id))
    return json.dumps(res)


def dic_to_json(in_dict):
    """
    reference: 
        https://stackoverflow.com/questions/12734517/json-dumping-a-dict-throws-typeerror-keys-must-be-a-string
        https://stackoverflow.com/questions/1436703/difference-between-str-and-repr
    :param in_dict: 
    :return: 
    """
    local_dict = {}
    for key in in_dict.keys():
        if type(key) is not str:
            try:
                local_dict[str(key)] = in_dict[key]
            except:
                try:
                    local_dict[str(key)] = in_dict[key]
                except:
                    raise TypeError("current key:{} can't convert to str type!".format(key))
        else:
            local_dict[key] = in_dict[key]
    return json.dumps(local_dict)


def shuffle_dataset(arrays, seed=None):
    n_arrays = len(arrays)
    assert n_arrays > 0, "the number of arrays must be large than 0."

    n_samples = arrays[0].shape[0]
    idxs = np.arange(n_samples)
    rand = random.Random(seed)
    rand.shuffle(idxs)
    res = []
    for i in range(n_arrays):
        res.append(arrays[i][idxs])
    return res


def split_random(x, ratio, y=None, seed=None):
    n = x.shape[0]
    idxs = np.arange(n)
    rand = random.Random(seed)
    rand.shuffle(idxs)

    n_left = int(ratio * n)
    ind_left = idxs[0:n_left]
    ind_right = idxs[n_left::]

    if y is not None:
        return x[ind_left], y[ind_left], x[ind_right], y[ind_right]
    else:
        return x[ind_left], x[ind_right]


def split_stratified(x, y, ratio, seed=None, with_y=True):
    x, y = shuffle_dataset([x, y], seed=seed)
    distr = distribute_dataset(x, y)

    x_left, y_left = [], []
    x_right, y_right = [], []

    for key, x_batch in distr.items():
        n = x_batch.shape[0]
        i = int(np.floor(n * ratio))

        x_left.append(x_batch[:i])
        y_left.extend([key]*i)
        x_right.append(x_batch[i:])
        y_right.extend([key]*(n-i))

    x_left, y_left = np.vstack(x_left), np.array(y_left)
    x_right, y_right = np.vstack(x_right), np.array(y_right)

    assert (x_left.shape[0] + x_right.shape[0]) == x.shape[0]

    if with_y is False:
        return x_left, x_right
    else:
        return x_left, y_left, x_right, y_right


