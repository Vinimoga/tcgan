"""
    This implementation is for UCR15 which includes 85 univariate time-series classification datasets.
"""
import numpy as np
import os

from .. import base
from ...lib.data import utils


def load_ucr_concat(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """

    :param data_name: 
    :param data_dir_root: 
    :param encode_label:
    :param one_hot: 
    :return: 
    """
    x_tr, y_tr, x_te, y_te, n_class = load_ucr(data_name, data_dir_root, encode_label, one_hot, z_norm)
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te])

    return x_all, y_all, n_class


def save_ucr(data_name, data_dir_root, X_train, y_train, X_test=None, y_test=None):
    length = X_train.shape[1]
    str_fmt = '%d,' + '%.4f,' * (length)
    str_fmt = str_fmt[:(len(str_fmt) - 1)]

    dir_out = os.path.join(data_dir_root, data_name)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)

    trainset = np.concatenate([y_train[:, np.newaxis], X_train], axis=1)
    np.savetxt(
        os.path.join(dir_out, '{}_TRAIN'.format(data_name)),
        trainset, fmt=str_fmt, delimiter=',')
    if X_test is not None and y_test is not None:
        testset = np.concatenate([y_test[:, np.newaxis], X_test], axis=1)
        np.savetxt(os.path.join(dir_out, '{}_TEST'.format(data_name)),
                   testset, fmt=str_fmt, delimiter=',')


def load_ucr_dataset(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    x_tr, y_tr, x_te, y_te, n_class = load_ucr(data_name, data_dir_root, encode_label, one_hot, z_norm)
    train_set = base.Dataset(X=x_tr, y=y_tr)
    valid_set = None
    test_set = base.Dataset(X=x_te, y=y_te)
    return base.Datasets(train=train_set, valid=valid_set, test=test_set, nclass=n_class)


def load_ucr(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """
    
    :param data_name: 
    :param data_dir_root: 
    :param one_hot: 
    :return: 
    """
    # load from file
    filename_train = '{}/{}/{}_TRAIN'.format(data_dir_root, data_name, data_name)
    filename_test = '{}/{}/{}_TEST'.format(data_dir_root, data_name, data_name)
    data_train = np.genfromtxt(filename_train, delimiter=',', dtype=np.float32)
    data_test = np.genfromtxt(filename_test, delimiter=',', dtype=np.float32)

    # parse
    x_train = data_train[:, 1::]
    y_train = data_train[:, 0].astype(int)
    x_test = data_test[:, 1::]
    y_test = data_test[:, 0].astype(int)
    y_all = np.concatenate([y_train, y_test])
    classes, y_all = np.unique(y_all, return_inverse=True)
    n_class = len(classes)

    # z_norm
    if z_norm:
        x_train = utils.z_normalize(x_train)
        x_test = utils.z_normalize(x_test)

    # reconstruct label
    if encode_label or one_hot:
        # encode labels with value between 0 and n_classes-1.
        y_train = y_all[:x_train.shape[0]]
        y_test = y_all[x_train.shape[0]:]
        # convert to one hot label in need
        if one_hot is True:
            y_train = utils.dense_to_one_hot(y_train, n_class)
            y_test = utils.dense_to_one_hot(y_test, n_class)

    return x_train, y_train, x_test, y_test, n_class


