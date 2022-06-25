import os

from mlpy.configure import DIR_DATASET, UCR15, UCR18, UEA_MTS
from mlpy.lib.data import utils
from mlpy.datasets import ucr_uea


def main_ucr15():
    print("=========== Test to load UCR15 ===========")
    data_root = os.path.join(DIR_DATASET, UCR15)
    data_name = '50words'
    datasets = ucr_uea.load_ucr(data_name, data_root, one_hot=False)
    x_tr = datasets.train.X
    y_tr = datasets.train.y
    x_te = datasets.test.X
    y_te = datasets.test.y

    distr = utils.distribute_dataset(x_tr, y_tr)
    print(distr.keys())


def main_uea18_multi_ts():
    print("=========== Test to load UAE multivariate time series ===========")
    data_root = os.path.join(DIR_DATASET, UEA_MTS)
    data_name = 'ArticularyWordRecognition'
    x_tr, y_tr, x_te, y_te, nclass = ucr_uea.load_ucr_uae(data_name, data_root)
    print(x_tr.shape, y_tr.shape)
    print(x_te.shape, y_te.shape)
    print(nclass)
    print()


def main_ucr18_units_ts():
    print("=========== Test to load UCR18 ===========")
    data_root = os.path.join(DIR_DATASET, UCR18)
    data_name = 'ArrowHead'
    x_tr, y_tr, x_te, y_te, nclass = ucr_uea.load_ucr_uae(data_name, data_root)
    print(x_tr.shape, y_tr.shape)
    print(x_te.shape, y_te.shape)
    print(nclass)
    print()


if __name__ == '__main__':
    main_ucr15()
    main_uea18_multi_ts()
    main_ucr18_units_ts()
