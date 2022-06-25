import os

from mlpy.lib.utils.path import tag_path, makedirs
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets import ucr_uea

from configure import DIR_LOG
from vis import tsne


def run(data_name, dir_data, log_dir):
    log_path = makedirs(os.path.join(log_dir, data_name))
    x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(data_name, dir_data)
    tsne(x_tr, y_tr, x_te, y_te, log_path)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 4)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))

    data_name = 'Two_Patterns'

    run(data_name, DIR_DATA_UCR15, log_dir)