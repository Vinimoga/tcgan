"""
python -m pip install tslearn

"""
import numpy as np
import pandas as pd
import os
from time import time

from mlpy.lib.utils.path import tag_path, makedirs
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS
from mlpy.datasets import ucr_uea
from mlpy.tsclu.base import kshape

from configure import DIR_LOG


def run(data_name_list, dir_data, dir_out):
    print("******** kshape over raw datasets")
    records = []
    n_datasets = len(data_name_list)
    for i, data_name in enumerate(data_name_list):
        print("******** [{}/{}] processing {}".format(i+1, n_datasets, data_name))
        # load datasets
        x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(data_name, dir_data)
        x = np.vstack([x_tr, x_te])
        y = np.hstack([y_tr, y_te])
        # start to run
        res = kshape(x, y, n_classes)
        res.update({'data_name': data_name})
        records.append(res)
    # save result
    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(dir_out, 'kshape_raw.csv'), index=False)
    return df


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))

    data_name_list = ['50words']
    # data_name_list = UCR85_DATASETS

    run(data_name_list, DIR_DATA_UCR15, log_dir)
