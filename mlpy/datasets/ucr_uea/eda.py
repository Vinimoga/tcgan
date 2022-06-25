import os
import numpy as np
import pandas as pd
import json

from mlpy.lib.features.uni_ts_dataset import autocorr, znorm_statistics
from mlpy.lib.features.imbalance_dataset import imbalance_scores
from mlpy.lib.data.utils import distribute_y_json, distribute_y
from mlpy.lib.utils.path import makedirs

from mlpy.datasets import ucr_uea
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

LOG_DIR = 'result'


class EDAReport(object):
    def __init__(self, data_name_list, data_dir, log_dir):
        self.data_name_list = data_name_list
        self.data_dir = data_dir
        self.log_dir = makedirs(log_dir)

        self.n_datasets = len(self.data_name_list)

    def run(self):
        self.check_znorm()
        self.cal_autocorr()
        self.cal_imbalanced_degree()
        self.get_data_info()

    def check_znorm(self):
        print("========== check_znorm ==========")
        records = []
        for i, data_name in enumerate(self.data_name_list):
            print(f"[{i+1}/{self.n_datasets}] processing data {data_name}")
            x_tr, _, x_te, _, _ = ucr_uea.load_ucr(data_name, self.data_dir)
            res = {'data_name': data_name}
            features = znorm_statistics(np.vstack([x_tr, x_te]))
            res.update(features)
            records.append(res)
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(self.log_dir, "znorm_statistics.csv"), index=False)

    def cal_autocorr(self):
        print("========== cal_autocorr ==========")
        records = []
        for i, data_name in enumerate(self.data_name_list):
            print(f"[{i+1}/{self.n_datasets}] processing data {data_name}")
            x_tr, _, x_te, _, _ = ucr_uea.load_ucr(data_name, self.data_dir)
            res = {'data_name': data_name}
            features = autocorr(np.vstack([x_tr, x_te]))
            res.update(features)
            records.append(res)
            print(res)
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(self.log_dir, "autocorr.csv"), index=False)

    def cal_imbalanced_degree(self):
        print("========== cal_imbalanced_degree ==========")
        records = []
        for i, data_name in enumerate(self.data_name_list):
            print(f"[{i+1}/{self.n_datasets}] processing data {data_name}")
            _, y_train, _, _, n_classes = ucr_uea.load_ucr(data_name, self.data_dir)
            res = {'data_name': data_name}
            im_scores = imbalance_scores(y_train, n_classes)
            res.update(im_scores)
            records.append(res)
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(self.log_dir, 'imbalance_scores.csv'), index=False)

    def get_data_info(self):
        print("========== get_data_info ==========")
        res_col = ['data_name', 'n_classes', 'length',
                   'size_all', 'size_train', 'size_test',
                   'distribution_all', 'distribution_train', 'distribution_test']
        res_df = pd.DataFrame(columns=res_col)
        for i, data_name in enumerate(self.data_name_list):
            print(f"[{i+1}/{self.n_datasets}] processing data {data_name}")

            x_train, y_train, x_test, y_test, n_class = ucr_uea.load_ucr(data_name, self.data_dir)
            x_all = np.concatenate([x_train, x_test], axis=0)
            y_all = np.concatenate([y_train, y_test])

            res_df.loc[i, 'data_name'] = data_name
            res_df.loc[i, 'n_classes'] = n_class
            res_df.loc[i, 'length'] = x_all.shape[1]
            res_df.loc[i, 'size_all'] = x_all.shape[0]
            res_df.loc[i, 'size_train'] = x_train.shape[0]
            res_df.loc[i, 'size_test'] = x_test.shape[0]
            res_df.loc[i, 'avg_size_on_class_tr'] = int(res_df.loc[i, 'size_train'] / n_class)
            res_df.loc[i, 'avg_size_on_class_te'] = int(res_df.loc[i, 'size_test'] / n_class)
            y_distr_tr, y_distr_te = distribute_y(y_train), distribute_y(y_test)
            res_df.loc[i, 'max_nsamples_on_class_tr'] = max(y_distr_tr.values())
            res_df.loc[i, 'min_nsamples_on_class_tr'] = min(y_distr_tr.values())
            res_df.loc[i, 'max_nsamples_on_class_te'] = max(y_distr_te.values())
            res_df.loc[i, 'min_nsamples_on_class_te'] = min(y_distr_te.values())
            res_df.loc[i, 'distribution_all'] = distribute_y_json(y_all)
            res_df.loc[i, 'distribution_train'] = distribute_y_json(y_train)
            res_df.loc[i, 'distribution_test'] = distribute_y_json(y_test)


        res_df.to_csv(os.path.join(self.log_dir, 'dataset_info.csv'), index=False)


if __name__ == '__main__':
    er = EDAReport(UCR85_DATASETS, DIR_DATA_UCR15, LOG_DIR)
    er.run()



