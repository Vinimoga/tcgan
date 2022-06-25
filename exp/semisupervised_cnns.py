import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.datasets.ucr_uea import load_ucr
from mlpy.tsc.nn import ResNet, FCN
from mlpy.lib.data.utils import dense_to_one_hot, split_stratified, distribute_y_json
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.configure import DIR_DATA_UCR15

from configure import DIR_LOG

_, K = tf_keras_set_gpu_allow_growth()


def run_resnet(data_name, dir_data, ratios):
    x_tr, y_tr, x_te, y_te, n_classes = load_ucr(data_name, dir_data)
    x_tr = x_tr.reshape(x_tr.shape + (1,))
    x_te = x_te.reshape(x_te.shape + (1,))
    y_te_onehot = dense_to_one_hot(y_te, n_classes)
    n_epochs = 300
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        print("*** processing ratio={}".format(r))
        x_tr_cur, y_tr_cur, _, _ = split_stratified(x_tr, y_tr, r)
        y_tr_cur_onehot = dense_to_one_hot(y_tr_cur, n_classes)
        model = ResNet(x_tr_cur.shape[1:], n_classes)
        df_metrics = model.fit(x_tr_cur, y_tr_cur_onehot, n_epochs=n_epochs)
        acc_te = model.evaluate(x_te, y_te_onehot)
        last = df_metrics.loc[df_metrics.shape[0] - 1, :]
        res['ratio'].append(r)
        res['distr'].append(distribute_y_json(y_tr_cur))
        res['acc'].append(last['acc'])
        res['acc_te'].append(acc_te)
    df_res = pd.DataFrame(res)
    return df_res


def run_fcn(data_name, dir_data, ratios):
    x_tr, y_tr, x_te, y_te, n_classes = load_ucr(data_name, dir_data)
    x_tr = x_tr.reshape(x_tr.shape + (1,))
    x_te = x_te.reshape(x_te.shape + (1,))
    y_te_onehot = dense_to_one_hot(y_te, n_classes)
    n_epochs = 300
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        print("*** processing ratio={}".format(r))
        x_tr_cur, y_tr_cur, _, _ = split_stratified(x_tr, y_tr, r)
        y_tr_cur_onehot = dense_to_one_hot(y_tr_cur, n_classes)
        model = FCN(x_tr_cur.shape[1:], n_classes)
        df_metrics = model.fit(x_tr_cur, y_tr_cur_onehot, n_epochs=n_epochs)
        acc_te = model.evaluate(x_te, y_te_onehot)
        last = df_metrics.loc[df_metrics.shape[0] - 1, :]
        res['ratio'].append(r)
        res['distr'].append(distribute_y_json(y_tr_cur))
        res['acc'].append(last['acc'])
        res['acc_te'].append(acc_te)
    df_res = pd.DataFrame(res)
    return df_res


if __name__ == '__main__':
    dir_data = DIR_DATA_UCR15
    tag = tag_path(os.path.abspath(__file__), 2)
    dir_out = makedirs(os.path.join(DIR_LOG, tag))

    data_name_list = ['FordA', 'wafer']
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01]
    n_runs = 10
    for data_name in data_name_list:
        dir_out_cur = makedirs(os.path.join(dir_out, data_name))
        # RestNet
        for i in range(n_runs):
            df_res = run_resnet(data_name, dir_data, ratios)
            df_res.to_csv(os.path.join(dir_out_cur, 'resnet_{}.csv'.format(i)), index=False)
        # FCN
        for i in range(n_runs):
            df_res = run_fcn(data_name, dir_data, ratios)
            df_res.to_csv(os.path.join(dir_out_cur, 'fcn_{}.csv'.format(i)), index=False)
