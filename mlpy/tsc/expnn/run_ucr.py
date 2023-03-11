import os
from time import time
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from mlpy.datasets import ucr_uea
from mlpy.lib.utils.path import makedirs
from mlpy.lib.data.utils import one_hot_to_dense
from mlpy.lib.tfops.base import set_gpu_allow_growth_keras, tf_keras_set_gpu_allow_growth
from mlpy.tsc.nn import *

MODELS_FOR_RUN = {
    'fcn': (FCN, 1),
    'resnet': (ResNet, 1),
}


def run(ModelClass, fname_list, dir_data, dir_log, extend_dim=0, save_model=False, fit_kwargs=None):
    _, K = tf_keras_set_gpu_allow_growth()

    fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    res = {'datasets': [], 'acc_train': [], 'acc_test': [], 'f1_macro': [], 'f1_weighted': [],
           'n_params': [], 't_train': [], 't_test': []}
    n = len(fname_list)
    for i, data_name in enumerate(fname_list):
        dir_log_data = makedirs(os.path.join(dir_log, data_name))

        print("********** [{}]/[{}] {} **********".format(i + 1, n, data_name))
        K.clear_session()
        # load datasets and set parameters
        x_train, y_train, x_test, y_test, n_classes = ucr_uea.load_ucr(
            data_name, dir_data, one_hot=True)
        if extend_dim > 0:
            extend_shape = ()
            for _ in range(extend_dim):
                extend_shape += (1,)
            x_train = x_train.reshape(x_train.shape + extend_shape)
            x_test = x_test.reshape(x_test.shape + extend_shape)
        # train model and log
        t0 = time()
        model = ModelClass(x_train.shape[1:], n_classes, name=data_name)
        df_metrics = model.fit(x_train, y_train, **fit_kwargs)
        t_train = time() - t0
        if save_model:
            model.save(dir_log)
        # test model
        t0 = time()
        acc_train = model.evaluate(x_train, y_train)
        acc_test = model.evaluate(x_test, y_test)
        t_test = time() - t0
        # log batch result
        y_pred_tr = model.predict(x_train)
        y_pred_te = model.predict(x_test)
        np.savetxt(os.path.join(dir_log_data, 'y_pred_tr.txt'), y_pred_tr)
        np.savetxt(os.path.join(dir_log_data, 'y_pred_te.txt'), y_pred_te)
        df_metrics.to_csv(os.path.join(dir_log_data, 'train_log.csv'))

        # imbalanced classification metrics
        y_true = one_hot_to_dense(y_test)
        f1_macro = f1_score(y_true, y_pred_te, average='macro')
        f1_weighted = f1_score(y_true, y_pred_te, average='weighted')

        res['datasets'].append(data_name)
        res['acc_train'].append(acc_train)
        res['acc_test'].append(acc_test)
        res['f1_macro'].append(f1_macro)
        res['f1_weighted'].append(f1_weighted)
        res['n_params'].append(model.count_params())
        res['t_train'].append(t_train)
        res['t_test'].append(t_test)

    df_res = pd.DataFrame(res)
    print(df_res.mean())
    df_res.to_csv(os.path.join(dir_log, 'metrics.csv'), index=False)



