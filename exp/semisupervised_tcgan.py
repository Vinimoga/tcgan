import os
from copy import deepcopy
import fire
import pandas as pd

from mlpy.lib.utils.log import set_logging
from mlpy.lib.utils.path import makedirs
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets import ucr_uea
from mlpy.configure import DIR_DATA_UCR15
from mlpy.lib.data.utils import train_test_split

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig
from tcgan.lib.classfiers import *

from tcgan.lib.exp_clf import ExpUnitClf

tf_keras_set_gpu_allow_growth()


class ExpUnitClfUCR(ExpUnitClf):
    def _init_raw_data(self):
        train_size = self.kwargs['train_size']

        x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(self.data_name, self.data_dir, one_hot=True)
        x_tr = x_tr[..., np.newaxis]
        x_te = x_te[..., np.newaxis]

        if train_size < 1:
            x_tr, _, y_tr, _ = train_test_split(x_tr, y_tr, train_size=train_size, stratify=y_tr)

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.n_classes = n_classes
        self.input_shape = self.x_tr.shape[1:]

    def run(self):
        feat_tr, feat_te = self.prepare_data()

        clfs = self.kwargs['classifiers']
        res = {}
        for name, clf in clfs.items():
            print(f"processing {name}")
            _res = clf(feat_tr, self.y_tr, feat_te, self.y_te, self.n_classes)
            for k, v in _res.items():
                res[f'{name}_{k}'] = v

        tf.keras.backend.clear_session()

        return res


class ExperimentSSL(object):
    def __init__(self,
                 tag,
                 model_obj,
                 model_cfg_obj,
                 data_dir,
                 data_name_list,
                 log_dir,
                 train_size_list=None,
                 exp_uni_obj=None,
                 model_cfg_kwargs=None,
                 **kwargs):
        self.tag = tag
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.data_dir = data_dir
        self.data_name_list = data_name_list
        self.model_cfg_kwargs = model_cfg_kwargs
        self.log_dir = log_dir
        self.train_size_list = train_size_list
        if self.train_size_list is None:
            self.train_size_list = [1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01]
        self.exp_uni_obj = exp_uni_obj
        if self.exp_uni_obj is None:
            self.exp_uni_obj = ExpUnitClfUCR
        self.kwargs = kwargs

    def _run(self, data_name, log_dir):
        records = []
        for train_size in self.train_size_list:
            kwargs = deepcopy(self.kwargs)
            kwargs['train_size'] = train_size

            exp_unit = self.exp_uni_obj(self.tag,
                                        log_dir,
                                        self.data_dir,
                                        data_name,
                                        self.model_obj,
                                        self.model_cfg_obj,
                                        model_cfg_kwargs=self.model_cfg_kwargs,
                                        **kwargs)
            res = exp_unit.run()
            res.update({'train_size': train_size})
            records.append(res)

        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(log_dir, 'ssl.csv'), index=False)
            
    def run(self, i_run=0):
        log_dir = makedirs(os.path.join(self.log_dir, str(i_run)))
        logger = set_logging(self.tag, log_dir)

        for data_name in self.data_name_list:
            logger.info(f"****** process dataset {data_name}")
            _log_dir = makedirs(os.path.join(log_dir, data_name))
            self._run(data_name, _log_dir)


if __name__ == '__main__':
    tag = 'exp_tcgan_' # the pre-trained model
    log_dir = os.path.join(DIR_LOG, tag)
    logger = set_logging(tag, log_dir)

    data_name_list = ['wafer', 'FordA']

    model_cfg = dict(acc_threshold_to_train_d=0.75, kernel_size=10)

    exp_tag = 'semisupervised'
    exp_cfg = {
        'train_size_list': [1.0, 0.8, 0.6, 0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01],
        'extractor_params': [
            {
                'idx_layer': 'conv_3_relu',  # that is equal to 'idx_layer': -3,
                'pool': {
                    'obj': tf.keras.layers.MaxPool1D,
                    'kwargs': {
                        'pool_size': 2,
                        'strides': 1
                    }}
            },
        ],
        'norm': None,
        'classifiers': {
            'softmax': softmax,
            'knn': knn,
            'lr': lr,
            'lsvc': lsvc,
        },
    }

    exp = ExperimentSSL(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15,
                        data_name_list, log_dir, model_cfg_kwargs=model_cfg,
                        **exp_cfg)

    fire.Fire(exp.run)










