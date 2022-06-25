import os
import fire
import tensorflow as tf
import numpy as np

from mlpy.lib.data.utils import one_hot_to_dense
from mlpy.lib.utils.path import makedirs
from mlpy.lib.utils.log import set_logging
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig
from tcgan.lib.exp_clf import ExpUnitClfUCR

from vis import tsne

SEED = 86
np.random.seed(SEED)
tf_keras_set_gpu_allow_growth()


class ExpUnitVisUCR(ExpUnitClfUCR):
    def run(self):
        feat_tr, feat_te = self.prepare_data()
        y_tr = one_hot_to_dense(self.y_tr)
        y_te = one_hot_to_dense(self.y_te)

        path = makedirs(os.path.join(self.model_cfg.eval_dir, 'vis'))
        tsne(feat_tr, y_tr, feat_te, y_te, path)

        tf.keras.backend.clear_session()

        return None


class Experiment(object):
    def __init__(self, tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                 exp_uni_obj=None,
                 model_cfg_kwargs=None,
                 training=True,
                 **kwargs):
        self.tag = tag
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.data_dir = data_dir
        self.data_name_list = data_name_list
        self.model_cfg_kwargs = model_cfg_kwargs
        self.log_dir = log_dir
        self.exp_uni_obj = exp_uni_obj
        if self.exp_uni_obj is None:
            self.exp_uni_obj = ExpUnitClfUCR
        self.training = training
        self.kwargs = kwargs

    def run(self, i_run=0):
        log_dir = makedirs(os.path.join(self.log_dir, str(i_run)))
        logger = set_logging(self.tag, log_dir)

        for data_name in self.data_name_list:
            logger.info(f"****** process dataset {data_name}")
            _log_dir = makedirs(os.path.join(log_dir, data_name))

            exp_unit = self.exp_uni_obj(self.tag,
                                        _log_dir,
                                        self.data_dir,
                                        data_name,
                                        self.model_obj,
                                        self.model_cfg_obj,
                                        model_cfg_kwargs=self.model_cfg_kwargs,
                                        training=self.training,
                                        **self.kwargs)
            exp_unit.run()


if __name__ == '__main__':
    tag = 'exp_tcgan_'  # the pre-trained model
    log_dir = os.path.join(DIR_LOG, tag)
    logger = set_logging(tag, log_dir)

    # data_name_list = UCR85_DATASETS
    data_name_list = ['Two_Patterns']

    model_cfg = dict(acc_threshold_to_train_d=0.75, kernel_size=10)

    exp_cfg = {
        'extractor_params': [
            {
                'idx_layer': 'conv_2_relu',  # that is equal to 'idx_layer': -3,
                'pool': {
                    'obj': tf.keras.layers.MaxPool1D,
                    'kwargs': {
                        'pool_size': 1 / 3,
                        'strides': 1,
                        'padding': 'same'
                    }}
            },
            {
                'idx_layer': 'conv_3_relu',  # that is equal to 'idx_layer': -3,
                'pool': {
                    'obj': tf.keras.layers.MaxPool1D,
                    'kwargs': {
                        'pool_size': 1/3,
                        'strides': 1,
                        'padding': 'same'
                    }}
            },
        ],
        'norm': 'znorm'
    }

    exp = Experiment(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                     exp_uni_obj=ExpUnitVisUCR,
                     model_cfg_kwargs=model_cfg,
                     **exp_cfg)

    fire.Fire(exp.run)
