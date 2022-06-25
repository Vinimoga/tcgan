import os
import fire
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow.keras import layers


from mlpy.datasets import ucr_uea
from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.lib.utils.log import set_logging
from mlpy.configure import DIR_DATA_UCR15
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig

tf_keras_set_gpu_allow_growth()

""" 

Replace the last layer of GAN.discriminator with softmax layer, train it from scratch.
The resulted network = a FCN.

"""


class Classifier(object):
    def __init__(self, log_dir, logger, model_obj, model_cfg_obj, idx_layer, data_name, data_dir,
                 model_cfg_kwargs=None):
        self.log_dir = log_dir
        self.logger = logger
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.idx_layer = idx_layer
        self.data_name = data_name
        self.data_dir = data_dir
        self.model_cfg_kwargs = model_cfg_kwargs if model_cfg_kwargs is not None else {}

        self._load_data()
        self._init_model()

    def _load_data(self):
        self.x_tr, self.y_tr, self.x_te, self.y_te, self.n_classes = ucr_uea.load_ucr(
            self.data_name, self.data_dir, one_hot=True)
        self.x_tr = self.x_tr[..., np.newaxis]
        self.x_te = self.x_te[..., np.newaxis]

    def _init_model(self):
        self.input_shape = self.x_tr.shape[1:]
        self.cfg = self.model_cfg_obj(self.input_shape, self.log_dir, self.logger, **self.model_cfg_kwargs)
        self.base_model = self.model_obj(self.cfg, None).discriminator

    def run(self):
        inputs = self.base_model.input
        base_output = self.base_model.layers[self.idx_layer].output

        # new model
        flat = layers.Flatten()(base_output)
        predictions = layers.Dense(self.n_classes, activation='softmax')(flat)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

        if self.cfg.verbose:
            print(f"#trainable_variables={len(model.trainable_variables)}")

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        t = time()
        history = model.fit(self.x_tr, self.y_tr,
                            batch_size=self.cfg.batch_size, epochs=self.cfg.epochs, verbose=self.cfg.verbose,
                            validation_data=(self.x_te, self.y_te))
        t_tr = time() - t

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.cfg.train_dir, 'history.csv'))
        model.save(self.cfg.ckpt_dir)

        _, acc_tr = model.evaluate(self.x_tr, self.y_tr)
        t = time()
        _, acc_te = model.evaluate(self.x_te, self.y_te)
        t_te = time() - t

        res = {
            'data_name': self.data_name,
            'acc_train': acc_tr,
            'acc_test': acc_te,
            'n_params': model.count_params(),
            'time_train': t_tr,
            'time_test': t_te
        }

        return res


class ExperimentClf(object):
    def __init__(self,
                 tag,
                 model_obj,
                 model_cfg_obj,
                 data_dir,
                 data_name_list,
                 log_dir,
                 model_cfg_kwargs=None):
        self.tag = tag
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.data_dir = data_dir
        self.data_name_list = data_name_list
        self.log_dir = log_dir
        self.model_cfg_kwargs = model_cfg_kwargs if model_cfg_kwargs is not None else {}

    def run(self, i_run=0):
        log_dir = makedirs(os.path.join(self.log_dir, str(i_run)))
        logger = set_logging(self.tag, log_dir)

        records = []
        n_datasets = len(self.data_name_list)
        for i, data_name in enumerate(self.data_name_list):
            logger.info(f"****** [{i+1}/{n_datasets}] process dataset {data_name}")
            _log_dir = os.path.join(log_dir, data_name)
            tf.keras.backend.clear_session()

            clf = Classifier(_log_dir, logger, self.model_obj, self.model_cfg_obj, -3, data_name, self.data_dir,
                             self.model_cfg_kwargs)
            res = clf.run()
            print(res)
            records.append(res)
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(log_dir, 'metrics.csv'), index=False)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)
    # data_name_list = UCR85_DATASETS
    data_name_list = ['50words']

    model_cfg_kwargs = dict(acc_threshold_to_train_d=0.75, kernel_size=10)
    exp = ExperimentClf(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        model_cfg_kwargs=model_cfg_kwargs)
    fire.Fire(exp.run)

