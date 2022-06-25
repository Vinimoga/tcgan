import os
from abc import ABC

import numpy as np
import json
import pandas as pd
import tensorflow as tf

from mlpy.lib.data.utils import train_test_split, one_hot_to_dense
from mlpy.lib.utils.log import set_logging
from mlpy.lib.utils.path import makedirs
from mlpy.datasets import ucr_uea

from .eval import EvaluatorGAN, EvaluatorClf


class ExpUnitBase(ABC):
    def __init__(self,
                 tag,
                 log_dir,
                 data_dir,
                 data_name,
                 model_obj,
                 model_cfg_obj,
                 model_cfg_kwargs=None,
                 training=True,
                 **kwargs):
        self.tag = tag
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.model_cfg_kwargs = model_cfg_kwargs if model_cfg_kwargs is not None else {}
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.data_name = data_name
        self.training = training
        self.kwargs = kwargs

        self.logger = set_logging("{}_{}".format(self.tag, self.data_name), self.log_dir)
        self._init_raw_data()
        self._init_input_shape()
        self.model_cfg = self.model_cfg_obj(self.input_shape, self.log_dir, self.logger, **self.model_cfg_kwargs)
        self._init_gan_data()

    def _init_raw_data(self):
        self.x_tr = None
        self.x_te = None

    def _init_input_shape(self):
        self.input_shape = self.x_tr.shape[1:]

    def _init_gan_data(self):
        self.x_tr_gan = self.x_tr
        self.x_te_gan = self.x_te

    def fit(self):
        pass

    def eval(self):
        pass

    def run(self):
        if self.training:
            self.fit()
        self.eval()


class ExpUnitYData(ExpUnitBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_raw_data(self):
        super()._init_raw_data()
        self.y_tr = None
        self.y_te = None


class ExpUnitClfData(ExpUnitYData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_raw_data(self):
        super()._init_raw_data()
        self.n_classes = None


class ExpUnitGAN(ExpUnitBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = EvaluatorGAN(self.model_cfg)

    def fit(self):
        evaluator = EvaluatorGAN(self.model_cfg)
        model = self.model_obj(self.model_cfg, evaluator)
        model.fit(self.x_tr_gan, self.x_te_gan)
        tf.keras.backend.clear_session()

    def eval(self):
        model = self.model_obj(self.model_cfg, self.evaluator)
        epoch = model.load()
        x_fake = model.generate_data(self.x_te.shape[0])
        self.evaluator.eval(self.x_te, x_fake, epoch='last-{}'.format(epoch))
        tf.keras.backend.clear_session()


class ExpUnitClf(ExpUnitClfData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'idx_layer' in kwargs:
            self.idx_layer = kwargs['idx_layer']
        else:
            raise ValueError("Please assign the required parameter 'idx_layer', it defines which layer in the "
                             "pre-trained network will be reused.")
        print(f"Reuse layer={self.idx_layer} for classification.")

    def get_extractor(self):
        model = self.model_obj(self.model_cfg, None)
        trained_epoch = model.load()
        base_model = model.discriminator

        inputs = base_model.input
        if isinstance(self.idx_layer, int):
            base_output = base_model.layers[self.idx_layer].output
        else:
            base_output = base_model.get_layer(self.idx_layer).output

        extractor = tf.keras.models.Model(inputs=inputs, outputs=base_output)
        extractor.trainable = False

        return extractor, trained_epoch

    def clf(self):
        extractor, trained_epoch = self.get_extractor()
        evaluator_clf = EvaluatorClf(self.model_cfg.eval_dir,
                                     self.model_cfg.logger,
                                     extractor,
                                     verbose=self.model_cfg.verbose)
        evaluator_clf.eval(self.x_tr, self.y_tr, self.x_te, self.y_te, self.n_classes,
                           epoch='last-{}'.format(trained_epoch))
        tf.keras.backend.clear_session()


class ExpUnitUCRData(ExpUnitClfData):
    def __init__(self, *args, **kwargs):
        if 'use_testset' in kwargs:
            self.use_testset = kwargs['use_testset']
        else:
            raise ValueError("Please assign the required parameter 'use_testset', which defines the way to construct "
                             "data for GAN training.")
        print(f"use_testset={self.use_testset}")
        super().__init__(*args, **kwargs)

    def _init_raw_data(self):
        x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(self.data_name, self.data_dir, one_hot=True)
        x_tr = x_tr[..., np.newaxis]
        x_te = x_te[..., np.newaxis]

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.n_classes = n_classes
        self.input_shape = self.x_tr.shape[1:]

    def _init_gan_data(self):
        if self.use_testset:
            # all data can be used in unsupervised learning
            x_all = np.vstack([self.x_tr, self.x_te])
            y_all = one_hot_to_dense(np.vstack([self.y_tr, self.y_te]))
            _, x_te_gan, _, _ = train_test_split(
                x_all, y_all, train_size=0.9, random_state=self.model_cfg.seed, stratify=y_all)
            x_tr_gan = x_all  # use all
        else:
            x_tr_gan, x_te_gan = self.x_tr, self.x_te

        self.x_tr_gan = x_tr_gan
        self.x_te_gan = x_te_gan


class ExpUnitUCRGANClf(ExpUnitUCRData, ExpUnitGAN, ExpUnitClf):
    res_eval_fnames = ['similarity.json', 'join_clf.json']

    def __init__(self, *args, **kwargs):
        if 'use_testset' not in kwargs:
            kwargs['use_testset'] = True,  # use test set for GAN training.
        if 'idx_layer' not in kwargs:
            kwargs['idx_layer'] = -3  # use the last conv layer in DCGAN.
        super().__init__(*args, **kwargs)

    def run(self):
        if self.training:
            self.fit()
        self.eval()
        self.clf()


class Experiment(object):
    def __init__(self, tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                 exp_uni_obj=None,
                 res_eval_fnames=None,
                 res_out_fname='res',
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
            self.exp_uni_obj = ExpUnitUCRGANClf
        self.res_eval_fnames = res_eval_fnames
        if self.res_eval_fnames is None:
            self.res_eval_fnames = self.exp_uni_obj.res_eval_fnames
        self.res_out_fname = res_out_fname
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

        self.reduce(log_dir, self.data_name_list, self.res_eval_fnames, self.res_out_fname)

    @staticmethod
    def reduce(log_dir, data_name_list, json_fnames, out_fname):
        res_list = []
        for data_name in data_name_list:
            res = {'data_name': data_name}
            for fname in json_fnames:
                with open(os.path.join(log_dir, data_name, 'evaluation', fname), 'r') as f:
                    last = f.readlines()[-1]
                    res.update(json.loads(last))
            res_list.append(res)

        df = {}
        if len(res_list) > 0:
            for key in res_list[0].keys():
                df[key] = []
            for r in res_list:
                for key in r.keys():
                    df[key].append(r[key])
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(log_dir, f'{out_fname}.csv'), index=False)
