"""
    Extend the base clf. For example, the extractor can process some customized settings.
"""
import os
import json
from copy import deepcopy

from sklearn.preprocessing import StandardScaler

from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets import ucr_uea

from tcgan.lib.exp import ExpUnitClfData, Experiment
from tcgan.lib.utils import extract
from tcgan.lib.classfiers import *

tf_keras_set_gpu_allow_growth()


class ExpUnitClf(ExpUnitClfData):
    # res_eval_fnames = ['clf.json']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_model()

    def _load_model(self):
        self.model = self.model_obj(self.model_cfg, None)
        self.trained_epoch = self.model.load()

    def get_base_model(self):
        return self.model.discriminator

    def get_extractor(self, idx_layer, pool=None):
        base_model = self.get_base_model()

        inputs = base_model.input
        if isinstance(idx_layer, int):
            base_output = base_model.layers[idx_layer].output
        else:
            base_output = base_model.get_layer(idx_layer).output

        if pool is None:
            h = base_output
        else:
            kwargs = deepcopy(pool['kwargs'])
            if kwargs['pool_size'] < 1.0:
                kwargs['pool_size'] = int(base_output.shape[1] * kwargs['pool_size'])
            if kwargs['pool_size'] >= base_output.shape[1]:
                kwargs['pool_size'] = base_output.shape[1] // 2
            pool_layer = pool['obj'](**kwargs)
            h = pool_layer(base_output)
        if len(h.shape) <= 2:  # 1-dimensional vector, needn't Flatten
            flat = h
        else:
            flat = tf.keras.layers.Flatten()(h)

        extractor = tf.keras.models.Model(inputs=inputs, outputs=flat)
        extractor.trainable = False

        return extractor

    def extract_features(self, extractor_list):
        feature_list_tr = []
        feature_list_te = []
        for e in extractor_list:
            feat_tr = extract(e, self.x_tr, self.model_cfg.batch_size)
            feat_te = extract(e, self.x_te, self.model_cfg.batch_size)
            feature_list_tr.append(feat_tr)
            feature_list_te.append(feat_te)

        feat_tr = np.hstack(feature_list_tr)
        feat_te = np.hstack(feature_list_te)

        return feat_tr, feat_te

    def prepare_data(self):
        extractor_list = []
        for param in self.kwargs['extractor_params']:
            idx_layer = param['idx_layer']
            pool = param['pool']
            extractor = self.get_extractor(idx_layer, pool=pool)
            extractor_list.append(extractor)

        feat_tr, feat_te = self.extract_features(extractor_list)

        norm = self.kwargs['norm']
        if norm is None:
            pass
        elif norm == 'znorm':
            scaler = StandardScaler()
            scaler.fit(feat_tr, self.y_tr)
            feat_tr = scaler.transform(feat_tr)
            feat_te = scaler.transform(feat_te)
        else:
            raise ValueError(f"norm={norm} can not be found!")
        return feat_tr, feat_te

    def run(self):
        t_start = time()
        feat_tr, feat_te = self.prepare_data()
        t_encode = time() - t_start

        clfs = self.kwargs['classifiers']
        res = {}
        res['time_encode'] = t_encode
        for name, clf in clfs.items():
            print(f"processing {name}")
            t_start = time()
            _res = clf(feat_tr, self.y_tr, feat_te, self.y_te, self.n_classes)
            t_clf = time() - t_start
            for k, v in _res.items():
                res[f'{name}_{k}'] = v
            res[f'time_{name}'] = t_clf

        out_file = self.kwargs['out_file']
        with open(os.path.join(self.model_cfg.eval_dir, out_file), 'a') as f:
            f.write(json.dumps(res) + "\n")

        tf.keras.backend.clear_session()


class ExpUnitClfUCR(ExpUnitClf):
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


class ExperimentClf(Experiment):
    def __init__(self, tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                 exp_uni_obj=None,
                 model_cfg_kwargs=None,
                 **kwargs):
        if exp_uni_obj is None:
            exp_uni_obj = ExpUnitClfUCR
        training = False
        res_eval_fnames = [kwargs['out_file']]
        super().__init__(tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                         exp_uni_obj=exp_uni_obj,
                         res_eval_fnames=res_eval_fnames,
                         model_cfg_kwargs=model_cfg_kwargs,
                         training=training, **kwargs)
