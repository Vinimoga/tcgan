import os
import fire
import json
import numpy as np
import tensorflow as tf

from mlpy.lib.data.utils import one_hot_to_dense
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS
from mlpy.tsclu.base import kmeans

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig
from tcgan.lib.exp_clf import ExpUnitClfUCR, ExperimentClf

tf_keras_set_gpu_allow_growth()


class ExpUnitClusterUCR(ExpUnitClfUCR):
    def run(self):
        feat_tr, feat_te = self.prepare_data()

        feat = np.vstack([feat_tr, feat_te])
        y = np.hstack([one_hot_to_dense(self.y_tr), one_hot_to_dense(self.y_te)])

        models = self.kwargs['models']
        res = {}
        for name, model in models.items():
            print(f"processing {name}")
            _res = model(feat, y, self.n_classes)
            for k, v in _res.items():
                res[f'{name}_{k}'] = v

        out_file = self.kwargs['out_file']
        with open(os.path.join(self.model_cfg.eval_dir, out_file), 'a') as f:
            f.write(json.dumps(res) + "\n")

        tf.keras.backend.clear_session()

        return res


if __name__ == '__main__':
    tag = 'exp_tcgan_' # the pre-trained model
    log_dir = os.path.join(DIR_LOG, tag)

    data_name_list = ['50words']
    # data_name_list = UCR85_DATASETS

    model_cfg = dict(acc_threshold_to_train_d=0.75, kernel_size=10)

    exp_tag = 'kmeans'
    exp_cfg = {
        'extractor_params': [
            {
                'idx_layer': 'conv_3_dropout',  # that is equal to 'idx_layer': -3,
                'pool': {
                    'obj': tf.keras.layers.MaxPool1D,
                    'kwargs': {
                        'pool_size': 2,
                        'strides': 1
                    }}
            },
        ],
        'norm': None,
        'models': {
            'kmeans': kmeans
        },
        'out_file': 'clus_{}.json'.format(exp_tag),
        'res_out_fname': 'res_clus_{}'.format(exp_tag)
    }

    exp = ExperimentClf(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        exp_uni_obj=ExpUnitClusterUCR,
                        model_cfg_kwargs=model_cfg,
                        **exp_cfg)

    fire.Fire(exp.run)










