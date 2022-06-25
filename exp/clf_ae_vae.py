import os
import fire

from mlpy.lib.utils.path import makedirs
from mlpy.lib.utils.log import set_logging
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.configure import DIR_DATA_UCR15

from configure import DIR_LOG
from tcgan.lib.classfiers import *

from tcgan.model.tcgan import TCGANConfig
from tcgan.model.vae.vae_tcgan_klloss import VAETCGANKLLoss

from ae import ExpUnitClfUCRAE
from tcgan.lib.exp_clf import ExperimentClf

tf_keras_set_gpu_allow_growth()


if __name__ == '__main__':
    tag = 'exp_ae_vae' # the pre-trained model
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)

    # data_name_list = UCR85_DATASETS
    data_name_list = ['50words']

    model_cfg = dict(kernel_size=10)

    exp_tag = 'base'
    exp_cfg = {
        'extractor_params': [
            {
                'idx_layer': -3,
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
            'svc': svc
        },
        'out_file': 'clf_{}.json'.format(exp_tag),
        'res_out_fname': 'res_clf_{}'.format(exp_tag)
    }

    exp = ExperimentClf(tag, VAETCGANKLLoss, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        model_cfg_kwargs=model_cfg,
                        exp_uni_obj=ExpUnitClfUCRAE,
                        **exp_cfg)
    fire.Fire(exp.run)
