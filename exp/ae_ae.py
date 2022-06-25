import os
import fire

from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.lib.utils.log import set_logging
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.configure import DIR_DATA_UCR15

from tcgan.lib.exp import Experiment
from tcgan.model.ae.exp import AEUCRExpUnitClf
from configure import DIR_LOG

from tcgan.model.tcgan import TCGANConfig
from tcgan.model.ae.ae_tcgan import AETCGAN

tf_keras_set_gpu_allow_growth()


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)

    # data_name_list = UCR85_DATASETS
    data_name_list = ['50words']

    model_cfg = dict(kernel_size=10)
    exp_cfg = dict(use_testset=True, idx_layer=-3)  # re-use the last conv layer.
    exp = Experiment(tag, AETCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                     exp_uni_obj=AEUCRExpUnitClf,
                     model_cfg_kwargs=model_cfg,
                     **exp_cfg)
    fire.Fire(exp.run)
