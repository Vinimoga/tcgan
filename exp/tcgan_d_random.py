import os
import fire

from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.lib.utils.log import set_logging
from mlpy.configure import DIR_DATA_UCR15
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig

tf_keras_set_gpu_allow_growth()

from tcgan_d import ExperimentClf


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)
    # data_name_list = UCR85_DATASETS
    data_name_list = ['50words']

    model_cfg_kwargs = dict(epochs=0, acc_threshold_to_train_d=0.75, kernel_size=10)
    exp = ExperimentClf(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        model_cfg_kwargs=model_cfg_kwargs)
    fire.Fire(exp.run)

