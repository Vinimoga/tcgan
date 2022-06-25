import os
import numpy as np

from .path import makedirs


class MLModelConfig(object):
    def __init__(self,
                 log_dir,
                 logger,
                 seed=42,
                 verbose=0,
                 **kwargs
                 ):
        self.logger = logger
        self.log_dir = log_dir

        self.logger.info("****** configure init ******")

        """Set directories."""
        self.train_dir = makedirs(os.path.join(self.log_dir, 'training'))
        self.eval_dir = makedirs(os.path.join(self.log_dir, 'evaluation'))
        self.ckpt_dir = makedirs(os.path.join(self.log_dir, 'checkpoint'))
        self.ckpt_prefix = os.path.join(self.ckpt_dir, "ckpt")

        """Set random seed"""
        self.seed = seed
        self.np_rs = np.random.RandomState(self.seed)

        """Set others"""
        self.verbose = verbose

    def clean_paths(self):
        self.train_dir = makedirs(self.train_dir, clean=True)
        self.eval_dir = makedirs(self.eval_dir, clean=True)
        self.ckpt_dir = makedirs(self.ckpt_dir, clean=True)

    def print_items(self):
        log_str = "The settings are as follows: \n"
        for key, value in self.__dict__.items():
            log_str += f'{key}:{value}\n'
        self.logger.info(log_str)
