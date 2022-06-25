import numpy as np
import tensorflow as tf

from mlpy.lib.utils.config import MLModelConfig

from .ops import get_normalization_layer_class


class GANConfig(MLModelConfig):
    def __init__(self,
                 input_shape,
                 log_dir,
                 logger,
                 noise_dim=100,  # follows the tutorial
                 noise_method='normal',  # the method to generate noise
                 batch_size=16,  # ts dataset is usually small
                 epochs=300,
                 g_lr=0.0002,
                 d_lr=0.0002,
                 g_beta1=0.5,
                 d_beta1=0.5,
                 g_units_base=32,
                 d_units_base=32,
                 g_layers=4,  # >= 1
                 d_layers=4,  # >= 1
                 d_dropout_rate=0.0,  # default, without dropout
                 g_norm_method='batch',
                 d_norm_method='batch',
                 kernel_size=5,
                 acc_threshold_to_train_d=1.0,  # default, no balance, D:G=1:1
                 n_epochs_to_save_ckpt=15,
                 n_examples_to_generate=16,
                 n_to_evaluate=3,  # trade-off: evaluation is time wasting.
                 ckpt_max_to_keep=2,  # spends memory
                 seed=42,
                 verbose=0,
                 metrics=None,
                 **kwargs
                 ):
        super().__init__(log_dir, logger, seed=seed, verbose=verbose, **kwargs)
        self.x_shape = input_shape

        """Set noise"""
        if isinstance(noise_dim, int):
            self.noise_shape = (noise_dim, )
        elif isinstance(noise_dim, tuple):
            self.noise_shape = noise_dim
        else:
            dim = 1
            for s in input_shape:
                dim *= s
            if isinstance(noise_dim, float):
                self.noise_shape = (int(np.ceil(noise_dim * dim)),)
            self.noise_shape = (dim, )
        self.noise_method = noise_method
        if self.noise_method == 'normal':
            self.noise_sampler = tf.random.normal
        elif self.noise_method == 'uniform':
            self.noise_sampler = lambda shape, seed=None: tf.random.uniform(shape, -1, 1, seed=seed)
        else:
            raise ValueError(f"noise_method={self.noise_method} can not be found!")

        """Set training parameters"""
        self.batch_size = batch_size
        self.epochs = epochs
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_beta1 = g_beta1
        self.d_beta1 = d_beta1
        self.g_units_base = g_units_base
        self.d_units_base = d_units_base
        self.d_layers = d_layers
        self.g_layers = g_layers
        self.d_dropout_rate = d_dropout_rate
        self.g_norm = get_normalization_layer_class(g_norm_method)
        self.d_norm = get_normalization_layer_class(d_norm_method)
        self.kernel_size = kernel_size
        self.acc_threshold_to_train_d = tf.constant(acc_threshold_to_train_d, dtype=tf.float32)

        """Set logging parameters"""
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.n_epochs_to_save_ckpt = n_epochs_to_save_ckpt
        self.n_to_evaluate = n_to_evaluate
        self.n_epochs_to_evaluate = self.epochs // self.n_to_evaluate
        self.n_examples_to_generate = n_examples_to_generate

        """Init some parameters"""
        self.metrics = metrics
        if metrics is None:
            self.metrics = ['nnd', 'mmd', 'vis']
        # We will reuse the following seed samples overtime (so it's easier) to visualize progress.
        self._init_noise_seed()
        # self.noise_seed = self.noise_sampler((self.n_examples_to_generate, ) + self.noise_shape, seed=self.seed)

    def _init_noise_seed(self):
        self.noise_seed = self.noise_sampler((self.n_examples_to_generate,) + self.noise_shape, seed=self.seed)

