"""
    Re-use DCGAN network to construct a De-noise AE.
"""

import tensorflow as tf


from ..tcgan.config import TCGANConfig
from ..ae.ae_tcgan import AETCGAN


class DAETCGANConfig(TCGANConfig):
    def __init__(self,
                 input_shape,
                 log_dir,
                 logger,
                 dnoise_factor=1.0,
                 dnoise_method='uniform',
                 **kwargs
                 ):
        super().__init__(input_shape, log_dir, logger, **kwargs)
        self.dnoise_factor = dnoise_factor
        # Note: the following sampler assumes that the data are z-normalized.
        self.dnoise_method = dnoise_method
        if self.dnoise_method == 'uniform':
            self.dnoise_sampler = lambda shape, seed=None: \
                tf.random.uniform(shape=shape, minval=-1, maxval=1, seed=seed)
        elif self.dnoise_method == 'normal':
            self.dnoise_sampler = lambda shape, seed=None: \
                tf.random.normal(shape=shape, seed=seed)
        else:
            raise ValueError(f"dnoise_method={self.dnoise_method} can not be found!")


class DAETCGAN(AETCGAN):  # denoised AE
    def __init__(self, cfg: DAETCGANConfig):
        super().__init__(cfg)

    def fit(self, x_tr, x_te=None, restore=False):
        x_tr_noisy = x_tr + self.cfg.dnoise_factor * self.cfg.dnoise_sampler(shape=x_tr.shape, seed=self.cfg.seed)
        x_te_noisy = x_te + self.cfg.dnoise_factor * self.cfg.dnoise_sampler(shape=x_te.shape, seed=self.cfg.seed)

        history = self.model.fit(x_tr_noisy, x_tr,
                                 epochs=self.cfg.epochs,
                                 shuffle=True,
                                 batch_size=self.cfg.batch_size,
                                 validation_data=(x_te_noisy, x_te))
        self.save()
        self.save_results(x_te, history)


