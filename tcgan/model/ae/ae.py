"""
    Vanila AE
"""

from abc import ABC
import os
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from mlpy.lib.utils.config import MLModelConfig

from ...lib.vis import save_series


class AEConfig(MLModelConfig):
    def __init__(self,
                 input_shape,
                 log_dir,
                 logger,
                 latent_dim=100,
                 batch_size=16,
                 epochs=300,
                 seed=42,
                 verbose=0,
                 **kwargs
                 ):
        super().__init__(log_dir, logger, seed=seed, verbose=verbose, **kwargs)

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs


class AutoEncoder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.model = AutoEncoderModel(self.cfg)
        self.model.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def fit(self, x_tr, x_te=None, restore=False):
        history = self.model.fit(x_tr, x_tr,
                                 epochs=self.cfg.epochs,
                                 shuffle=True,
                                 batch_size=self.cfg.batch_size,
                                 validation_data=(x_te, x_te))
        self.save()
        self.save_results(x_te, history)

    def save(self):
        self.model.save(os.path.join(self.cfg.ckpt_dir, 'model'))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join(self.cfg.ckpt_dir, 'model'))

    def save_results(self, x_te, history):
        df = pd.DataFrame(history.history)
        df.to_csv(os.path.join(self.cfg.train_dir, 'loss.csv'))
        df.plot()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.train_dir, 'loss.png'))
        plt.close()

        decoded_samples = self.model(x_te)
        save_series(x_te, os.path.join(self.cfg.eval_dir, 'real_sample.png'))
        save_series(decoded_samples, os.path.join(self.cfg.eval_dir, 'decoded_sample.png'))


class AutoEncoderModel(Model, ABC):
    def __init__(self, cfg):
        super(AutoEncoderModel, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(self.cfg.latent_dim, activation='relu'),
        ])

        dense_dim = 1
        for s in self.cfg.input_shape:
            dense_dim *= s
        self.decoder = tf.keras.Sequential([
            layers.Dense(dense_dim),
            layers.Reshape(self.cfg.input_shape)
        ])

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
