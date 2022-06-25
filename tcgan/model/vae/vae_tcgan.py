import tensorflow as tf
from tensorflow.keras import layers
from abc import ABC

from .vae import CVAEModel, CVAE
from ..tcgan import TCGAN


class VAETCGAN(CVAE):
    def __init__(self, cfg):
        self.lr = cfg.d_lr
        super().__init__(cfg)

    def _build_model(self):
        self.model =VAETCGANModel(self.cfg)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)


class VAETCGANModel(CVAEModel, ABC):
    def __init__(self, cfg):
        super(VAETCGANModel, self).__init__(cfg)

    def _build_model(self):
        input_layer = layers.Input(self.cfg.x_shape)
        output_layer, latent_dim = self.build_encoder(input_layer, self.cfg.noise_shape)
        self.encoder = tf.keras.Model(input_layer, output_layer)

        z_shape = tf.TensorShape(latent_dim)
        z = layers.Input(z_shape)
        output_layer = self.build_decoder(self.cfg.x_shape, z)
        self.decoder = tf.keras.Model(z, output_layer)

    def build_encoder(self, input_layer, output_shape):
        _, flat = TCGAN.build_discriminator_network(self, input_layer, with_flat=True)
        units = 1
        for s in output_shape:
            units *= s
        h = layers.Dense(units + units)(flat)  # mean, logvar
        return h, [units]

    def build_decoder(self, generated_sample_shape, input_layer):
        h = TCGAN.build_generator_network(self, generated_sample_shape, input_layer)
        return h


