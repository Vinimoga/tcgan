"""
    Re-use DCGAN network to construct a CNN auto-encoder.
"""


from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers, losses

from .ae import AutoEncoder, AutoEncoderModel
from ..tcgan import TCGAN


class AETCGAN(AutoEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _build_model(self):
        self.model = AEDCGANModel(self.cfg)
        self.model.compile(optimizer='adam', loss=losses.MeanSquaredError())


class AEDCGANModel(AutoEncoderModel, ABC):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _build_model(self):
        input_layer = layers.Input(self.cfg.x_shape)
        output_layer = self.build_encoder(input_layer, self.cfg.noise_shape)
        self.encoder = tf.keras.Model(input_layer, output_layer)

        z_shape = self.encoder.layers[-1].output.shape[1:]
        z = layers.Input(z_shape)
        output_layer = self.build_decoder(self.cfg.x_shape, z)
        self.decoder = tf.keras.Model(z, output_layer)

    def build_encoder(self, input_layer, output_shape):
        """That is similar to the discriminator in GAN."""
        _, flat = TCGAN.build_discriminator_network(self, input_layer, with_flat=True)
        units = 1
        for s in output_shape:
            units *= s
        h = layers.Dense(units, activation='relu')(flat)  # relu? activation = 'reLu'
        return h

    def build_decoder(self, generated_sample_shape, input_layer):
        h = TCGAN.build_generator_network(self, generated_sample_shape, input_layer)
        return h


