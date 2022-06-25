__author__ = 'Fanling Huang'

import pandas as pd
import warnings
import tensorflow as tf

from .base import BaseClassifierDNNKeras


class ResNet(BaseClassifierDNNKeras):
    def __init__(self, input_shape, n_classes, verbose=1, name='resnet'):
        super(ResNet, self).__init__(input_shape, n_classes, verbose, name)
        
        # default parameters
        self.n_conv_filters = 64
        self.batch_size = 16
        self.n_epochs = 1500

        # set up model
        self.x = tf.keras.Input(self.input_shape)
        self.output = self.build_model()
        self.model = tf.keras.models.Model(inputs=self.x, outputs=self.output)
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        res1 = self.block_restnet(self.x, self.n_conv_filters)
        res2 = self.block_restnet(res1, self.n_conv_filters*2)
        res3 = self.block_restnet(res2, self.n_conv_filters*2)

        gap = tf.keras.layers.GlobalAveragePooling1D()(res3)
        out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(gap)

        return out

    @staticmethod
    def block_restnet(x, n_conv_filters):
        conv1 = tf.keras.layers.BatchNormalization()(x)
        conv1 = tf.keras.layers.Conv1D(n_conv_filters, 8, padding='same')(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation('relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(n_conv_filters, 5, padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(n_conv_filters, 3, padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)

        # Note: Two inputs of Add() layer should align on all dimensions, i.e., (batch_size, length, features) in which,
        # length dim: is definitely equal because it is the same as the input when padding='same'.
        # features dim: is adjusted by the following code if needed.
        is_expand_channels = not (x.shape[-1] == n_conv_filters)
        if is_expand_channels:
            conv1_skip = tf.keras.layers.Conv1D(n_conv_filters, 1, padding='same')(x)
            conv1_skip = tf.keras.layers.BatchNormalization()(conv1_skip)
        else:
            conv1_skip = tf.keras.layers.BatchNormalization()(x)
        conv4 = tf.keras.layers.Add()([conv1_skip, conv3])

        conv4 = tf.keras.layers.Activation('relu')(conv4)

        return conv4

    def fit(self, x, y,
            batch_size=None,
            n_epochs=None,
            validation_data=None,
            shuffle=True,
            **kwargs):
        # set parameters
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = min(x.shape[0] // 10, batch_size)  # default: \cite{wang2017time}
        if batch_size == 0:
            batch_size = min(self.batch_size, x.shape[0])
            warnings.warn("Reset the batch size to {} because batch size can not be 0."
                          .format(batch_size))
        if n_epochs is None:
            n_epochs = self.n_epochs

        # start to train
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if validation_data is None:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
            hist = self.model.fit(
                x, y, batch_size=batch_size, epochs=n_epochs,
                verbose=self.verbose, callbacks=[reduce_lr])
        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
            hist = self.model.fit(
                x, y, batch_size=batch_size, epochs=n_epochs,
                verbose=self.verbose, validation_data=validation_data, callbacks=[reduce_lr])
        return pd.DataFrame(hist.history)