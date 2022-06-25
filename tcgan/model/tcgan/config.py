
import tensorflow as tf

from ...lib.config import GANConfig


class TCGANConfig(GANConfig):
    def __init__(self,
                 input_shape,
                 log_dir,
                 logger,
                 strides=2,
                 padding='same',
                 initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
                 leak_slope=0.2,
                 **kwargs):
        self.strides = strides
        self.padding = padding
        self.initializer = initializer
        self.leak_slope = leak_slope
        super().__init__(input_shape, log_dir, logger, **kwargs)
        self.print_items()

