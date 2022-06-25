import math
import os
import tensorflow as tf


def conv_out_size_same(size, stride):  # padding = same
    return int(math.ceil(float(size) / float(stride)))


def conv_concat(h, y):
    y_dup = tf.keras.layers.RepeatVector(h.shape[1])(y)
    return tf.keras.layers.Concatenate(axis=-1)([h, y_dup])


def output_padding(kernel_size, strides):  # padding = same
    """
    References:
        - https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv1DTranspose?hl=zh_cn
            - new_timesteps = ((timesteps - 1) * strides + kernel_size - 2 * padding + output_padding)
    """
    return (kernel_size - 1) % strides


def get_normalization_layer_class(method):
    if method == 'batch':
        return tf.keras.layers.BatchNormalization
    elif method == 'layer':
        return tf.keras.layers.LayerNormalization
    elif method == 'instance':
        return InstanceNormalization
    elif method == 'null':
        return None
    else:
        raise ValueError(f"normalization method={method} can not be found!")


def tensor2numpy(res):
    for key, val in res.items():
        res[key] = val.numpy()
    return res


def get_log_str(res, epoch, epochs):
    str_log = f"epoch[{epoch + 1}/{epochs}], "
    for key, val in res.items():
        str_log += f"{key}={val:.4}, "
    return str_log


def load_ckpt(ckpt, ckpt_manager, ckpt_number=None):
    if ckpt_number is None:
        if ckpt_manager.latest_checkpoint:
            # I did not apply assert_consumed() for the following reason, which is copied from the official
            # tutorial ( https://www.tensorflow.org/guide/checkpoint ):
            # "There are many objects in the checkpoint which haven't matched, including the layer's kernel and the
            # optimizer's variables. status.assert_consumed only passes if the checkpoint and the program match exactly,
            # and would throw an exception here."
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored checkpoint from {}".format(ckpt_manager.latest_checkpoint))
            epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        else:
            print("Initializing from scratch.")
            epoch = 0
    else:
        print("Restored checkpoint from a specific checkpoint_number={}".format(ckpt_number))
        ckpt_dir = os.path.dirname(ckpt_manager.latest_checkpoint)
        ckpt_latest = os.path.basename(ckpt_manager.latest_checkpoint)
        ckpt_path = os.path.join(ckpt_dir, ckpt_latest.split('-')[0] +"-" + str(ckpt_number))
        ckpt.restore(ckpt_path)
        epoch = ckpt_number
    return epoch


class InstanceNormalization(tf.keras.layers.Layer):
    """ Instance Normalization Layer (https://arxiv.org/abs/1607.08022). """

    def __init__(self, epsilon=1e-5, name=None):
        super(InstanceNormalization, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
