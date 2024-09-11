import tensorflow as tf
from tensorflow.keras import layers
import os
from time import time
import pandas as pd

from ...lib.ops import conv_out_size_same, tensor2numpy, get_log_str, output_padding, load_ckpt
from ...lib.metrics import gan_accuracy, gan_loss
from ...lib.vis import save_series, save_loss


class TCGAN(object):
    def __init__(self, cfg, evaluator):
        self.cfg = cfg
        self.evaluator = evaluator
        self._build_model()

    def _build_model(self):
        # build generator
        self.generator = self.build_generator()
        if self.cfg.verbose:
            # print(self.cfg.x_shape)
            self.cfg.logger.info("Generator's summary: ")
            self.generator.summary(print_fn=self.cfg.logger.info)
        # build discriminator
        self.discriminator = self.build_discriminator()
        if self.cfg.verbose:
            self.cfg.logger.info("Discriminator's summary: ")
            self.discriminator.summary(print_fn=self.cfg.logger.info)

        # build optimizer
        self.g_opt = tf.keras.optimizers.Adam(self.cfg.g_lr, beta_1=self.cfg.g_beta1, name='g_opt')
        self.d_opt = tf.keras.optimizers.Adam(self.cfg.d_lr, beta_1=self.cfg.d_beta1, name='d_opt')

        # set checkpoint
        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.cfg.ckpt_dir, self.cfg.ckpt_max_to_keep)

    def build_generator_network(self, generated_sample_shape, input_layer):
        n_layers = self.cfg.g_layers
        kernel_size = self.cfg.kernel_size

        steps = generated_sample_shape[0]
        layer_steps = [steps]
        for i in range(n_layers):
            layer_steps.append(conv_out_size_same(layer_steps[-1], self.cfg.strides))
        layer_steps.reverse()

        conv_units = []
        if n_layers > 1:
            conv_units.append(self.cfg.g_units_base)
            for _ in range(n_layers - 2):  # minus the first and the last layers
                conv_units.append(conv_units[-1] * 2)
        conv_units.reverse()
        # the last layer must be aligned to the number of dimensions of input.
        conv_units.append(generated_sample_shape[-1])

        name = 'dense_0'
        h = layers.Dense(layer_steps[0] * conv_units[0] * 2, kernel_initializer=self.cfg.initializer,
                         name=f'{name}_dense')(input_layer)
        if self.cfg.g_norm is not None:
            h = self.cfg.g_norm(name=f'{name}_norm')(h)
        h = layers.ReLU(name=f'{name}_relu')(h)
        h = layers.Reshape((layer_steps[0], conv_units[0] * 2))(h)
        assert h.shape[1] == layer_steps[0]

        # fractional conv layers
        for i in range(n_layers):
            name = f'conv_{i}'
            if layer_steps[i] * self.cfg.strides == layer_steps[i + 1]:
                conv = layers.Conv1DTranspose(conv_units[i], kernel_size, self.cfg.strides, self.cfg.padding,
                                              kernel_initializer=self.cfg.initializer, name=f'{name}_conv')
            else:
                conv = layers.Conv1DTranspose(conv_units[i], kernel_size, self.cfg.strides, self.cfg.padding,
                                              output_padding=output_padding(kernel_size, self.cfg.strides),
                                              kernel_initializer=self.cfg.initializer, name=f'{name}_conv')
            h = conv(h)
            if i < n_layers - 1:
                # the last layer
                # - does not apply ReLU
                # - does not apply BatchNorm
                if self.cfg.g_norm is not None:
                    h = self.cfg.g_norm(name=f'{name}_norm')(h)
                h = layers.ReLU(name=f'{name}_relu')(h)
            assert h.shape[1] == layer_steps[i + 1]
        assert h.shape[-1] == generated_sample_shape[-1]

        return h

    def build_generator(self):
        z = layers.Input(self.cfg.noise_shape)
        output_layer = self.build_generator_network(self.cfg.x_shape, z)
        model = tf.keras.Model(z, output_layer)
        return model

    def build_discriminator_network(self, input_layer, with_flat=False):
        n_layers = self.cfg.d_layers
        kernel_size = self.cfg.kernel_size

        units = [self.cfg.d_units_base]
        for _ in range(n_layers - 1):  # exclude the first layer.
            units.append(units[-1] * 2)

        # conv layers
        h = input_layer
        for i in range(n_layers):
            name = f'conv_{i}'
            h = layers.Conv1D(units[i], kernel_size, self.cfg.strides, self.cfg.padding, kernel_initializer=self.cfg.initializer,
                              name=f'{name}_conv')(h)
            if i > 1:  # the first layer without batch-norm
                if self.cfg.d_norm is not None:
                    h = self.cfg.d_norm(name=f'{name}_norm')(h)
            h = layers.LeakyReLU(self.cfg.leak_slope, name=f'{name}_relu')(h)
            h = layers.Dropout(self.cfg.d_dropout_rate, name=f'{name}_dropout')(h)

        # fc layer
        flat = layers.Flatten(name='flatten')(h)
        out = layers.Dense(1)(flat)

        if with_flat:
            return out, flat
        else:
            return out

    def build_discriminator(self):
        input_layer = layers.Input(self.cfg.x_shape)
        output_layer = self.build_discriminator_network(input_layer)
        model = tf.keras.Model(input_layer, output_layer)
        return model

    # Notice the use of `tf.function`. It causes the function to be "compiled".
    @tf.function
    def train_step(self, data):
        samples, noise = data

        # Note:
        # GradientTape.gradient can only be called once on non-persistent tapes, therefore there are two tapes according
        # to generator and discriminator, respectively.
        # For more details about tf.GradientTape, please refer to the official tutorial:
        # https://www.tensorflow.org/api_docs/python/tf/GradientTape.
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_samples = self.generator(noise, training=True)

            real_logits = self.discriminator(samples, training=True)
            fake_logits = self.discriminator(generated_samples, training=True)

            d_loss, g_loss, real_loss, fake_loss = gan_loss(real_logits, fake_logits)

        acc = gan_accuracy(real_logits, fake_logits)

        def _opt_g():
            g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        def _opt_d():
            d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        def _null():
            pass

        _opt_g()
        tf.cond(acc <= self.cfg.acc_threshold_to_train_d, true_fn=_opt_d, false_fn=_null)

        res = {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'acc': acc
        }
        return res

    def load(self, ckpt_number=None):
        epoch = load_ckpt(self.ckpt, self.ckpt_manager, ckpt_number)
        return epoch

    def _fit_load(self, restore, ckpt_number):
        if restore:
            init_epoch = self.load(ckpt_number)
            if init_epoch >= self.cfg.epochs:
                self.cfg.logger.info(f"init_epoch={init_epoch} is greater than epochs={self.cfg.epochs}.")
            self.cfg.logger.info(f"restore model from epoch-{init_epoch} and continue to train.")
        else:
            self.cfg.logger.info("train from scratch.")
            self.cfg.clean_paths()
            init_epoch = 0
        return init_epoch

    def fit(self, data, test_data=None, restore=False, ckpt_number=None):
        def _get_file_name(ep):
            return os.path.join(self.cfg.train_dir, 'epoch_{:04d}.png'.format(ep))

        def _eval(_epoch):
            self.evaluator.eval(test_data, self.generate_data(test_data.shape[0]), epoch=_epoch)

        def _generate(_epoch):
            self.generate(self.cfg.noise_seed, _get_file_name(_epoch))

        self.cfg.logger.info("****** fit start ******")

        init_epoch = self._fit_load(restore, ckpt_number)

        if test_data is None:
            test_data = data
        dataset = tf.data.Dataset.from_tensor_slices(data).\
            shuffle(self.cfg.batch_size, reshuffle_each_iteration=True).\
            batch(self.cfg.batch_size)

        # take 1 batch of real samples to show
        real_samples = [e for e in dataset.take(1)][0]
        save_series(real_samples, os.path.join(self.cfg.train_dir, '000_real.png'))
        _generate(0)
        _eval(0)

        res_records = []
        for epoch in range(init_epoch, self.cfg.epochs):
            t_start = time()

            _res_list = []
            for i, batch in dataset.enumerate():
                noise = self.cfg.noise_sampler((batch.shape[0],) + self.cfg.noise_shape)
                _res = self.train_step((batch, noise))
                _res_list.append(_res)

            _res_list_np = [tensor2numpy(r) for r in _res_list]
            _res_df = pd.DataFrame.from_records(_res_list_np)
            res = _res_df.mean().to_dict()
            log_str = get_log_str(res, epoch, self.cfg.epochs)
            t = time() - t_start
            log_str += f", time={t:.4}"
            self.cfg.logger.info(log_str)
            res.update({'epoch': epoch, 'time': t})
            res_records.append(res)

            if (epoch + 1) % self.cfg.n_epochs_to_save_ckpt == 0 \
                    or (epoch + 1) == self.cfg.epochs:
                self.ckpt_manager.save(checkpoint_number=epoch+1)
                _generate(epoch+1)
            if (epoch + 1) % self.cfg.n_epochs_to_evaluate == 0 \
                    or (epoch + 1) == self.cfg.epochs:
                _eval(epoch+1)

        res_df = pd.DataFrame.from_records(res_records)
        save_loss(res_df, self.cfg.train_dir)

        self.cfg.logger.info("****** fit end ******")

    def generate(self, noise, file=None):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        generated_samples = self.generator(noise, training=False)

        if file is not None:
            save_series(generated_samples, file)

        return generated_samples

    def generate_data(self, n_samples):
        noise = self.cfg.noise_sampler((n_samples,) + self.cfg.noise_shape)
        fake_data = self.generate(noise)
        return fake_data.numpy()

    def discriminate(self, samples):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predicted_labels = self.discriminator(samples, training=False)
        return predicted_labels
