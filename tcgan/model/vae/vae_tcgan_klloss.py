import tensorflow as tf

from .vae_tcgan import VAETCGANModel, VAETCGAN


class VAETCGANKLLoss(VAETCGAN):
    def __init__(self, cfg):
        self.lr = cfg.d_lr
        super().__init__(cfg)

    def _build_model(self):
        self.model =VAETCGANModel(self.cfg)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def compute_loss(self, model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)

        error = tf.metrics.mean_squared_error(x, x_logit)
        axis = tf.range(1, tf.size(error.shape))

        error = tf.reduce_mean(tf.reduce_sum(error, axis=axis))

        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=axis))
        loss = error + kl_loss

        res = {
            'loss': loss,
            'error': error,
            'kl_loss': kl_loss
        }

        return res



