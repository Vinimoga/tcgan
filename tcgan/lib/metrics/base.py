import tensorflow as tf


def gan_accuracy(real_logits, fake_logits):
    real_prob = tf.keras.activations.sigmoid(real_logits)
    fake_prob = tf.keras.activations.sigmoid(fake_logits)
    y_real = tf.ones_like(real_prob)
    y_fake = tf.zeros_like(fake_prob)
    y = tf.concat([y_real, y_fake], axis=0)
    y_pred = tf.concat([real_prob, fake_prob], axis=0)
    acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y, y_pred))
    return acc


def gan_loss(real_logits, fake_logits):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # The discriminator's loss quantifies how well it is able to distinguish real images from fakes.
    # It compares the discriminator's predictions on real samples to an array of 1s, and the discriminator's
    # predictions on fake (generated) samples to an array of 0s.
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    d_loss = real_loss + fake_loss

    # The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator
    # is performing well, the discriminator will classify the fake samples as real (or 1). Here, we will compare the
    # discriminators decisions on the generated samples to an array of 1s.
    g_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)

    return d_loss, g_loss, real_loss, fake_loss