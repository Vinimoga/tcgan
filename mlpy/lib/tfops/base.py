import tensorflow as tf
import keras
import os


def tf_keras_set_gpu_allow_growth():
    """
    reference: Feb 7 2020, https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    :return:
    """
    import tensorflow as tf
    if tf.__version__.startswith('1.'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K = tf.keras.backend
        K.set_session(sess)
    else:  # tf2.x
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        K = tf.keras.backend
    return tf, K


def set_gpu_allow_growth_keras(tfo):
    config = tfo.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tfo.Session(config=config)
    K = keras.backend
    K.set_session(sess)
    return K


def save_checkpoints(saver, sess, dir_save, step):
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    prefix = os.path.join(dir_save, "model.ckpt")
    saver.save(sess, prefix, global_step=step)


def load_checkpoints(saver, sess, dir_load):
    import re
    print("[*]  Reading checkpoints ...")
    ckpt = tf.train.get_checkpoint_state(dir_load)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(dir_load, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def optimizer_adam(loss, var_list, name_scope, lr=0.001, beta1=0.9):
    step = tf.Variable(0, trainable=False)
    with tf.variable_scope('opt_{}'.format(name_scope), reuse=tf.AUTO_REUSE):
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)\
            .minimize(loss=loss, var_list=var_list, global_step=step)
        return opt