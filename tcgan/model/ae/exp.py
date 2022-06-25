import tensorflow as tf

from ...lib.exp import ExpUnitClfData, ExpUnitUCRData, ExpUnitClf


class AEExpUnit(ExpUnitClfData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self):
        model = self.model_obj(self.model_cfg)
        model.fit(self.x_tr_gan, self.x_te_gan)
        tf.keras.backend.clear_session()

    def eval(self):
        pass


class AEExpUnitClf(ExpUnitClf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_extractor(self):
        model = self.model_obj(self.model_cfg)
        trained_epoch = model.load()
        base_model = model.model.encoder

        inputs = base_model.input
        if isinstance(self.idx_layer, int):
            base_output = base_model.layers[self.idx_layer].output
        else:
            base_output = base_model.get_layer(self.idx_layer).output

        extractor = tf.keras.models.Model(inputs=inputs, outputs=base_output)
        extractor.trainable = False

        return extractor, trained_epoch


class AEUCRExpUnitClf(ExpUnitUCRData, AEExpUnit, AEExpUnitClf):
    res_eval_fnames = ['join_clf.json']

    def __init__(self, *args, **kwargs):
        if 'use_testset' not in kwargs:
            kwargs['use_testset'] = True,  # use test set for AE training.
        if 'idx_layer' not in kwargs:
            kwargs['idx_layer'] = -1  # use the last layer in the Encoder.
        super().__init__(*args, **kwargs)

    def run(self):
        if self.training:
            self.fit()
        self.clf()
