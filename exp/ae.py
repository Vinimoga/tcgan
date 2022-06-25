from tcgan.lib.exp_clf import ExpUnitClfUCR


class ExpUnitClfUCRAE(ExpUnitClfUCR):

    def _load_model(self):
        self.model = self.model_obj(self.model_cfg)
        self.trained_epoch = self.model.load()

    def get_base_model(self):
        return self.model.model.encoder

