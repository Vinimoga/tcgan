from time import time
import numpy as np
import os
import json

from .metrics import *
from .classfiers import softmax
from .utils import extract


class EvaluatorGAN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = self.cfg.eval_dir
        self.logger = self.cfg.logger
        self.metrics = self.cfg.metrics

    def eval(self, x: np.array, x_fake: np.array, epoch: str = ""):
        self.logger.info("****** eval start ******")
        x_2d = np.reshape(x, (x.shape[0], -1))
        x_fake_2d = np.reshape(x_fake, (x_fake.shape[0], -1))

        if 'vis' in self.metrics:  # visually compare distributions.
            t0 = time()
            # Note: It is time-consuming. Visual analysis is cumbersome.
            tsne(x_2d, x_fake_2d,
                 os.path.join(self.log_dir, f'eval_tsne_{epoch}.png'))
            self.logger.info("tsne, time={}".format(time() - t0))

        metrics = {'epoch': epoch}

        # scores to measure similarity
        if 'nnd' in self.metrics:
            t0 = time()
            nnd = nnd_score(x_2d, x_fake_2d, metric='euclidean', batch_size=self.cfg.batch_size)
            self.logger.info("nnd, time={}".format(time() - t0))
            metrics.update({'nnd': np.round(np.float(nnd), 4)})

        if 'mmd' in self.metrics:
            t0 = time()
            mmd = mmd_score(x_fake_2d, x_2d)
            self.logger.info("mmd, time={}".format(time() - t0))
            metrics.update({'mmd': np.round(np.float(mmd), 4)})

        with open(os.path.join(self.log_dir, 'similarity.json'), 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        self.logger.info("****** eval end ******")

        return metrics


class EvaluatorClf(object):
    """
        reference
        - (kratzert's answer): https://github.com/keras-team/keras/issues/3465
        - official tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning
    """

    def __init__(self, log_dir, logger, extractor,
                 epochs=100, batch_size=16, random_runs=5, verbose=0):
        self.extractor = extractor
        self.log_dir = log_dir
        self.logger = logger
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_runs = random_runs
        self.verbose = verbose

    def eval(self, x_tr, y_tr, x_te, y_te, n_classes, epoch=""):
        self.logger.info("****** eval_clf start ******")

        # extract features
        t0 = time()
        feat_tr = extract(self.extractor, x_tr, self.batch_size)
        feat_te = extract(self.extractor, x_te, self.batch_size)
        feat_tr = feat_tr.reshape([feat_tr.shape[0], -1])
        feat_te = feat_te.reshape([feat_te.shape[0], -1])
        t_feat = time() - t0

        # eval with classifier
        acc_tr_list = []
        acc_te_list = []
        t0 = time()
        for i in range(self.random_runs):
            tt = time()
            _res = softmax(feat_tr, y_tr, feat_te, y_te, n_classes, self.epochs, self.batch_size, self.verbose)
            acc_tr = _res['acc_tr']
            acc_te = _res['acc_te']
            acc_tr_list.append(acc_tr)
            acc_te_list.append(acc_te)
            self.logger.info(
                f"[{i + 1}/{self.random_runs}], acc_tr={acc_tr:.4}, acc_te={acc_te:.4}, time={time() - tt}")
        t_eval = time() - t0

        res = {
            'epoch': epoch,
            'acc_tr': np.mean(acc_tr_list),
            'acc_tr_std': np.std(acc_tr_list),
            'acc_te': np.mean(acc_te_list),
            'acc_te_std': np.std(acc_te_list),
            'time_feature_extract': t_feat,
            'time_clf': t_eval
        }
        with open(os.path.join(self.log_dir, 'join_clf.json'), 'a') as f:
            f.write(json.dumps(res) + "\n")

        self.logger.info("time={}".format(time() - t0))
        self.logger.info("****** eval_clf end ******")
        return res