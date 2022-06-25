"""Base class for all TSC models realized by Keras

Primary references:
---------------------
https://github.com/hfawaz/dl-4-tsc
@article{fawaz2019deep,
  title={Deep learning for time series classification: a review},
  author={Fawaz, Hassan Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal={Data Mining and Knowledge Discovery},
  volume={33},
  number={4},
  pages={917--963},
  year={2019},
  publisher={Springer}
}
Some experiment details:
    - train model with train split and use test split to monitor the training processing (Exactly it should be forbidden to use test set during training process).
    - author run 10 times to get average accuracy. (Note, it is not cross-validation, the differences maybe come from various random state).

Other open source:
- https://github.com/uea-machine-learning/sktime-dl
    That package also refer to the above reference.

"""

__author__ = 'Fanling Huang'

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import os
import numpy as np
import tensorflow as tf


class BaseClassifierDNNKeras(BaseEstimator):
    """Base class for the TSC model realized by Keras
    Notes:
        - tensor is a three-dimensional array corresponding multivariate time series.
        - y should be represented by one-hot vector
    """
    def __init__(self, input_shape, n_classes, verbose=1, name='model'):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.verbose = verbose
        self.name = name
        self.batch_size_tr = None
        self.model = None

    def build_model(self, **kwargs):
        raise NotImplementedError('This is an abstract method.')

    def fit(self, x,
            y,
            batch_size=None,
            n_epochs=None,
            validation_data=None,
            shuffle=True,
            **kwargs):
        raise NotImplementedError('This is an abstract method.')

    def predict_proba(self, x, batch_size=None, **kwargs):
        probas = self.model.predict(x, batch_size, **kwargs)
        return probas

    def predict(self, x, batch_size=None, **kwargs):
        probas = self.predict_proba(x, batch_size)
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    def evaluate(self, x, y, batch_size=None, **kwargs):
        y_pred = self.predict(x, batch_size)
        y_true = np.argmax(y, axis=1)
        acc = accuracy_score(y_true, y_pred)
        return acc

    def count_params(self):
        return self.model.count_params()

    def save(self, dir):
        self.model.save(os.path.join(dir, '{}.hdf5'.format(self.name)))

    def load(self, dir):
        self.model = tf.keras.models.load_model(os.path.join(dir, '{}.hdf5'.format(self.name)))


