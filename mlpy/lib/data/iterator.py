import numpy as np

from sklearn import utils as skutils


class DataIterator(object):
    """ refer to tensorflow.examples.tutorials.mnist.input_data """
    # def __init__(self, X, y=None, batch_size=16, with_shuffle=True, seed=42):
    def __init__(self, X, y=None, batch_size=16, with_shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.with_shuffle = with_shuffle
        self.seed = seed

        self.n_samples = X.shape[0]
        self.reset()

    def reset(self):
        self.n_epochs_completed = 0
        self.i_in_epoch = 0

    def next_batch(self):
        if self.y is None:
            return self._next_batch_X()
        else:
            return self._next_batch_X_y()

    def _next_batch_X(self):
        i_start = self.i_in_epoch
        # shuffle for the first epoch
        if self.n_epochs_completed == 0 and i_start == 0 and self.with_shuffle:
            self.X = shuffle(self.seed, self.X)
        # Go to the next batch
        if i_start + self.batch_size > self.n_samples:
            # Finished epoch
            self.n_epochs_completed += 1
            # Get the rest samples in this epoch
            n_samples_rest = self.n_samples - i_start
            X_rest = self.X[i_start:self.n_samples]
            # Shuffle the datasets
            if self.with_shuffle:
                self.X = shuffle(self.seed, self.X)
            # Start next epoch
            i_start = 0
            self.i_in_epoch = self.batch_size - n_samples_rest
            i_end = self.i_in_epoch
            X_new = self.X[i_start:i_end]
            X_batch = np.concatenate([X_rest, X_new], axis=0)
        else:
            self.i_in_epoch += self.batch_size
            i_end = self.i_in_epoch
            X_batch = self.X[i_start:i_end]

        return X_batch

    def _next_batch_X_y(self):
        i_start = self.i_in_epoch
        # shuffle for the first epoch
        if self.n_epochs_completed == 0 and i_start == 0 and self.with_shuffle:
            self.X, self.y = shuffle(self.seed, self.X, self.y)
        # Go the next batch
        if i_start + self.batch_size > self.n_samples:
            # Finished epoch
            self.n_epochs_completed += 1
            # Get the rest samples in this epoch
            n_samples_rest = self.n_samples - i_start
            X_rest = self.X[i_start:self.n_samples]
            y_rest = self.y[i_start:self.n_samples]
            # Shuffle the datasets
            if self.with_shuffle:
                self.X, self.y = shuffle(self.seed, self.X, self.y)
            # Start next epoch
            i_start = 0
            self.i_in_epoch = self.batch_size - n_samples_rest
            i_end = self.i_in_epoch
            X_new = self.X[i_start:i_end]
            y_new = self.y[i_start:i_end]

            X_batch = np.concatenate([X_rest, X_new], axis=0)
            y_batch = np.concatenate([y_rest, y_new], axis=0)
        else:
            self.i_in_epoch += self.batch_size
            i_end = self.i_in_epoch
            X_batch = self.X[i_start:i_end]
            y_batch = self.y[i_start:i_end]

        return X_batch, y_batch


def shuffle_list(seed, *data):
    from numpy.random import RandomState
    np_rng = RandomState(seed)
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shuffle(seed, *arrays):
    if isinstance(arrays[0][0], str):
        return shuffle_list(seed, *arrays)
    else:
        return skutils.shuffle(*arrays, random_state=seed)




