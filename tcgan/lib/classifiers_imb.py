import tensorflow as tf
from time import time
import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from mlpy.lib.data.utils import one_hot_to_dense, dense_to_one_hot


def imbalanced_metrics(y_true, y_pred, **kwargs):
    res = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', **kwargs),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', **kwargs)
    }
    return res


def softmax(x_tr, y_tr, x_te, y_te, n_classes, epochs=100, batch_size=16, verbose=0):
    t0 = time()

    if y_tr.ndim == 1:
        y_tr = dense_to_one_hot(y_tr, n_classes)
        y_te = dense_to_one_hot(y_te, n_classes)
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(n_classes,
                              input_shape=x_tr.shape[1:],
                              activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_tr, y_tr,
                        batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=(x_te, y_te))
    t = time() - t0

    y_pred = model.predict(x_te)
    res_imb = imbalanced_metrics(one_hot_to_dense(y_te), one_hot_to_dense(y_pred))

    acc_tr = model.evaluate(x_tr, y_tr, verbose=verbose)[1]
    acc_te = model.evaluate(x_te, y_te, verbose=verbose)[1]

    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t
    }
    res.update(res_imb)

    return res


def standard_clf(x_tr, y_tr, x_te, y_te, n_classes, model_cls, **kwargs):
    t0 = time()

    if y_tr.ndim == 2:
        y_tr = one_hot_to_dense(y_tr)
        y_te = one_hot_to_dense(y_te)

    model = model_cls(**kwargs)
    model.fit(x_tr, y_tr)
    acc_tr = accuracy_score(y_tr, model.predict(x_tr))
    acc_te = accuracy_score(y_te, model.predict(x_te))
    t = time() - t0

    y_pred = model.predict(x_te)
    res_imb = imbalanced_metrics(y_te, y_pred)

    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t
    }
    res.update(res_imb)
    return res


def lr(x_tr, y_tr, x_te, y_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, n_classes, LogisticRegression, **kwargs)


def lsvc(x_tr, y_tr, x_te, y_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, n_classes, LinearSVC, **kwargs)


def svc(x_tr, y_tr, x_te, y_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, n_classes, SVC, **kwargs)


def knn(x_tr, y_tr, x_te, y_te, n_classes):
    kwargs = dict(n_neighbors=1)
    return standard_clf(x_tr, y_tr, x_te, y_te, n_classes, KNeighborsClassifier, **kwargs)


def model_search(x_tr, y_tr, x_te, y_te, n_classes, model_cls, cv=5, n_jobs=1):
    t_0 = time()

    if y_tr.ndim == 2:
        y_tr = one_hot_to_dense(y_tr)
        y_te = one_hot_to_dense(y_te)

    # select penalty
    C = np.inf
    max_iter = 500
    model = model_cls(C=C, max_iter=max_iter)
    if x_tr.shape[0] // n_classes < 5 or x_tr.shape[0] < 50:
        model.fit(x_tr, y_tr)
    else:
        param_grid = {
            'C': [
                0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                np.inf
            ],
            'max_iter': [max_iter],
        }
        gs = GridSearchCV(
            model, param_grid, cv=cv, n_jobs=n_jobs, error_score=0
        )
        gs.fit(x_tr, y_tr)

        model = gs.best_estimator_
        C = gs.best_params_['C']

    acc_tr = accuracy_score(y_tr, model.predict(x_tr))
    acc_te = accuracy_score(y_te, model.predict(x_te))
    t = time() - t_0

    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t,
        'C': C
    }
    return res


def lr_search(x_tr, y_tr, x_te, y_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, LogisticRegression, cv=cv, n_jobs=n_jobs)


def lsvc_search(x_tr, y_tr, x_te, y_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, LinearSVC, cv=cv, n_jobs=n_jobs)


def svc_search(x_tr, y_tr, x_te, y_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, SVC, cv=cv, n_jobs=n_jobs)
