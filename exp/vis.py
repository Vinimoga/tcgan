import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tsne(x_tr, y_tr, x_te, y_te, out_dir, n_components=2):
    model = TSNE(n_components)

    manifold_tr = model.fit_transform(x_tr)
    plt.scatter(manifold_tr[:, 0], manifold_tr[:, 1], c=y_tr)
    plt.savefig(os.path.join(out_dir, 'tr.png'))
    plt.clf()

    manifold_te = model.fit_transform(x_te)
    plt.scatter(manifold_te[:, 0], manifold_te[:, 1], c=y_te)
    plt.savefig(os.path.join(out_dir, 'te.png'))
    plt.clf()

    x = np.vstack([x_tr, x_te])
    y = np.hstack([y_tr, y_te])
    manifold = model.fit_transform(x)
    plt.scatter(manifold[:, 0], manifold[:, 1], c=y)
    plt.savefig(os.path.join(out_dir, 'all.png'))
    plt.clf()

    np.savetxt(os.path.join(out_dir, 'train.data'), manifold_tr, delimiter=',')
    np.savetxt(os.path.join(out_dir, 'test.data'), manifold_te, delimiter=',')

