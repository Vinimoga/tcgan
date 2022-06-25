"""
    Use t-SNE or PCA to visualize how close both real and synthetic samples are.
    Reference:
        - https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
"""

import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def tsne(real_data, fake_data, path):
    n_components = 2

    n_samples = min(real_data.shape[0], fake_data.shape[0])
    colors = ['red'] * n_samples + ['blue'] * n_samples

    model = TSNE(n_components)
    results = model.fit_transform(np.concatenate([real_data[:n_samples], fake_data[:n_samples]], axis=0))

    f, ax = plt.subplots(1)
    plt.scatter(results[:n_samples, 0], results[:n_samples, 1], c=colors[:n_samples], alpha=0.2, label='real')
    plt.scatter(results[n_samples:, 0], results[n_samples:, 1], c=colors[n_samples:], alpha=0.2, label='fake')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pca(real_data, fake_data, path):
    n_components = 2

    n_samples = min(real_data.shape[0], fake_data.shape[0])
    colors = ['red'] * n_samples + ['blue'] * n_samples

    model = PCA(n_components)
    model.fit(real_data)
    res_real = model.transform(real_data)
    res_fake = model.transform(fake_data)

    f, ax = plt.subplots(1)
    plt.scatter(res_real[:n_samples, 0], res_real[:n_samples, 1], c=colors[:n_samples], alpha=0.2, label='real')
    plt.scatter(res_fake[n_samples:, 0], res_fake[n_samples:, 1], c=colors[n_samples:], alpha=0.2, label='fake')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

