import numpy as np


def extract(extractor, x, batch_size):
    features = []
    n_samples = x.shape[0]
    for i in range(n_samples // batch_size):
        feat = extractor(x[i * batch_size:(i + 1) * batch_size]).numpy()
        features.append(feat)
    residual = n_samples % batch_size
    if residual > 0:
        feat = extractor(x[-residual:])
        features.append(feat)
    features = np.vstack(features)

    return features
