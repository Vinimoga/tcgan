from time import time
from sklearn.metrics import cluster

normalized_mutual_info_score = cluster.normalized_mutual_info_score


def kshape(x, y, n_clusters):
    """python -m pip install tslearn"""
    from tslearn.clustering import KShape

    model = KShape(n_clusters=n_clusters)
    t_start = time()
    model.fit(x)
    t = time() - t_start
    nmi = normalized_mutual_info_score(y, model.labels_)

    res = {
        'nmi': nmi,
        'time': t
    }

    return res


def kmeans(x, y, n_clusters):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters)
    t_start = time()
    model.fit(x)
    t = time() - t_start
    nmi = normalized_mutual_info_score(y, model.labels_)

    res = {
        'nmi': nmi,
        'time': t
    }

    return res
