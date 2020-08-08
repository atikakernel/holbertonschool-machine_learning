#!/usr/bin/env python3
"""
Kmean
"""

import sklearn.cluster


def kmeans(X, k):
    """ performs K-means on a dataset
    Returns: (C, clss) or (None, None) on failure
        - C: np.ndarray (k, d) with the centroid means for each cluster
        - clss: np.ndarray (n,) with the index of the cluster in C that
                each data point belongs to
    """
    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)
    C = Kmean.cluster_centers_
    clss = Kmean.labels_
    return C, clss
