#!/usr/bin/env python3

"""
Module description
"""
from pipeline import main

if __name__ == '__main__':
    dataset_ids = [1, 2, 3, 4]
    c_method = 'agglomerative'
    n_clusters = 2
    distance_threshold = 2
    encoding = 'use'

    # for dataset 1, agglomerative + w dist_threshold 2
    # for dataset 2, agglomerative + w dist_threshold 2
    # for dataset 3, cosine
    # for dataset 4, birch with threshold 0.2

    main(dataset_ids,
         clustering_method=c_method,
         encoding=encoding,
         cluster_num=n_clusters,
         distance_threshold=distance_threshold,
         verbose=True,
         evaluate=True,
         optimize_method=True,
         store=True)
