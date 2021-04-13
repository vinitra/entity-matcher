#!/usr/bin/env python3

"""
Module description
"""
from pipeline import main

if __name__ == '__main__':
    dataset_ids = [1, 2, 3, 4]
    c_method = 'agglomerative'
    n_clusters = 2
    distance_threshold = 2.25
    encoding = 'use'

    main(dataset_ids,
         clustering_method=c_method,
         encoding=encoding,
         cluster_num=n_clusters,
         distance_threshold=distance_threshold,
         verbose=True,
         evaluate=True,
         store=True)
