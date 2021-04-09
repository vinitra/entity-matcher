#!/usr/bin/env python3

"""
Module description
"""
from pipeline import main

if __name__ == '__main__':
    dataset_ids = [1, 2, 3, 4]
    cluster_num = 2
    c_method = 'kmeans'
    encoding = 'use'

    main(dataset_ids,
         clustering_method=c_method,
         cluster_num=cluster_num,
         encoding=encoding,
         evaluate=False,
         store=True)
