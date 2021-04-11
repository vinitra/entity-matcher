#!/usr/bin/env python3

"""
Module description
"""
from pipeline import main

if __name__ == '__main__':
    dataset_ids = [1, 2, 3, 4]
    c_method = 'cosine'
    encoding = 'use'

    main(dataset_ids,
         clustering_method=c_method,
         encoding=encoding,
         evaluate=False,
         store=True)
