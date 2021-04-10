#!/usr/bin/env python3

# Created by romanou at 08.04.21

"""
Module description
"""
import pandas as pd

from pipeline import main


if __name__ == '__main__':
    dataset_ids = [1, 2, 3, 4]
    clusters_nums = [2, 5, 10]
    clustering_step = ['cosine', 'kmeans', 'jaccard']
    encodings = ['use']

    scores_to_store = list()
    for c_method in clustering_step:
        if c_method == 'kmeans':
            for cluster_num in clusters_nums:
                for text_encoding in encodings:
                    print('-' * 50)
                    print("\nRunning {} for the clustering step.".format(c_method))
                    print("Running for cluster number: {}".format(cluster_num))
                    eval_scores = main(dataset_ids,
                                       evaluate=True,
                                       clustering_method=c_method,
                                       cluster_num=cluster_num,
                                       encoding=text_encoding)
                    scores_to_store += eval_scores

        elif c_method == 'jaccard':
            print('-' * 50)
            print("\nRunning {} for the clustering step.".format(c_method))
            eval_scores = main(dataset_ids,
                               evaluate=True,
                               clustering_method=c_method)
            scores_to_store += eval_scores

        else:
            for text_encoding in encodings:
                print('-' * 50)
                print("\nRunning {} for the clustering step with {} encoding.".format(c_method, text_encoding))
                eval_scores = main(dataset_ids,
                                   evaluate=True,
                                   clustering_method=c_method,
                                   encoding=text_encoding)
                scores_to_store += eval_scores

    pd.DataFrame(scores_to_store).to_csv('hyper_param_res', index=False)
