#!/usr/bin/env python3

"""
Module description
"""

from data_loader import DataLoader
from blocking import X2Blocker, Blocker, X4Blocker
from config import DATA_PATH, ROOT_DIR
from evaluation import get_scores
from clustering import Clustering

import itertools
import pandas as pd
import os


def create_pairs(groups):
    """
    Given a list of groups, create combinations of pairs.

    :param groups: lit, of lists with the instance_id's groups
    :return: pd.DataFrame with the left and right instance_id pairs
    """
    pairs_pred = list()
    for group in groups:
        iterator = itertools.combinations(group, 2)
        for combo in iterator:
            res = {'left_instance_id': combo[0],
                   'right_instance_id': combo[1]}
            pairs_pred.append(res)

    return pd.DataFrame(data=pairs_pred)


def run_pipeline(data, y, dataset_id, evaluate=False, verbose=False, optimize_method=False, **kwargs):
    """
    It performs the basic logic pipeline for getting the data, creates blocking and clustering
    and stores the matching pairs.

    :param data: pd.DataFrame, the input dataset
    :param y: pd.DataFrame, the dataset with the actual pairs
    :param dataset_id: int, the id of the input dataset
    :param evaluate: boolean, if evaluation should be run
    :param verbose: boolean, if logging
    :return: pd.DataFrame, with the predicted matching pairs
    """
    cluster_n = kwargs.get('cluster_num', 0)
    distance_threshold = kwargs.get('distance_threshold', 0)
    method = kwargs.get('clustering_method', 'agglomerative')
    encoding = kwargs.get('encoding', 'use')

    # blocking
    if 'title' in data.columns:  # Instantiate correct blocker based on dataset
        blocker = X2Blocker()
        # identify X3
        if optimize_method and ("source" in data.instance_id[0]):
            method = 'cosine'
    elif 'name' in data.columns:
        blocker = X4Blocker()
        data["title"] = data["name"]
        if optimize_method:
            method = 'birch'
    else:
        raise ValueError("Please add a valid dataset id")

    blocker.fit(data=data)
    # blocks is of type [[instance_id]]
    # list of lists of instance_ids belonging to same group
    # transform returns modified data frame for further use
    blocks, data = blocker.transform()

    # apply clustering for each block to get matching pairs
    clusters = list()

    print("Method: ", method)
    print("Encoding: ", encoding, "\n")
    cls = Clustering(method=method, cluster_n=cluster_n, distance_threshold=distance_threshold, encoding=encoding)
    for block in blocks:
        if len(block) > 1:
            # filter data based on the instance_id's presented in the block
            block_df = data[data['instance_id'].isin(block)]
            clusters_l = cls.run(block_df)

            for c in clusters_l:
                clusters.append(c)

    # create pairs from clusters
    pairs_pred_df = create_pairs(clusters)

    dataset_scores = dict()
    if evaluate:
        # run performance evaluation
        dataset_scores = get_scores(actual=y, pred=pairs_pred_df)
        if verbose:
            print('Precision: {:.3f}'.format(dataset_scores['precision_score']))
            print('Recall: {:.3f}'.format(dataset_scores['recall_score']))
            print('F1 score: {:.3f}'.format(dataset_scores['f1_score']))

        dataset_scores['cluster_n'] = cluster_n
        dataset_scores['method'] = method
        dataset_scores['dataset'] = dataset_id
        dataset_scores['encoding'] = encoding

    return pairs_pred_df, dataset_scores


def main(datasets, evaluate=False, store=True, verbose=False, **kwargs):
    """
    It runs the pipeline for each dataset, collects predicted pairs
    and computes evaluation scores based on all of them.

    :param datasets: list, of int with the ids of the datasets
    :param evaluate: boolean, if evaluation should be run
    :param store: boolean, if the output should be stored
    :param verbose: boolean, if logging
    :return: dict, with the overall evaluations scores for all datasets
    """
    outputs = list()
    scores = list()

    cluster_n = kwargs.get('cluster_num', 0)
    distance_threshold = kwargs.get('distance_threshold', 1.5)
    clustering_method = kwargs['clustering_method']
    encoding = kwargs.get('encoding', None)
    optimize_method = kwargs.get('optimize_method', False)
    verbose = kwargs.get('verbose', True)
    evaluate = kwargs.get('evaluate', True)

    print('\n=== Pipeline === ')
    dl = DataLoader()
    for ds in datasets:
        # load data
        data, y = dl.load_data(ds)
        print("\nRunning for dataset {}\n".format(ds))
        predicted_pairs, dataset_scores = run_pipeline(data=data,
                                                       dataset_id=ds,
                                                       y=y,
                                                       clustering_method=clustering_method,
                                                       cluster_num=cluster_n,
                                                       distance_threshold=distance_threshold,
                                                       evaluate=evaluate,
                                                       verbose=verbose,
                                                       optimize_method=optimize_method,
                                                       encoding=encoding)
        outputs.append(predicted_pairs)
        scores.append(dataset_scores)

    output = pd.concat(outputs, ignore_index=True)
    actual_pairs = pd.concat(dl.actual_pairs_loaded, ignore_index=True)

    if store:
        output.to_csv(os.path.join(DATA_PATH, 'output.csv'), index=False)

    if evaluate:
        # run performance evaluation
        global_scores = get_scores(actual=actual_pairs, pred=output)
        if verbose:
            print('\n=== Evaluation === ')
            print('Precision: {:.3f}'.format(global_scores['precision_score']))
            print('Recall: {:.3f}'.format(global_scores['recall_score']))
            print('F1 score: {:.3f}'.format(global_scores['f1_score']), '\n')

        global_scores['cluster_n'] = cluster_n
        global_scores['method'] = clustering_method
        global_scores['dataset'] = 'all'
        global_scores['encoding'] = encoding

        scores.append(global_scores)

    return scores
