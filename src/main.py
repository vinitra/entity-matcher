#!/usr/bin/env python3

"""
Module description
"""

from data_loader import DataLoader
from blocking import X2Blocker, Blocker
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


def run_pipeline(dataset_id, params, evaluate=False, store=True):
    """
    It performs the basic logic pipeline for getting the data, creates blocking and clustering
    and stores the matching pairs.

    :param dataset_id: int, the id of the input dataset
    :param evaluate: bool, run performance evaluation
    :param store: bool, store output file
    :param params: float, the cutting threshold during entity similarity
    """
    # load data
    dl = DataLoader()
    data, pairs_true = dl.load_data(dataset_id)

    # blocking
    if dataset_id == 2:  # Instantiate correct blocker based on dataset
        blocker = X2Blocker()
    else:
        blocker = Blocker()
    blocker.fit(data=data)
    # blocks is of type [[instance_id]]
    # list of lists of instance_ids belonging to same group
    # transform returns modified data frame for further use
    blocks, data = blocker.transform()

    # apply clustering for each block to get matching pairs
    clusters = list()
    cls = Clustering(cluster_n=params['clusters'])
    for block in blocks:
        # filter data based on the instance_id's presented in the block
        block_df = data[data['instance_id'].isin(block)]

        clusters_l = cls.run(block_df)

        for c in clusters_l:
            clusters.append(c)

    # create pairs from clusters
    pairs_pred_df = create_pairs(clusters)

    if store:
        pairs_pred_df.to_csv(os.path.join(DATA_PATH, 'results', 'results_{}.csv'.format(dataset_id)))

    if evaluate:
        # run performance evaluation
        scores = get_scores(actual=pairs_true, pred=pairs_pred_df)
        print('Precision: {:.3f}'.format(scores['precision_score']))
        print('Recall: {:.3f}'.format(scores['recall_score']))
        print('F1 score: {:.3f}'.format(scores['f1_score']))


if __name__ == '__main__':
    pipeline_args = {
        "clusters": 2
    }
    run_pipeline(dataset_id=2, params=pipeline_args, evaluate=True)
