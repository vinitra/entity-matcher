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


def run_pipeline(data, dataset_id, params):
    """
    It performs the basic logic pipeline for getting the data, creates blocking and clustering
    and stores the matching pairs.

    :param data: pd.DataFrame, the input dataset
    :param dataset_id: int, the id of the input dataset
    :param params: float, the cutting threshold during entity similarity
    """
    # blocking
    if dataset_id in [1, 2, 3]:  # Instantiate correct blocker based on dataset
        blocker = X2Blocker()
    elif dataset_id == 4:
        blocker = X4Blocker()
    else:
        raise ValueError("Please add a valid dataset id")

    blocker.fit(data=data)
    # blocks is of type [[instance_id]]
    # list of lists of instance_ids belonging to same group
    # transform returns modified data frame for further use
    blocks, data = blocker.transform()

    # apply clustering for each block to get matching pairs
    clusters = list()
    cls = Clustering(cluster_n=params['clusters'])
    for block in blocks:
        if len(block) > 1:
            # filter data based on the instance_id's presented in the block
            block_df = data[data['instance_id'].isin(block)]
            clusters_l = cls.run(block_df)

            for c in clusters_l:
                clusters.append(c)

    # create pairs from clusters
    pairs_pred_df = create_pairs(clusters)

    return pairs_pred_df


def main(datasets, evaluate=False, store=True):
    """

    :param datasets:
    :param evaluate:
    :param store:
    :return:
    """
    outputs = list()

    dl = DataLoader()
    for ds in datasets:
        # load data
        data = dl.load_data(ds)
        print("\nRunning for dataset {}".format(ds))
        outputs.append(run_pipeline(data=data, dataset_id=ds, params=pipeline_args))

    output = pd.concat(outputs, ignore_index=True)
    actual_pairs = pd.concat(dl.actual_pairs_loaded, ignore_index=True)

    if store:
        output.to_csv(os.path.join(DATA_PATH, 'output.csv'), index=False)

    scores = dict()
    if evaluate:
        # run performance evaluation
        scores = get_scores(actual=actual_pairs, pred=output)
        print('Precision: {:.3f}'.format(scores['precision_score']))
        print('Recall: {:.3f}'.format(scores['recall_score']))
        print('F1 score: {:.3f}'.format(scores['f1_score']))

    return scores


if __name__ == '__main__':
    datasets_ids = [1, 2, 3, 4]
    hyperparams = {
        "clusters": [2, 4, 6, 8, 10],
    }

    res = list()
    for cluster_n in hyperparams['clusters']:
        print('-' * 50)
        print("\nRunning for cluster number: {}".format(cluster_n))
        pipeline_args = {
            "clusters": cluster_n
        }
        eval_scores = main(datasets_ids, evaluate=True)
        eval_scores['cluster_n'] = cluster_n
        eval_scores['confusion_matrix'] = ''
        res.append(eval_scores)

    pd.DataFrame(res).to_csv('hyper_param_res', index=False)
