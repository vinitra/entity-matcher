#!/usr/bin/env python3

"""
Module description
"""

from data_loader import DataLoader
from blocking import X2Blocker, Blocker
from config import DATA_PATH

import pandas as pd
import os


def main(dataset_name):
    """
    It performs the basic logic pipeline for getting hte data, creates blocking and clustering
    and stores the matching pairs.

    :param dataset_name: str, the name of the input dataset
    """
    # load data
    dl = DataLoader()
    data = dl.load_data(dataset_name)

    # blocking
    if dataset_name == 'X2':  # Instantiate correct blocker based on dataset
        blocker = X2Blocker()
    else:
        blocker = Blocker()
    blocker.fit(data=data)
    # blocks is of type [[instance_id]]
    # list of lists of instance_ids belonging to same group
    # transform returns modified data frame for further use
    blocks, data = blocker.transform()

    pairs = list()
    for block in blocks:
        # apply clustering for each block to get matching pairs
        print(len(block))

    # write pairs
    pairs_df = pd.DataFrame(data=pairs, columns=['left_instance_id', 'right_instance_id'])
    pairs_df.to_csv(os.path.join(DATA_PATH, 'results', 'results_{}.csv'.format(dataset_name)))


if __name__ == '__main__':
    main(dataset_name='X2')
