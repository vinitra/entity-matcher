#!/usr/bin/env python3

"""
Module description
"""
from src.data_loader import DataLoader
from src.blocking import Blocking
from src import DATA_PATH

import pandas as pd
import os


def main(dataset_name):
    """
    It performs the basic logic pipeline for getting hte data, creates blocking and clustering
    and stores the matching pairs.

    :param dataset_name: str, the name of the inptu dataset
    """
    # load data
    dl = DataLoader()
    data = dl.load_data(dataset_name)

    # blocking
    blocking = Blocking()
    blocking.fit(data=data)
    blocks = blocking.transform(data=data)

    pairs = []
    for block in blocks:
        # apply clustering for each block to get matching pairs
        break

    # write pairs
    pairs_df = pd.DataFrame(data=pairs, columns=['left_instance_id', 'right_instance_id'])
    pairs_df.to_csv(os.path.join(DATA_PATH, 'results', 'results_{}.csv'.format(dataset_name)))


if __name__ == '__main__':
    main(dataset_name='X2')





