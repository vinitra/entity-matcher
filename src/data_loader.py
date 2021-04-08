#!/usr/bin/env python3

from config import DATA_PATH

import pandas as pd
import os


class DataLoader:
    def __init__(self):
        self.data = None
        self.pairs = None

    def load_data(self, dataset_id):
        """
        It loads the dataset based on the input dataset name.

        :param dataset_id: int, dataset id
        :return: pd.DataFrame, with the dataset
        """
        data = pd.read_csv(os.path.join(DATA_PATH, 'X{}.csv'.format(dataset_id)))
        pairs_raw = pd.read_csv(os.path.join(DATA_PATH, 'Y{}.csv'.format(dataset_id)))

        # keep only relevant column
        if dataset_id == 4:
            data["title"] = data["name"]

        self.data = data
        self.pairs = pairs_raw

        return data, pairs_raw
