#!/usr/bin/env python3

"""
Module description
"""
from config import DATA_PATH

import pandas as pd
import os


class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, dataset_name):
        """
        It loads the dataset based on the input dataset name.

        :param dataset_name: str, dataset name
        :return: pd.DataFrame, with the dataset
        """
        data = pd.read_csv(os.path.join(DATA_PATH, '{}.csv'.format(dataset_name)))
        self.data = data

        return data
