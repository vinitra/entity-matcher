#!/usr/bin/env python3

from config import DATA_PATH, ROOT_DIR

import tensorflow_hub as hub
from sklearn.cluster import KMeans
import pandas as pd
import os


class Clustering:
    def __init__(self, cluster_n):
        """
        Responsible for the clustering process of ER.

        :param cluster_n: int, the number of clusters to produce for each block
        """
        self.cluster_n = cluster_n
        self.model = self.__import_sentence_encoder()

    def run(self, data):
        """
        Performs clustering to the input block data.

        :param data: pd.DataFrame, subset of the data which belong to a specific block
        :return: list, of lists with the clusters
        """
        # title encoding
        sentences = data.title
        sentence_embeddings = self.model(sentences)
        embeddings_array = sentence_embeddings.numpy()

        # instantiate and fit clustering model
        kmeans = KMeans(n_clusters=self.cluster_n).fit(embeddings_array)

        # get cluster assignments
        clustering_res = pd.DataFrame()
        clustering_res['instance_id'] = data.instance_id
        clustering_res['cluster_labels'] = kmeans.labels_
        clustering_res_l = clustering_res.groupby('cluster_labels', as_index=False).agg(list)

        return clustering_res_l['instance_id'].tolist()

    @staticmethod
    def __import_sentence_encoder():
        """
        Loads universal sentence encoder
        :return: tf.model
        """
        model_path = os.path.join(ROOT_DIR, 'models', 'universal-sentence-encoder_4')
        model = hub.load(model_path)
        print("Model %s loaded" % model_path)

        return model
