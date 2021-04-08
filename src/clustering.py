#!/usr/bin/env python3

from config import DATA_PATH, ROOT_DIR

import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import pandas as pd
import os


class Clustering:
    def __init__(self, **kwargs):
        """
        Responsible for the clustering process of ER.

        :param cluster_n: int, the number of clusters to produce for each block
        """
        self.method = kwargs.get("clustering_method", 'kmeans')
        self.cluster_n = kwargs.get("cluster_n", 0)
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
        if not os.path.exists(model_path):
            model_path = "https://tfhub.dev/google/universal-sentence-encoder/4"

        model = hub.load(model_path)
        print("Model %s loaded" % model_path)

        return model
