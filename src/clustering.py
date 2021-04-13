#!/usr/bin/env python3

from config import DATA_PATH, ROOT_DIR
from similarity import calc_jac_titles

import tensorflow_hub as hub
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        self.distance_threshold = kwargs.get("distance_threshold", 0)
        self.encoding = kwargs.get('encoding', 'use')

        if self.encoding == 'use':
            self.use_model = self.__import_universal_sentence_encoder()
        # elif self.encoding == 'bert':
        #     self.bert_model = self.__import_bert_sentence_encoder()

        self.method = kwargs.get('method', 'kmeans')
        self.cluster_n = kwargs.get('cluster_n', 0)
        self.threshold = 0.75

    def run(self, data):
        """
        Performs clustering to the input block data.

        :param data: pd.DataFrame, subset of the data which belong to a specific block
        :return: list, of lists with the clusters
        """
        if self.method == 'kmeans':
            return self.__run_kmeans(data)

        elif self.method == 'agglomerative':
            return self.__run_agglomerative_clustering(data)

        elif self.method == 'birch':
            return self.__run_birch(data)

        elif self.method == 'cosine':
            return self.__run_cosine(data)

        elif self.method == 'jaccard':
            return self.__run_jaccard(data)

        else:
            raise ValueError("Please set a valid clustering method between: Kmeans, Jaccard and cosine similarity")

    def __get_embeddings(self, data):
        if self.encoding == 'use':
            # Universal Sentence Encoder Model
            sentences = data.title
            sentence_embeddings = self.use_model(sentences)
            embeddings_array = sentence_embeddings.numpy()
            return embeddings_array

        # elif self.encoding == 'bert':
        #     # sentence BERT encoder model
        #     sentences = data.title.tolist()
        #     embeddings_array = self.bert_model.encode(sentences)
        #     return embeddings_array

    def __run_cosine(self, data):
        """

        :param data:
        :return:
        """
        # title encoding
        embeddings_array = self.__get_embeddings(data)

        # calculate cosine similarity score among data samples
        cosine_matrix = cosine_similarity(embeddings_array)
        data_samples = data.instance_id.tolist()
        cosine_df = pd.DataFrame(cosine_matrix)

        pairs = list()
        for idx, row in cosine_df.iterrows():
            for col in range(len(data_samples)):
                # if cosine score above threshold and the score is not referred to the diagonal
                if row[col] > self.threshold and col != idx:
                    pair = [data_samples[idx], data_samples[col]]
                    pairs.append(pair)

        return pairs

    def __run_jaccard(self, data_cp):
        # tokenize title
        data = data_cp.copy()
        data['tokenized_title'] = data['title'].apply(lambda x: x.replace("/", " ").split())

        # calculate jaccard similarity score among data samples
        pairs = list()
        for idx1, row1 in data.iterrows():
            for idx2, row2 in data.iterrows():
                score = 0
                # if cosine score above threshold and the score is not referred to the diagonal
                if idx1 != idx2:
                    tokenized_title1 = row1['tokenized_title']
                    tokenized_title2 = row2['tokenized_title']
                    if len(tokenized_title1) > len(tokenized_title2):
                        score = calc_jac_titles(tokenized_title1, tokenized_title2)
                    else:
                        score = calc_jac_titles(tokenized_title2, tokenized_title1)

                if score > self.threshold:
                    pair = [row1['instance_id'], row2['instance_id']]
                    pairs.append(pair)
        return pairs

    def __run_kmeans(self, data):
        """
        Runs kmeans clustering for the input data and returns the list of clusters
        """
        if len(data) >= self.cluster_n:
            # title encoding
            embeddings_array = self.__get_embeddings(data)

            # instantiate and fit clustering model
            kmeans = KMeans(n_clusters=self.cluster_n).fit(embeddings_array)

            # get cluster assignments
            clustering_res = pd.DataFrame()
            clustering_res['instance_id'] = data.instance_id
            clustering_res['cluster_labels'] = kmeans.labels_
            clustering_res_l = clustering_res.groupby('cluster_labels', as_index=False).agg(list)

            return clustering_res_l['instance_id'].tolist()

        # if less samples that clusters, return one cluster with all the samples
        else:
            return data['instance_id'].tolist()

    def __run_agglomerative_clustering(self, data):
        """
        Runs agglomerative clustering for the input data and returns the list of clusters
        """
        # title encoding
        embeddings_array = self.__get_embeddings(data)

        # instantiate and fit clustering model
        agg_clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold).fit(embeddings_array)

        # get cluster assignments
        clustering_res = pd.DataFrame()
        clustering_res['instance_id'] = data.instance_id
        clustering_res['cluster_labels'] = agg_clusters.labels_
        clustering_res_l = clustering_res.groupby('cluster_labels', as_index=False).agg(list)

        return clustering_res_l['instance_id'].tolist()


    def __run_birch(self, data):
        """
        Runs birch for the input data and returns the list of clusters
        """
        if len(data) >= self.cluster_n:
            # title encoding
            embeddings_array = self.__get_embeddings(data)

            # instantiate and fit clustering model
            birch = Birch(threshold=0.2, n_clusters=self.cluster_n).fit(embeddings_array)

            # get cluster assignments
            clustering_res = pd.DataFrame()
            clustering_res['instance_id'] = data.instance_id
            clustering_res['cluster_labels'] = birch.labels_
            clustering_res_l = clustering_res.groupby('cluster_labels', as_index=False).agg(list)

            return clustering_res_l['instance_id'].tolist()

        # if less samples that clusters, return one cluster with all the samples
        else:
            return data['instance_id'].tolist()


    @staticmethod
    def __import_universal_sentence_encoder(verbose=False):
        """
        Loads universal sentence encoder
        :return: tf.model
        """
        model_path = os.path.join(ROOT_DIR, 'models', 'universal-sentence-encoder_4')
        if not os.path.exists(model_path):
            model_path = "https://tfhub.dev/google/universal-sentence-encoder/4"

        model = hub.load(model_path)
        print("Model %s loaded" % model_path) if verbose else ''

        return model

    # @staticmethod
    # def __import_bert_sentence_encoder(verbose=False):
    #     """
    #     Loads SentenceBERT encoder
    #     :return: tf.model
    #     """
    #     model = SentenceTransformer('bert-base-nli-mean-tokens')
    #     print("Model %s loaded" % model) if verbose else ''
    #
    #     return model
