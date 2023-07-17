# -*- coding:utf-8 -*-
import openai
import torch
# import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import jieba
import nltk
from annoy import AnnoyIndex
import faiss
from sklearn.neighbors import BallTree


class Recommender(object):
    def __init__(self, embeddings, sentences, metric='euclidean', num_trees=10):
        self.embeddings = embeddings
        self.sentences = sentences

        # 创建annoy搜索器
        self.annoy_index = AnnoyIndex(len(embeddings[0]), metric=metric)
        for i, embedding in enumerate(embeddings):
            self.annoy_index.add_item(i, embedding)
        self.annoy_index.build(num_trees)

        # 创建faiss搜索器
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
        self.faiss_index.add(embeddings)

        # 创建HNSW搜索器
        self.hnsw_index = faiss.IndexHNSWFlat(embeddings.shape[-1], 100)
        self.probe_num = 10

    def annoySearch(self, new_embedding, topK=3):
        topK_indices = self.annoy_index.get_nns_by_vector(new_embedding, topK)
        topK_embeddings = [self.embeddings[i] for i in topK_indices]
        topK_sentences = [self.sentences[i] for i in topK_indices]
        return topK_embeddings, topK_sentences

    def faissSearch(self, new_embedding, topK=3):
        distances, indices = self.faiss_index.search(new_embedding, topK)
        return [self.embeddings[i] for i in indices[0]], [self.sentences[i] for i in indices[0]]
        return indices[0], distances[0], [self.sentences[i] for i in indices[0]]


class AnnoyRecommender(object):
    def __init__(self, embeddings, sentences, metric='euclidean', num_trees=10):
        self.embeddings = embeddings
        self.sentences = sentences
        # 创建annoy搜索器
        self.annoy_index = AnnoyIndex(len(embeddings[0]), metric=metric)
        for i, embedding in enumerate(embeddings):
            self.annoy_index.add_item(i, embedding)
        self.annoy_index.build(num_trees)

    def recommend(self, new_embedding, topK):
        # new_embedding = (512)
        topK_indices = self.annoy_index.get_nns_by_vector(new_embedding, topK)
        topK_embeddings = [self.embeddings[i] for i in topK_indices]
        topK_sentences = [self.sentences[i] for i in topK_indices]
        return topK_embeddings, topK_sentences

    def addEmbedding(self, embedding, sentence):
        index = len(self.sentences)
        self.annoy_index.add_item(index, embedding)
        self.sentences = np.append(self.sentences, sentence)
        self.embeddings = np.vstack((self.embeddings, embedding))


class FaissRecommender(object):
    def __init__(self, embeddings, sentences):
        self.embeddings = embeddings
        self.sentences = sentences

        # 创建faiss搜索器
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[-1])
        self.faiss_index.add(embeddings)

    def recommend(self, new_embedding, topK=3):
        # new_embedding = (1, 512)
        distances, indices = self.faiss_index.search(new_embedding, topK)
        return [self.embeddings[i] for i in indices[0]], [self.sentences[i] for i in indices[0]]
        # return indices[0], distances[0], [self.sentences[i] for i in indices[0]]

    def addEmbedding(self, embedding, sentence):
        index = len(self.sentences)
        self.faiss_index.add(embedding.reshape(-1, self.embeddings.shape[-1]))
        self.sentences = np.append(self.sentences, sentence)
        self.embeddings = np.vstack((self.embeddings, embedding))


class HnswRecommender(object):
    def __init__(self, embeddings, sentences):
        self.embeddings = embeddings
        self.sentences = sentences

        # 创建HNSW搜索器
        self.hnsw_index = faiss.IndexHNSWFlat(embeddings.shape[-1], 100)
        self.probe_num = 10
        self.hnsw_index.train(self.embeddings)
        self.hnsw_index.add(self.embeddings)

    def search_with_text(self, query_embedding, topK, embeddings, texts):
        distances, indices = self.hnsw_index.search(query_embedding, topK)
        result_embeddings = embeddings[indices]
        result_texts = [texts[i] for i in indices]
        return distances, result_embeddings, result_texts

    def recommend(self, new_embedding, topK=3):
        distances, result_embeddings, result_texts = self.search_with_text(new_embedding, topK, self.embeddings,
                                                                           self.sentences)
        return result_embeddings, result_texts


class BallTreeRecommender(object):
    def __init__(self, embeddings, sentences, leaf_size=40):
        self.leaf_size = leaf_size

        # 初始化BallTree索引
        self.index = BallTree(embeddings, leaf_size=self.leaf_size)
        self.sentences = sentences
        self.embeddings = embeddings

    def recommend(self, query_embedding, topK):
        # 搜索相似的向量
        D, I = self.index.query(query_embedding.reshape(1, -1), k=topK)

        # 获取相似向量的embedding和原文
        result_embeddings = [np.frombuffer(self.index.data[i]) for i in I[0]]
        result_texts = [self.sentences[i] for i in I[0]]
        return result_embeddings, result_texts

    def addEmbedding(self, embedding, sentence):
        index = len(self.sentences)
        self.index.insert(embedding.reshape(-1, self.embeddings[-1]))
        self.sentences = np.append(self.sentences, sentence)
        self.embeddings = np.vstack((self.embeddings, embedding))