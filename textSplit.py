# -*- coding:utf-8 -*-
import openai
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import tensorflow_hub as hub
import jieba
import numpy as np
import tensorflow_hub as hub
import nltk


# # 加载Universal Sentence Encoder模型
# module_url = "../../model/universal-sentence-encoder_4"
# model = hub.load(module_url)

def textSplit_Chinese(text, model, similarity_threshold=0.5, window_size=2):
    """
    :param window_size:
    :param similarity_threshold: 需要合并的语句的相似度（两个语句的相似度）
    :param text: 需要进行分句的语句
    :param model: 进行语句embedding的模型
    :return: 分句结果
    """
    words = jieba.lcut(text)

    # 根据标点符号进行分句
    sentences = []
    start = 0
    for i, word in enumerate(words):
        if word in ["。", "！", "？", "；"]:
            sentences.append("".join(words[start:i + 1]))
            start = i + 1

    # 计算每个窗口内语句的相似度，并根据相似度高低来决定是否进行拼接
    window_size = window_size
    similarity_threshold = similarity_threshold
    new_sentences = []
    window_start = 0
    while window_start < len(sentences):
        window_end = min(window_start + window_size, len(sentences))
        window_sentences = sentences[window_start:window_end]
        window_vectors = model(window_sentences)
        similarity_matrix = np.inner(window_vectors, window_vectors)
        for i in range(len(window_sentences)):
            for j in range(i + 1, len(window_sentences)):
                if similarity_matrix[i, j] < similarity_threshold:
                    new_sentences.append("".join(window_sentences[:j]))
                    window_start += j
                    break
            else:
                continue
            break
        else:
            new_sentences.append("".join(window_sentences))
            window_start += window_size

    # 输出拼接后的语句
    for i, sentence in enumerate(new_sentences):
        print("Sentence {}: {}".format(i, sentence))

    return new_sentences


def textSplit_English(text, model, similarity_threshold=0.5, window_size=2):
    """
    :param window_size:
    :param similarity_threshold:
    :param text: 要进行分词的英文语句
    :param model: 用于生成语义embedding的模型
    :return: 分句结果
    """
    # 使用nltk库中的sent_tokenize函数对文本进行分句
    sentences = nltk.sent_tokenize(text)

    # 计算每个窗口内语句的相似度，并根据相似度高低来决定是否进行拼接
    window_size = window_size
    similarity_threshold = similarity_threshold
    new_sentences = []
    window_start = 0
    while window_start < len(sentences):
        window_end = min(window_start + window_size, len(sentences))
        window_sentences = sentences[window_start:window_end]
        window_vectors = model(window_sentences)
        similarity_matrix = np.inner(window_vectors, window_vectors)
        for i in range(len(window_sentences)):
            for j in range(i + 1, len(window_sentences)):
                if similarity_matrix[i, j] < similarity_threshold:
                    new_sentences.append(" ".join(window_sentences[:j]))
                    window_start += j
                    break
            else:
                continue
            break
        else:
            new_sentences.append(" ".join(window_sentences))
            window_start += window_size

    # 输出拼接后的语句
    for i, sentence in enumerate(new_sentences):
        print("Sentence {}: {}".format(i, sentence))

    return new_sentences


# 分词的主方法
def textSplit(text, model, lan, similarity_threshold=0.3, window_size=4):
    """
    :param text:
    :param model:
    :param lan: 表示什么语言 option：chinese/english
    :param similarity_threshold:
    :param window_size:
    :return:
    """
    if lan == "chinese":
        sentences = textSplit_Chinese(text, model, similarity_threshold, window_size)
    elif lan == "english":
        sentences = textSplit_English(text, model, similarity_threshold, window_size)
    else:
        sentences = ""
        print("现在只支持对中文以及英文进行分词！")
    return sentences

