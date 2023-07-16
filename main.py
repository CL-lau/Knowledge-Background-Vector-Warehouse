import numpy as np
import tensorflow_hub as hub
import pandas as pd

from embeddingTrans import embeddingTrans
from recommender.Recommend import FaissRecommender, AnnoyRecommender, HnswRecommender, BallTreeRecommender
import ast

from recommender.Recommender import Recommender


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    module_url = "../../model/universal-sentence-encoder_4"
    model = hub.load(module_url)
    while True:
        embeddingType = input("输入查询或新增：")
        if embeddingType == "search":
            data = pd.read_csv("./test/data.csv")
            sentences = data.sentences.values
            embeddings = [np.array(ast.literal_eval(val)[:-1]) for val in data['embeddings']]
            embeddings = np.array([np.array(ast.literal_eval(val)) for val in data['embeddings']])
            faissRecommender = FaissRecommender(embeddings=embeddings, sentences=sentences) #（1，512）
            annoyRecommender = AnnoyRecommender(embeddings=embeddings, sentences=sentences) # （512）
            ballTreeRecommender = BallTreeRecommender(embeddings, sentences)
            recommender = Recommender(ballTreeRecommender)
            text = input("输入查询的内容：")
            new_embedding = model([text])
            new_embeddings, new_sentences = recommender.recommend(np.array(new_embedding[0]), 3)
            print(new_embeddings)
            for item in new_embeddings:
                print(item)
            print(new_sentences)
        elif embeddingType == "新增":
            fileType = input("请输入文件类型：")
            fileName = input("输入文件的名称：")
            embeddingTrans(fileName, fileType, targetFile="./test/data.csv", model=model)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
