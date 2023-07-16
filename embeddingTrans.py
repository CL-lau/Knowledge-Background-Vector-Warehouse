# -*- coding:utf-8 -*-
import random

import openai
# import torch
# import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import tensorflow_hub as hub
import jieba
import numpy as np
import tensorflow_hub as hub
import nltk
import json
import os
import pandas as pd
import PyPDF2
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
from cnocr import CnOcr
from transformers import pipeline
from textSplit import textSplit
from utils import *

"""
功能介绍： 对输入文本进行编码。 知识背景的默认存储是csv。
分支功能：
    1 对json格式的文本进行编码。（指明需要进行编码的内容。可以是list，但是要求在一个整体）
    2 对csv格式的内容进行编码。 （单列csv内容。）
    3 对输入的整体文本进行编码。 （整个文本作为一个基础知识背景来进行展示。）
    4 对输入的PDF进行文本的编码。（整个PDF文档作为一个完整的知识背景来进行设置。）
    5 对输入的图片进行编码，以及简要内容的embedding
    6 对QA文件进行编码，其中包括txt，csv，json，pdf，
"""


def jsonFileEmbedding(jsonFile, col):
    # 读取json文件
    assert jsonFile is not None, "name is empty"
    assert str(jsonFile).endswith(".json"), "请确保是json格式文件！"
    # folder_path = "".join(targetFile.split(".")[:-1])
    #
    # # 判断文件夹是否存在，如果不存在就创建
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path, exist_ok=True)

    with open(jsonFile, "r", encoding="utf-8") as f:
        jsonList = json.load(f)
    sentences = jsonEmbedding(jsonList, col)
    return sentences


def jsonEmbedding(jsonList, col):
    # 进行编码的语句。
    sentences = list(map(lambda x: x[col], jsonList))
    return sentences


def csvEmbedding(csvFile, col):
    assert csvFile is not None, "name is empty"
    assert str(csvFile).endswith(".csv"), "请确保输入的是csv文件！"
    sentences = None
    data = pd.read_csv(csvFile)
    sentences = data[col].values
    return sentences


def pdfEmbedding(pdfFile, model, lan, startPage=None, endPage=None):
    assert pdfFile is not None, "name is empty"
    assert str(pdfFile).endswith(".pdf"), "请确保输入的是pdf文件！"
    # 打开PDF文件
    with open(pdfFile, 'rb') as f:
        # 创建PDF资源管理器
        rsrcmgr = PDFResourceManager()
        # 创建一个字符串缓冲区
        output = io.StringIO()
        # 创建PDF设备对象
        device = TextConverter(rsrcmgr, output, codec='utf-8', laparams=LAParams())
        # 创建PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # 读取PDF文件中的每一页
        for page in PDFPage.get_pages(f):
            # 使用解释器对象解析每一页
            interpreter.process_page(page)
        # 获取PDF文件中的文本
        text = output.getvalue()
        # 关闭PDF设备对象和字符串缓冲区
        device.close()
        output.close()
        # 输出PDF文件中的文本
    text = text.replace('\n', '')
    # assert text == "", "读取pdf内容失败！"
    sentences = textSplit(text, model, lan)
    return sentences


def textEmbedding(text, model, lan):
    assert (text is not None), "不能输入空的文本！"
    sentences = textSplit(text, model, lan)
    return sentences


def imageInterpretEmbedding(text, model=None, lan=None):
    """
    功能：根据图片的语义来实现对图片的编码。
    :param text:
    :param model: 变化传参，
    :param lan:
    :return:
    图片释义模型， Salesforce/blip-image-captioning-base nlpconnect/vit-gpt2-image-captioning
    """
    if str(text).startswith("http"):
        if model is None:
            model = "nlpconnect/vit-gpt2-image-captioning"
        image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        image_to_text(text)
        return [text[0]['generated_text']]
    else:
        if model is not None:
            if len(model) == 3:
                model, feature_extractor, tokenizer = model
                ans = interpretImage(text, model, feature_extractor, tokenizer)
            else:
                ans = interpretImage(text)
        else:
            ans = interpretImage(text)
        return [ans]


def imageTextEmbedding(text, model=None, lan=None):
    """
    :param text:
    :param model: 这里的model 代表的是图片文本提取模型
    :param lan:
    :return: 对图片上的文字提取(str)
    这部分的提取部分主要是。（根据图片上的文字，以及对图片的描述来生成图片的embedding）
    1 针对我们的具体任务，我们首先需要对图片上的文字来进行提取。
    2 在这个基础之上为了完成对图片应用更加复杂的描述，增加了对图片的附加信息来进行描述。
    3 综合图片的文字提取以及文字描述的提取来实现对信息的嵌入embedding
    """
    if str(text).startswith("http"):
        # 下载到本地
        path = downloadImage(text, save_dir="./data/index_images/")
        if path == '':
            return None
    else:
        path = text

    model = CnOcr()
    if model is None:
        model = CnOcr()
    result = model.ocr(path)
    sentences = [r['text'] for r in result]
    sentence = ''.join(sentences)
    return [sentence]


def embeddingRecursiveTrans(sentences):
    # 这里展示的是直接对文本切文本进行embedding，这里可以进而加一步分步总结的请求。
    #
    # document ----- document1 ------ summary1
    #    ｜               ｜               ｜
    #  --------------document2----------------------- summary2
    #    ｜               ｜                               ｜
    # ---------------document3------------------------------------------summary3
    #
    sentences = AbbreviateMultiParagraph(sentences, openai_api_key=None)
    return sentences


def QAEmbedding(file_path):
    questions, answers = read_QA_file(file_path)
    return questions, answers


def textEmbedding(input, type, col, lan, imageDesc, model=None):
    if type == "jsonFile":
        assert col is not None, ""
        """
        input : json文件名， col：需要进行编码的健。
        """
        sentences = jsonFileEmbedding(input, col)
    elif type == "json":
        assert col is not None, ""
        """
        input : json文本， col ：需要编码的键。
        """
        sentences = jsonEmbedding(input, col)
    elif type == "csv":
        assert col is not None, ""
        """
        input : 输入的csv文件名， col：需要编码的列。
        """
        sentences = csvEmbedding(input, col)
    elif type == "pdf":
        """
        input:输入的pdf名称， model：进行语义embedding的模型。 lan：文字的类型（chinese， english）
        """
        assert lan is not None, "对于pdf输入必须指定语言类型（chinese，english）。"
        sentences = pdfEmbedding(input, model, lan)
    elif type == "text":
        """
        input:输入的文本， model：进行语义embedding的模型。 lan：文字的类型（chinese， english）
        """
        assert lan is not None, "对于text输入必须指定语言类型（chinese，english）。"
        sentences = textEmbedding(input, model, lan)
    elif type == "imageText":
        # TODO 完成对图片显示内容的重写，如果显示的内容过多。
        """
        input:输入的图片名称可以是网址也可以是本地地址，model：对用的是文本提取模型。
        """
        sentences = imageTextEmbedding(input, model)
        if imageDesc is not None:
            # TODO 判断图片描述单词的数量，在这个基础上对数量进行控制，及重写。
            sentences += imageDesc
    elif type == "imageInterpret":
        # TODO 完成对图片显示内容的重写，如果显示的内容过多。
        """
        input:输入的图片名称可以是网址也可以是本地地址，
        model：对用的是图片解释模型提取模型。可选的（模型名称（hugging face）或者传入模型）
            要是传入模型的话,需要三部分模型[model, feature_extractor, tokenizer]
        """
        sentences = imageInterpretEmbedding(input, model)
        if imageDesc is not None:
            # TODO 判断图片描述单词的数量，在这个基础上对数量进行控制，及重写。
            sentences += imageDesc
    else:
        assert True, '未指定具体的输入类型.'
        return
    return sentences


def embeddingTrans(input, type, targetFile, col=None, lan="chinese", imageDesc=None, model=None, abbreviation=False):
    """
    :param abbreviation: 判断是否需要进行缩写
    :param input:
    :param type: opt(json, csv, pdf, text, imageText, imageInterpret)
    :param targetFile:
    :param col:
    :param lan:
    :param imageDesc:
    :param model:
    :return:
    """
    if model is None:
        module_url = "../../model/universal-sentence-encoder_4"
        model = hub.load(module_url)
    elif isinstance(model, str):
        model = hub.load(model)
    else:
        model = model

    sentences = None


    # 判断分段之后的段是否需要进行缩写（用户控制）
    if abbreviation:
        sentences = AbbreviateMultiParagraph(sentences)

    embeddings = np.array(model(sentences))
    sentences = np.array(sentences)

    if os.path.exists(targetFile):
        new_df = pd.DataFrame({'sentences': sentences.tolist(), 'embeddings': embeddings.tolist()})
        result = random.choice(['A', 'B'])
        if result == 'A':
            df = pd.read_csv(targetFile)
            last_index = df.index[-1]
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(targetFile, index=False)
        else:
            with open(targetFile, mode='a', newline='') as file:
                new_df.to_csv(file, header=file.tell() == 0, index=False)
    else:
        df = pd.DataFrame({'sentences': sentences.tolist(), 'embeddings': embeddings.tolist()})
        df.to_csv(targetFile, index=False)
    return sentences, embeddings

