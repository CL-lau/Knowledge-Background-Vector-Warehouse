from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
import numpy as np
import ast
import os
import json
import csv
import PyPDF2
import requests
from bs4 import BeautifulSoup
from recommender.Recommend import AnnoyRecommender, FaissRecommender, BallTreeRecommender


# 图像内容提取模块
def interpretImage(image_paths, model=None, feature_extractor=None, tokenizer=None):
    # 加载模型
    if type(model) == str:
        model = VisionEncoderDecoderModel.from_pretrained(model)
    if type(feature_extractor) == str:
        feature_extractor = ViTImageProcessor.from_pretrained(feature_extractor)
    if type(tokenizer) == str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if model is None:
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    if feature_extractor is None:
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    ans = predict_step(image_paths, model, feature_extractor, tokenizer, gen_kwargs)
    return ans[0]


def predict_step(image_paths, model, feature_extractor, tokenizer, gen_kwargs):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


# 保存图片
def downloadImage(url, save_dir):
    # 发送请求，获取图片数据
    response = requests.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        # 获取文件名
        filename = url.split('/')[-1]

        # 拼接保存路径
        save_path = os.path.join(save_dir, filename)

        # 判断保存路径是否存在，如果不存在则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存图片到本地
        with open(save_path, 'wb') as f:
            f.write(response.content)

        # 返回保存路径
        return save_path
    else:
        # 请求失败，返回空字符串
        return ''


# 对多段语句进行分段总结
def AbbreviateMultiParagraph(sentences, openai_api_key=None, openai_api_url=None):
    prompt = PromptTemplate(
        input_variables=["paragraph", "previous_summaries"],
        template="Summarize the following paragraph in one sentence:\n{paragraph}\n\n"
                 "Previous summaries:\n{previous_summaries}",
    )
    if openai_api_key is None:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key is None:
        openai_api_key = ""

    if openai_api_url is None:
        openai_api_url = os.environ.get('OPENAI_API_URL')
    if openai_api_url is None:
        openai_api_url = ""

    # Initialize the LLM wrapper
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key,
                 openai_api_base=openai_api_url)

    # Loop over your paragraphs
    if sentences is None:
        paragraphs = [
            "黑洞是时空的一个区域，其中重力如此强大，以至于任何东西——无论是粒子还是光这样的电磁辐射——都无法逃脱。广义相对论预测，足够紧凑的质量可以使时空变形，形成黑洞。",
            "新冠疫苗是一种用于预防新型冠状病毒感染的生物制品，它可以刺激人体产生免疫力，从而降低感染的风险或减轻病情。",
            "环保是指保护自然环境，防止污染和破坏，维持生态平衡，促进人类和自然和谐发展的一种行为或活动。"]
        # A list of paragraphs to summarize
    else:
        paragraphs = sentences
    summaries = []  # A list of summaries
    for paragraph in paragraphs:
        # Format the prompt with the input variables
        formatted_prompt = prompt.format(paragraph=paragraph, previous_summaries="\n".join(summaries))
        # Call the LLM on the formatted prompt
        summary = llm(formatted_prompt)
        # Append the summary to the list
        summaries.append(summary)
    return summaries


# 样例学习
def update_template(examples, template):
    # 将例子列表转换为字符串，每个例子占一行
    examples_str = "\n".join(examples)
    # 格式化模板，将 {examples} 替换为例子字符串
    updated_template = template.replace("{examples}", examples_str)
    # 返回更新后的模板
    return updated_template


# 定义一个函数，用于获取用户输入的问题，并调用问答链得到答案
def ask(qa_chain, question):
    # 调用问答链，传入问题
    answer = qa_chain.run(question=question)
    return answer


# 定义一个函数，用于获取用户输入的例子，并添加或删除到例子列表中
def edit(example, examples):
    # 如果例子以 + 开头，表示要添加到列表中
    if example.startswith("+"):
        # 去掉 + 号，得到真正的例子内容
        example = example[1:]
        # 如果例子不在列表中，就添加到列表中，并打印提示信息
        if example not in examples:
            examples.append(example)
            print(f"已添加 {example} 到例子列表中。")
        # 否则，打印错误信息
        else:
            print(f"{example} 已经在例子列表中了。")
    # 如果例子以 - 开头，表示要从列表中删除
    elif example.startswith("-"):
        # 去掉 - 号，得到真正的例子内容
        example = example[1:]
        # 如果例子在列表中，就从列表中删除，并打印提示信息
        if example in examples:
            examples.remove(example)
            print(f"已删除 {example} 从例子列表中。")
        # 否则，打印错误信息
        else:
            print(f"{example} 不在例子列表中。")
    # 否则，打印错误信息
    else:
        print(f"无效的输入，请以 + 或 - 开头。")
    return examples


def exampleStudy(examples, question, openai_api_key=None):
    if openai_api_key is None:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key is None:
        openai_api_key = ""

    # Initialize the LLM wrapper
    llm = OpenAI(temperature=.5, openai_api_key=openai_api_key,
                 openai_api_base="")

    # 定义一个问答的模板，使用少量示例进行学习
    template = """根据以下的例子来学习规律，进而来回答问题：

    {examples}

    问题：{question}
    答案：{answer}
    """

    template = update_template(examples, template)
    # 初始化 LLM 链，传入 LLM 和模板
    llm_chain = LLMChain(llm=llm, prompt_template=PromptTemplate(template=template,
                                                                 input_variables=["question", "answer"]))

    # 初始化问答链，传入 LLM 链和 chain_type 为 stuff
    qa_chain = load_qa_chain(llm_chain=llm_chain, chain_type="stuff")
    answer = ask(qa_chain, question)
    return answer


# 根据推荐器名称以及文件列表来生成推荐器
def createRecommenderImpl(recommenderName, dataFileList):
    data = pd.DataFrame()

    for file in dataFileList:
        temp_df = pd.read_csv(file)
        df = pd.concat([data, temp_df], ignore_index=True)
    sentences = data.sentences.values
    embeddings = np.array([np.array(ast.literal_eval(val)) for val in data['embeddings']])

    if recommenderName == "annoy":
        recommenderImpl = AnnoyRecommender(embeddings=embeddings, sentences=sentences)
    elif recommenderName == "faiss":
        recommenderImpl = FaissRecommender(embeddings=embeddings, sentences=sentences)
    elif recommenderName == "ballTree":
        recommenderImpl = BallTreeRecommender(embeddings=embeddings, sentences=sentences)
    else:
        recommenderImpl = FaissRecommender(embeddings=embeddings, sentences=sentences)

    return recommenderImpl


def read_QA_file(file_path):
    questions = []
    answers = []

    file_extension = os.path.splitext(file_path)[1]
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                questions.append(lines[i].strip())
                answers.append(lines[i+1].strip())
                if i+2 < len(lines) and lines[i+2].strip() == '':
                    i += 1

    elif file_extension == '.pdf':
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            for i in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(i)
                text = page.extractText()
                lines = text.split('\n')
                for j in range(0, len(lines), 2):
                    questions.append(lines[j].strip())
                    answers.append(lines[j+1].strip())
                    if j+2 < len(lines) and lines[j+2].strip() == '':
                        j += 1

    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                questions.append(item['question'])
                answers.append(item['answer'])

    elif file_extension == '.csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                questions.append(row[0])
                answers.append(row[1])

    elif file_extension == '.html' or file_extension == '.htm' \
            or str(file_path).startswith('http') or str(file_path).startswith('https'):
        response = requests.get(file_path)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            questions.append(tag.text.strip())
            next_tag = tag.find_next_sibling()
            if next_tag:
                answers.append(next_tag.text.strip())
            else:
                answers.append('')

    return questions, answers

