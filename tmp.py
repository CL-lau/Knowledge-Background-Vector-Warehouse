from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["paragraph", "previous_summaries"],
    template="Summarize the following paragraph in one sentence:\n{paragraph}\n\nPrevious summaries:\n{previous_summaries}",
)

from langchain.llms import OpenAI

# Set up your OpenAI API key
import os

os.environ["OPENAI_API_KEY"] = ""

# Initialize the LLM wrapper
llm = OpenAI()

# Loop over your paragraphs
paragraphs = [
    "黑洞是时空的一个区域，其中重力如此强大，以至于任何东西——无论是粒子还是光这样的电磁辐射——都无法逃脱。广义相对论预测，足够紧凑的质量可以使时空变形，形成黑洞。",
    "新冠疫苗是一种用于预防新型冠状病毒感染的生物制品，它可以刺激人体产生免疫力，从而降低感染的风险或减轻病情。目前，世界卫生组织已经批准了六种新冠疫苗，分别是辉瑞-生物技术公司的mRNA疫苗、牛津-阿斯利康公司的",
    "环保是指保护自然环境，防止污染和破坏，维持生态平衡，促进人类和自然和谐发展的一种行为或活动。环保对于人类的生存和发展具有重要意义，因为自然环境是人类赖以生存的基础，也是人类经济社会发展的资源。随着人口增长和工业化进程加快，人类对自然环境造成了越来越大的影响，导致了气候变化、生物多样性丧失、水"]  # A list of paragraphs to summarize
summaries = []  # A list of summaries
for paragraph in paragraphs:
    # Format the prompt with the input variables
    formatted_prompt = prompt.format(paragraph=paragraph, previous_summaries="\n".join(summaries))
    # Call the LLM on the formatted prompt
    summary = llm(formatted_prompt)
    # Append the summary to the list
    summaries.append(summary)

import os

os.environ["OPENAI_API_KEY"] = "你的api key"

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# 初始化 LLM 包装器
llm = OpenAI(temperature=0)

# 定义一个问答的模板，使用少量示例进行学习
template = """使用以下的例子来回答问题：

例子：
问题：北京是中国的什么？
答案：北京是中国的首都。

问题：太阳是什么颜色的？
答案：太阳是黄色的。

问题：{question}
答案：{answer}
"""

# 格式化模板，将 {question} 和 {answer} 作为输入变量
prompt_template = PromptTemplate(template=template, input_variables=["question", "answer"])

# 初始化 LLM 链，传入 LLM 和模板
llm_chain = LLMChain(llm=llm, prompt_template=prompt_template)

# 初始化问答链，传入 LLM 链和 chain_type 为 stuff
qa_chain = load_qa_chain(llm_chain=llm_chain, chain_type="stuff")


# 定义一个函数，用于获取用户输入的问题，并调用问答链得到答案
def ask(question):
    # 调用问答链，传入问题
    answer = qa_chain.run(question=question)
    # 打印答案
    print(answer)


# 开始一个问答循环
while True:
    # 获取用户输入的问题
    question = input("请输入你的问题：")
    # 调用问答函数，得到答案
    ask(question)

import os

os.environ["OPENAI_API_KEY"] = "你的api key"

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# 初始化 LLM 包装器
llm = OpenAI(temperature=0)

# 定义一个问答的模板，使用少量示例进行学习
template = """使用以下的例子来回答问题：

{examples}

问题：{question}
答案：{answer}
"""

# 定义一个列表，用于存储用户输入的例子
examples = []


# 定义一个函数，用于更新模板
def update_template():
    # 将例子列表转换为字符串，每个例子占一行
    examples_str = "\n".join(examples)
    # 格式化模板，将 {examples} 替换为例子字符串
    updated_template = template.replace("{examples}", examples_str)
    # 返回更新后的模板
    return updated_template


# 初始化 LLM 链，传入 LLM 和模板
llm_chain = LLMChain(llm=llm,
                     prompt_template=PromptTemplate(template=update_template(), input_variables=["question", "answer"]))

# 初始化问答链，传入 LLM 链和 chain_type 为 stuff
qa_chain = load_qa_chain(llm_chain=llm_chain, chain_type="stuff")


# 定义一个函数，用于获取用户输入的问题，并调用问答链得到答案
def ask(question):
    # 调用问答链，传入问题
    answer = qa_chain.run(question=question)
    # 打印答案
    print(answer)


# 定义一个函数，用于获取用户输入的例子，并添加或删除到例子列表中
def edit(example):
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


# 开始一个问答循环
while True:
    # 获取用户输入的内容
    content = input("请输入你要做的事情：")
    # 如果内容以 ? 开头，表示是一个问题，就调用问答函数得到答案
    if content.startswith("?"):
        question = content[1:]
        ask(question)
    # 如果内容以 ! 开头，表示是一个例子，就调用编辑函数添加或删除到例子列表中
    elif content.startswith("!"):
        example = content[1:]
        edit(example)
    # 否则，打印错误信息
    else:
        print(f"无效的输入，请以 ? 或 ! 开头。")

import os

os.environ["OPENAI_API_KEY"] = "你的api key"

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

# 初始化 LLM 包装器
llm = OpenAI(temperature=0)

# 初始化文本分割器，设置最大字符数为 1000
text_splitter = CharacterTextSplitter(max_chars=1000)

# 读取要摘要的文档，这里使用了一篇关于太阳的文章
with open("sun.txt") as f:
    sun_text = f.read()

# 使用文本分割器将文档分成多个段落
paragraphs = text_splitter.split_text(sun_text)

# 定义一个摘要的模板，使用少量示例进行学习
template = """摘要：

{summary}

示例：
原文：The core of the Sun is considered to extend from the center to about 0.2 to 0.25 of solar radius. It is the hottest part of the Sun and of the Solar System. It has a density of 150 g/cm3 at the center, and a temperature of 15 million kelvins.
摘要：太阳的核心是太阳和太阳系最热的部分，占据了太阳半径的四分之一左右，中心密度高达150克/立方厘米，温度高达1500万开尔文。

原文：The photosphere is the visible surface of the Sun that we are most familiar with. Since the Sun is a ball of gas, this is not a solid surface but is actually a layer about 100 km thick (very, very thin compared to the 700,000 km radius of the Sun). Conditions at the bottom of the photosphere are approximately 6,000 K and 0.1 MPa (1 bar) and at the top are about 4,400 K and 0.000019 MPa (0.0002 bar).
摘要：光球是我们最熟悉的太阳的可见表面。由于太阳是一团气体，这不是一个固体表面，而是实际上是一个大约100公里厚的层（与太阳的70万公里半径相比非常非常薄）。光球底部的条件大约是6000 K和0.1 MPa（1 bar），顶部大约是4400 K和0.000019 MPa（0.0002 bar）。

原文：{text}
摘要：{summary}
"""

# 格式化模板，将 {text} 和 {summary} 作为输入变量
prompt_template = PromptTemplate(template=template, input_variables=["text", "summary"])

# 初始化 LLM 链，传入 LLM 和模板
llm_chain = LLMChain(llm=llm, prompt_template=prompt_template)

# 初始化 map_reduce 链，传入 LLM 链和段落列表
map_reduce_chain = MapReduceChain(llm_chain=llm_chain, documents=paragraphs)

# 运行 map_reduce 链，得到每个段落的摘要列表
summaries = map_reduce_chain.run()


# 定义一个函数，用于截断历史信息，保证输入不超过 4k token
def truncate_history(history):
    # 将历史信息转换为 token 列表
    tokens = llm.tokenizer(history, return_tensors="pt")["input_ids"].tolist()[0]
    # 如果 token 数量超过 4k，就从头部删除多余的 token
    if len(tokens) > 4000:
        tokens = tokens[-4000:]
    # 将 token 列表转换回文本
    truncated_history = llm.tokenizer.decode(tokens)
    return truncated_history


# 初始化一个空字符串，用于存储历史信息
history = ""

# 遍历每个段落和对应的摘要
for paragraph, summary in zip(paragraphs, summaries):
    # 在历史信息中添加当前段落和摘要
    history += f"原文：{paragraph}\n摘要：{summary}\n\n"
    # 截断历史信息，保证不超过 4k token
    history = truncate_history(history)
    # 打印当前段落和摘要
    print(f"原文：{paragraph}\n摘要：{summary}\n")

1 / 0
# # To build a personal assistant that can interact with the outside world based on your data
# from langchain.llms import OpenAI
# from langchain.agents import OpenAPIAgent
#
# # Set up your OpenAI API key
# import os
# os.environ["OPENAI_API_KEY"] = "..."
#
# # Initialize the LLM wrapper
# llm = OpenAI()
#
# # Initialize the OpenAPI agent with the OpenAI spec
# agent = OpenAPIAgent("https://api.openai.com/v1/openapi.json")
#
# # Define a function that takes user input and returns a response
# def chat(user_input):
#     # Call the LLM on the user input
#     llm_output = llm(user_input)
#     # Check if the LLM output contains an API request
#     if agent.is_api_request(llm_output):
#         # Parse the API request from the LLM output
#         api_request = agent.parse_api_request(llm_output)
#         # Execute the API request and get the response
#         api_response = agent.execute_api_request(api_request)
#         # Return the API response as the chat response
#         return api_response
#     else:
#         # Return the LLM output as the chat response
#         return llm_output
#
# # Start a chat loop
# while True:
#     # Get user input
#     user_input = input("User: ")
#     # Get chat response
#     chat_response = chat(user_input)
#     # Print chat response
#     print("Chatbot:", chat_response)


# To summarize a paragraph in one sentence using few-shot learning
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Set up your OpenAI API key
import os

os.environ["OPENAI_API_KEY"] = "..."

# Define the prompt template with some examples
prompt = PromptTemplate(
    input_variables=["paragraph"],
    template="""Summarize the following paragraph in one sentence:

{paragraph}

Examples:
Paragraph: The core of the Sun is considered to extend from the center to about 0.2 to 0.25 of solar radius. It is the hottest part of the Sun and of the Solar System. It has a density of 150 g/cm3 at the center, and a temperature of 15 million kelvins.
Summary: The core of the Sun is a very hot and dense region that occupies about a quarter of its radius.

Paragraph: A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.
Summary: A black hole is a place where nothing can escape from its extreme gravity.

Paragraph: Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's metabolic activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water – hence the name photosynthesis, from the Greek phōs (φῶς), \"light\", and sunthesis (σύνθεσις), \"putting together\".
Summary: Photosynthesis is a process where light energy is used to make sugars from carbon dioxide and water.""",
)

# Initialize the LLM wrapper
llm = OpenAI()

# Define a paragraph to summarize
paragraph = "The Taj Mahal is an ivory-white marble mausoleum on the southern bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall."

# Format the prompt with the input variable
formatted_prompt = prompt.format(paragraph=paragraph)

# Call the LLM on the formatted prompt
summary = llm(formatted_prompt)

# Print the summary
print(summary)

# # 使用 LangChain 从 PDF 文件中提取文本，并进行摘要1:
# import os
# os.environ["OPENAI_API_KEY"] = "你的api key"
#
# from langchain.llms import OpenAI
# from langchain.indexes import Index
# from langchain.indexes.document_loaders import PyPDFLoader
#
# # 初始化 LLM 包装器
# llm = OpenAI()
#
# # 初始化 PDF 加载器，传入文件的路径
# loader = PyPDFLoader("path/to/file.pdf")
#
# # 初始化索引，传入加载器和 LLM
# index = Index(loader, llm)
#
# # 从索引中获取文档对象
# document = index.get_document()
#
# # 打印文档的标题和内容
# print(document.title)
# print(document.text)
#
# # 定义一个摘要的模板，使用少量示例进行学习
# template = """摘要：
#
# {summary}
#
# 示例：
# 原文：The core of the Sun is considered to extend from the center to about 0.2 to 0.25 of solar radius. It is the hottest part of the Sun and of the Solar System. It has a density of 150 g/cm3 at the center, and a temperature of 15 million kelvins.
# 摘要：太阳的核心是太阳和太阳系最热的部分，占据了太阳半径的四分之一左右，中心密度高达150克/立方厘米，温度高达1500万开尔文。
#
# 原文：A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.
# 摘要：黑洞是时空的一个区域，其中重力如此强大，以至于任何东西——无论是粒子还是光这样的电磁辐射——都无法逃脱。广义相对论预测，足够紧凑的质量可以使时空变形，形成黑洞。
#
# 原文：Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's metabolic activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water – hence the name photosynthesis, from the Greek phōs (φῶς), \"light\", and sunthesis (σύνθεσις), \"putting together\".
# 摘要：光合作用是植物和其他生物利用光能转化为化学能的过程，这种化学能可以通过细胞呼吸释放出来，为生物的新陈代谢提供动力。这种化学能储存在碳水化合物分子中，比如糖，它们是由二氧化碳和水合成的，因此得名光合作用，源自希腊语 phōs (φῶς)，\"光\"，和 sunthesis (σύνθεσις)，\"组合\"。
# """
#
# # 格式化模板，将文档的内容作为原文
# formatted_template = template.format(summary="{summary}", text=document.text)
#
# # 调用 LLM 并传入格式化后的模板
# summary = llm(formatted_template)
#
# # 打印摘要
# print(summary)
