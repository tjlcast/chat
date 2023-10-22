#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os

from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


def qa_base_long_doc():
    print("qa_base_long_doc")

    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    loader = TextLoader('wonderland.txt')
    doc = loader.load()
    print(f"You have {len(doc)} document")
    print(f"You have {len(doc[0].page_content)} characters in that document")

    # 将小说分割成多个部分
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    docs = text_spliter.split_documents(doc)

    # 获取字符的总数，以便可以计算平均值
    num_total_characters = sum([len(x.page_content) for x in docs])
    print(
        f"Now you have {len(docs)} documents that have a average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")

    # 设置 embedding 引擎
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Embed 文档，然后使用伪数据库将文档和原始文本结合起来
    # 同时再向 OpenAI 发起 API 请求
    docsearcher = FAISS.from_documents(docs, embeddings)
    # 创建 QA-retrieval chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearcher.as_retriever())

    query = "What does the author describe the Alice following with?"
    answer = qa.run(query)
    # 这个过程中，检索器会去获取类似的文件部分，并结合你的问题让 LLM 进行推理，最后得到答案
    # 这一步还有很多可以细究的步骤，比如如何选择最佳的分割大小，如何选择最佳的 embedding 引擎， 如何选择最佳的检索器等等
    # 同时也可以选择云端向量存储
    print(f"Finish: {answer}")


if __name__ == '__main__':
    qa_base_long_doc()
