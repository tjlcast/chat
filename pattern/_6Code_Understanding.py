#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time 2023/10/22 S{TIME} 
# @Name Code_Understanding. Py
# @Author：jialtang

# Helper to read local files
import csv
import os

# Vector support
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# model and chain
from langchain.chat_models import ChatOpenAI

# Text splitters
from langchain.document_loaders import TextLoader

from langchain.chains import RetrievalQA
from thefuzz import process

root_dir = '/Users/jialtang/Code/chat/'


def code_understanding():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key)

    docs = []

    # Go through each folder
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 含义，dirpath为当前目录的父目录，dirnames为当前目录下类型为'目录'的名字，filenames为当前目录下类型为'文件'的名字.
        # Go through each file
        for file in filenames:
            try:
                # Load up the file as a doc and split
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(e)

    print(f"You have {len(docs)} documents\n")
    print("----- Start Document -----")
    print(docs[0].page_content[:300])

    iterations = 100000
    reader = csv.DictReader(open('data/titledata.csv'), delimiter='|')
    titles = [i['custom_title'] for i in reader]
    title_blob = '\n'.join(titles)

    dosearch = FAISS.from_documents(docs, embeddings)

    # Get our retriever ready
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=dosearch.as_retriever())

    query = "What function do I use if I want to find the most similar item in a list of items?"
    output = qa.run(query)
    print(output)

    choices = ["New York Yankees",
               "Boston Red Sox",
               "Chicago Cubs",
               "Los Angeles Dodgers"]
    query = "new york mets vs atlanta braves"

    best_math = process.extractOne(query, choices)
    print(best_math)

    query = "Can you write the code to use the process.extractOne() function? Only respond with code. No other text or explanation"
    output = qa.run(query)
    print(output)


if __name__ == "__main__":
    code_understanding()
