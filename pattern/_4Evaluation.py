#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time 2023/10/22 S{TIME} 
# @Name Evaluation. Py
# @Author：jialtang
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import QAEvalChain
from langchain.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def evaluation():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    loader = TextLoader("wonderland.txt")
    text = loader.load()

    print(f"You have {len(text)} documents")
    print(f"You have {len(text[0].page_content)} in that document")

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    documents = splitter.split_documents(text)

    num_total_characters = sum([len(x.page_content) for x in documents])
    print(
        f"Now you have {len(documents)} documents that have an average of {num_total_characters / len(documents)} characters (smaller pieces)")

    # Embedding and docstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = FAISS.from_documents(documents, embeddings)

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                        input_key="question")
    # 注意这里的 input_key 参数，这个参数告诉了 chain 我的问题在字典中的哪个 key 里
    # 这样 chain 就会自动去找到问题并将其传递给 LLM

    question_answers = [
        {'question': "Which animal give alice a instruction?", 'answer': 'rabbit'},
        {'question': "What is the author of the book", 'answer': 'Elon Mask'}
    ]
    predictions = chain.apply(question_answers)
    print(f"predictions:{predictions}")
    # 使用LLM模型进行预测，并将答案与我提供的答案进行比较，这里信任我自己提供的人工答案是正确的

    # Start your eval chain
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(question_answers,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result",
                                         answer_key='answer')
    print(graded_outputs)


if __name__ == "__main__":
    evaluation()
