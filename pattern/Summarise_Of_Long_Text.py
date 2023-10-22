#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


def summarise_of_long_text():
    openai_aip_key = os.environ["OPENAI_API_KEY"]

    llm = OpenAI(temperature=0,
                 model_name='gpt-3.5-turbo',
                 openai_aip_key=openai_aip_key)

    with open('wonderland.txt', 'r') as file:
        text = file.read()

    # 打印小说的前285个字符
    print(text[:258])
    num_tokens = llm.get_num_tokens(text)
    print(f"There are {num_tokens} tokens in your file")
    # 全文一共有4w8词
    # 很明显这样的文本量是无法直接送进LLM进行处理和生成的

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],
                                                   chunk_size=5000,
                                                   chunk_overlap=350)
    # 也可使用其他的工具
    docs = text_splitter.create_documents([text])
    print(f"You now have {len(docs)} docs intend of 1 piece of text")

    # 设置 lang chain
    # 使用 map_reduce 的 chain_type，这样可以将多个文档合并为一个
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)

    # Use it. This will run through the 36 documents, summarize the chunks, then get a summary of summarys.
    # 典型MR的思路去解决问题，将文章拆分为多个部分，再将多个部分分别进行 summarize, 最后在进行 合并，对 summarys 进行 summary
    output = chain.run(docs)
    print(output)


if __name__ == '__main__':
    summarise_of_long_text()
