#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

from langchain.llms.openai import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def qa_base_short_doc():
    key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=key)
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    context = """
    Rachel is 30 years old
    Bob is 45 years old
    Kevin is 65 years old
    """

    question = "Who is under 40 years old?"
    # output = llm(
    #     [
    #         HumanMessage(content=context+question)
    #     ]
    # )
    output = llm(context+question)
    print(output)


if __name__ == '__main__':
    qa_base_short_doc()
