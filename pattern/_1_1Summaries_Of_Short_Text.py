#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

from langchain.llms import OpenAI
from langchain import PromptTemplate


def summaries_of_short_text():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    print(openai_api_key)
    llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)  # 初始化LLM模型

    # 创建模板
    template = """
    %INSTRUCTIONS:
    Please summarize the following piece of text.
    Respond in a manner that a 5 year old would understand.

    %TEXT:
    {text}
    """

    # 创建一个 Lang Chain Prompt 模版， 稍后可以插入值
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    confusing_text = confusing_text = """
    For the next 130 years, debate raged.
    Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
    “The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
    “And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
    """

    print("------- Prompt Begin -------")
    # 打印模板内容
    final_prompt = prompt.format(text=confusing_text)
    print(final_prompt)

    print("------- Prompt End -------")
    output = llm(final_prompt)
    print(output)


if __name__ == '__main__':
    summaries_of_short_text()