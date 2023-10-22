#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time 2023/10/22 S{TIME} 
# @Name Chatbots. Py
# @Author：jialtang
import os

from langchain.llms import OpenAI
from langchain import LLMChat
from langchain.prompts.prompt import PromptTemplate

# Chat specific components
from langchain.memory import ConversationBufferMemory

template = """
You are a chatbot that is unhelpful.
Your goal is to not help the user but only make jokes.
Take what the user is saying and make a joke out of it

{chat_history}
Human: {human_input}
Chatbot:
"""


def chatbox():
    global predict
    openai_api_key = os.environ["OPENAI_API_KEY"]
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template,
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chat = LLMChat(llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, verbose=True, memory=memory, )
    predict = llm_chat.predict(human_input="Is an pear a fruit or vegetable?")
    print(predict)


if __name__ == "__main__":
    chatbox()
