#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time 2023/10/22 S{TIME} 
# @Name _9Agent. Py
# @Author：jialtang

# Helpers
import os
import json

# Agent imports
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# Tool imports
from langchain.agents import Tool
from langchain.llms.openai import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import TextRequestsWrapper

os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID"
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
openai_api_key = os.environ["OPENAI_API_KEY"]


def chatbox():
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    searcher = GoogleSearchAPIWrapper()
    requests = TextRequestsWrapper()
    toolkit = [
        Tool(
            name="Search",
            func=searcher.run,
            description="Useful for when you need to search google to answer questions about current events"
        ),
        Tool(
            name="Requests",
            func=requests.get,
            description="Useful for when you to make a request to a URL"
        )
    ]
    agent = initialize_agent(tools=toolkit, llm=llm, agent="zero-shot-react-description", verbose=True,
                             return_intermediate_steps=True)
    response = agent({"input": "What is the capital of canada?"})
    response['output']


if __name__ == "__main__":
    chatbox()
