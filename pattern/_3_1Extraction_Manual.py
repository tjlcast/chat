#!/usr/bin/python
# -*- coding -*-
import os

from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI


def extraction_manual():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # Vanilla Extraction
    instructions = '''
    You willbe given a sentence with fruit names, extract those fruit names and assign an emoji to them
    Return the fruit name and emojis in a python dictionary.
    '''

    fruit_names = """
    Apple, Pear, this is an kiwi
    """

    # Make your prompt which combines the instructions with fruit names
    prompt = (instructions + fruit_names)

    # Call the llm
    output_dict = chat_model([HumanMessage(content=prompt)])

    print(output_dict)
    print(type(output_dict))

    print("Finish")


if __name__ == "__main__":
    extraction_manual()
