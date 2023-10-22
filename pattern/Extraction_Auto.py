#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage


def extraction_auto():
    # prepare llm
    openai_api_key = os.environ["OPENAI_API_KEY"]
    chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    response_schemas = [
        ResponseSchema(name="artist", description="The name of the musical artist"),
        ResponseSchema(name="song", description="The name of the song that the artist plays")
    ]

    # 解析器将会把LLM的输出使用我们定义的schema进行解析并返回期待的结构数据
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)
    print("----End format instructions ----")

    # 这个 Prompt 与之前构建的 Chat Model 的 Prompt 不同
    # 这个 Prompt 是一个 ChatPromptTemplate， 会自动将我们的输出转化为 python 对象
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("Given a command from the user, extract the artist and sone names \n \
                                                     {format_instructions}\n{user_prompt}")
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )

    artist_query = prompt.format_prompt(user_prompt="I really like So Young by Portugal. The Man")
    print(artist_query.messages[0].content)

    artist_output = chat_model(artist_query.to_messages())
    print(f"Chat return: {artist_output}")
    output = output_parser.parse(artist_output.content)

    print(output)
    print(type(output))
    print("Finish")


if __name__ == "__main__":
    extraction_auto()
