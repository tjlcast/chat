#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time 2023/10/22 S{TIME} 
# @Name Querying_Tabular_Data. Py
# @Author：jialtang
import os

from langchain.llms.openai import OpenAI
from langchain import SQLDatabase, SQLDatabaseChain

sqlite_db_path = 'data/San_Franciso_Trees.db'


def query_tabular_data():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    db = SQLDatabase.from_uri(f'sqlite:///{sqlite_db_path}')

    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    db_chain.run("How many Species of trees are there in San Francisco?")


# Find which table to use
# Find which column to use
# Construct the correct sql query
# Execute that query
# Get the result
# Return a natural language reponse back

import sqlite3
import pandas as pd


def confirm_result():
    # Connect to the SQLite database
    connection = sqlite3.connect(sqlite_db_path)

    # Define your SQL query
    query = "SELECT count(distinct qSpecies) FROM SFTrees"

    # Read the SQL query into a Pandas DataFrame
    df = pd.read_sql_query(query, connection)

    # Close the connection
    connection.close()

    # Display the result in the first column first cell
    print(df.iloc[0, 0])


if __name__ == "__main__":
    query_tabular_data()
