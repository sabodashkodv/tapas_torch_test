import pandas as pd
from pathlib import Path
from tableqa.agent import Agent
from tabulate import tabulate

root = Path(__file__).parent.parent

path = f'{root}/data/drivers.csv'
queries = ["The maximum points", "what were the drivers names?",
           "of these, which points did patrick carpentier and bruno junqueira score?",
           "who scored higher?",
           "Display all teams",
           "What is the most common team?"]
frame = pd.read_csv(path)
print(tabulate(frame))

agent = Agent(frame)
for query in queries:
    try:
        print(f'Query {query}')
        request = agent.get_query(query)
        print(f'SQL {request}')
        response = agent.query_db(query)
    except:
        continue
    print(f'Response is {response[0]}')
