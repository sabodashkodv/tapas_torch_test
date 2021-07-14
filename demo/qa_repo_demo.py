import pandas as pd
from pathlib import Path
from tableqa.agent import Agent
from tabulate import tabulate

root = Path(__file__).parent.parent

path = f'{root}/data/repo.csv'
queries = ["how many repositories?",
           "which programming language is the most used?",
           "which framework is implemented in rust?",
           "What is the total amount of stars?",
           "Sum of all start",
           "The maximum of repository"]
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
