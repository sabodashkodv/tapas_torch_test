import pandas as pd
from pathlib import Path
from tableqa.agent import Agent
from tabulate import tabulate

root = Path(__file__).parent.parent

path = f'{root}/data/pollution.csv'
queries = ["The maximum pollution",
           "What is the maximum wind_speed?",
           "What is the maximum temperature?",
           "What is the temperature range?",
           "What is press maximum and minimum?",
           "Rain at 2014-12-31 23:00:00"]
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
