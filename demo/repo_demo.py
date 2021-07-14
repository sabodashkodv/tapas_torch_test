import pandas as pd
from component.component import TapasModel
from easydict import EasyDict
from pathlib import Path

root = Path(__file__).parent.parent

path = f'{root}/data/repo.csv'
query = ["how many repositories?",
         "which programming language is the most used?",
         "which framework is implemented in rust?"]
frame = pd.read_csv(path)

model = TapasModel(EasyDict({'model_name': 'google/tapas-base-finetuned-wtq'}))
model.transform(frame, query)
