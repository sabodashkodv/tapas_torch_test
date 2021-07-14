import pandas as pd
from component.component import TapasModel
from easydict import EasyDict
from pathlib import Path

root = Path(__file__).parent.parent

path = f'{root}/data/drivers.csv'
query = ["what were the drivers names?",
         "of these, which points did patrick carpentier and bruno junqueira score?",
         "who scored higher?"]
frame = pd.read_csv(path)

model = TapasModel(EasyDict({'model_name': 'google/tapas-base-finetuned-wtq'}))
model.transform(frame, query)
