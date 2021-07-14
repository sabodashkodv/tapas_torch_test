import pandas as pd
from component.component import TapasModel
from easydict import EasyDict
from pathlib import Path

root = Path(__file__).parent.parent
n_rows = 5
path = f'{root}/data/pollution.csv'
cols = ['pollution', 'wind_speed', 'temperature']
query = ["What is the highest pollution?",
         "What is the highest wind_speed?",
         "What is the highest temperature?"]
frame = pd.read_csv(path).head(n_rows)[cols]

model = TapasModel(EasyDict({'model_name': 'google/tapas-base-finetuned-wtq'}))
model.transform(frame, query)

