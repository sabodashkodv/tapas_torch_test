from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
from tabulate import tabulate
from easydict import EasyDict


class TapasModel:
    def __init__(self, config: EasyDict):
        self.config = config
        self.model_name = config.model_name
        self.model = self.init_model()
        self.tokenizer = self.init_tokenizer()

    def init_model(self):
        return TapasForQuestionAnswering.from_pretrained(self.model_name)

    def init_tokenizer(self):
        return TapasTokenizer.from_pretrained(self.model_name)

    def fit(self):
        raise NotImplemented

    def transform(self, table: pd.DataFrame, queries: list):
        data_copy = table.copy()
        data_copy = data_copy.astype(str)
        inputs = self.tokenizer(table=data_copy, queries=queries, padding='max_length', return_tensors="pt")
        outputs = self.model(**inputs)

        predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach(),
            outputs.logits_aggregation.detach()
        )
        self.get_answer(predicted_aggregation_indices, predicted_answer_coordinates, data_copy, queries)

    def get_answer(self, predicted_aggregation_indices, predicted_answer_coordinates, table, queries):
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))

        print(tabulate(table, headers='firstrow'))
        for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
            print(query)
            if predicted_agg == "NONE":
                print("Predicted answer: " + answer)
            else:
                print("Predicted answer: " + predicted_agg + " > " + answer)
