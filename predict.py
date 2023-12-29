from transformers import AutoTokenizer, TapasForQuestionAnswering, TapasConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from generate import generate_merchants, generate_users, generate_queries
from dataset import TableDataset
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import MultiDateMatcher
from pyspark.sql.types import *
from datetime import datetime
import time
from collections import defaultdict
from InquirerPy import prompt
from argparse import ArgumentParser
import warnings
from typing import List
from params import *


def predict(model: nn.Module, inputs: dict, table: List[pd.DataFrame]):
    """Forward propagate inputs through the trained model and return predicted coordinates and corresponding answers using the original table.

    Args:
        model: Trained PyTorch model
        inputs: Dictionary of input tensors
        table: List of transaction tables associated with input queries
    
    Returns:
        Tuple of predicted coordinates and (post-aggregated) prediction
    """
    outputs = model(**inputs)
    predicted_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach(), cell_classification_threshold=0.5
    )

    predictions = []
    for i, coordinates in enumerate(predicted_coordinates):
        cell_values = []
        for coordinate in coordinates:
            cell_values.append(table[i].iat[coordinate])
        if predicted_aggregation_indices[i] == SUM:
            aggregated_prediction = str(round(sum([float(c) for c in cell_values]), 2))
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("SUM > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == COUNT:
            aggregated_prediction = str(len(cell_values))
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("COUNT > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == AVERAGE:
            aggregated_prediction = str(round(sum([float(c) for c in cell_values]) / len(cell_values), 2))
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("AVERAGE > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == ARGMAX_SUM:
            amounts = defaultdict(float)
            for coordinate in coordinates:
                amounts[table[i].iat[coordinate]] += float(table[i].iat[(coordinate[0], table[i].columns.get_loc('amount'))])
            aggregated_prediction = max(amounts, key=amounts.get)
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("ARGMAX_SUM > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == ARGMIN_SUM:
            amounts = defaultdict(float)
            for coordinate in coordinates:
                amounts[table[i].iat[coordinate]] += float(table[i].iat[(coordinate[0], table[i].columns.get_loc('amount'))])
            aggregated_prediction = min(amounts, key=amounts.get)
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("ARGMIN_SUM > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == ARGMAX_STD:
            amounts = defaultdict(list)
            for coordinate in coordinates:
                amounts[table[i].iat[coordinate]].append(float(table[i].iat[(coordinate[0], table[i].columns.get_loc('amount'))]))
            aggregated_prediction = max(amounts, key=lambda k: np.std(amounts[k]))
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("ARGMAX_STD > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        elif predicted_aggregation_indices[i] == ARGMIN_STD:
            amounts = defaultdict(list)
            for coordinate in coordinates:
                amounts[table[i].iat[coordinate]].append(float(table[i].iat[(coordinate[0], table[i].columns.get_loc('amount'))]))
            aggregated_prediction = min(amounts, key=lambda k: np.std(amounts[k]))
            aggregated_prediction = "\033[1;32m" + aggregated_prediction + "\033[0m"
            predictions.append("ARGMIN_STD > " + ", ".join(cell_values) + " > " + aggregated_prediction)
        else:
            predictions.append("NONE > " + ", ".join(cell_values))
    
    return predicted_coordinates, predictions

def score(test_dataloader: DataLoader, model: nn.Module):
    """Runs inference and scores trained model with a generated test dataset.

    Args:
        test_dataloader: PyTorch dataloader with test data
        model: Trained PyTorch model
    """
    model.eval()

    with torch.no_grad():
        for inputs, references in tqdm(test_dataloader):
            inputs = { k: v.to(DEVICE) for k, v in inputs.items() }

            predicted_coordinates, predictions = predict(model, inputs, [r['table'] for r in references])
            for coordinates, prediction, reference in zip(predicted_coordinates, predictions, references):
                print(f"Query: {reference['query']['question']}\nCoordinates: {coordinates}\nPrediction: {prediction}\nAnswer: {reference['query']['answer_text']}")


warnings.simplefilter(action='ignore', category=FutureWarning)

parser = ArgumentParser(prog='predict.py', description='Script for making query predictions on tabular data.')
parser.add_argument('-m', '--mode', choices=['score', 'input'], default='score', help='prediction mode (either score on a generated test dataset or manually input queries)')
parser.add_argument('-c', '--ckpt-file', type=str, default=CHECKPOINT_FILE, help='path to model checkpoint file')

args = parser.parse_args()

checkpoint = torch.load(args.ckpt_file, map_location=DEVICE)
config = TapasConfig(num_aggregation_labels=8, answer_loss_cutoff=1e5)
model = TapasForQuestionAnswering.from_pretrained(PRETRAINED_MODEL, config=config, ignore_mismatched_sizes=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

generate_merchants(save_file='test-merchants.csv')
generate_users(num=10 if args.mode == 'score' else 1, file='test-merchants.csv', save_file='test-users.csv')

if args.mode == 'score':
    generate_queries(file='test-users.csv', save_file=TEST_FILE)

    def collate_fn(batch):
        encodings, references = zip(*batch)
        encodings, references = list(encodings), list(references)
        return default_collate(encodings), references

    test_iter = TableDataset(TEST_FILE, 'test-users.csv', tokenizer, mode='test')
    test_dataloader = DataLoader(test_iter, batch_size=1, collate_fn=collate_fn)

    score(test_dataloader, model)
else:
    users = pd.read_csv('test-users.csv')
    transactions = pd.DataFrame.from_dict(json.loads(users.iloc[0]['transactions'])).astype(str)
    print(transactions.to_markdown())

    spark = sparknlp.start()

    while True:
        query = prompt({
            'type': 'input',
            'name': 'query',
            'message': 'Input a desired query for the above table. (Enter q to quit)'
        })['query']

        if query == 'q':
            break

        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        preprocess = Pipeline(stages=[
            DocumentAssembler().setInputCol("text").setOutputCol("document"),
            MultiDateMatcher().setInputCols("document").setOutputCol("multi_date").setOutputFormat("MM/dd/yy")
        ])

        spark_df = spark.createDataFrame([query], StringType()).toDF("text")
        result = preprocess.fit(spark_df).transform(spark_df)

        substrings = []
        timestamps = []
        for date in result.select("multi_date").collect()[0].multi_date:
            substrings.append(query[date.begin:date.end + 1])
            timestamp = str(int(time.mktime(datetime.strptime(date.result, '%m/%d/%y').timetuple())))
            timestamps.append(timestamp)
        
        for substring, timestamp in zip(substrings, timestamps):
            query = query.replace(substring, timestamp)

        inputs = tokenizer(
            table=transactions,
            queries=query,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        predicted_coordinates, predictions = predict(model, inputs, [transactions])
        
        print(f"Query: {query}\nCoordinates: {predicted_coordinates[0]}\nPrediction: {predictions[0]}")

