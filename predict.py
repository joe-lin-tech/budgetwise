from transformers import AutoTokenizer, TapasForQuestionAnswering, TapasConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from generate import generate_merchants, generate_users, generate_questions
from dataset import TableDataset
from tqdm import tqdm
import pandas as pd
import json
from InquirerPy import prompt
from argparse import ArgumentParser
import warnings
from typing import List
from params import *


def predict(model: nn.Module, inputs: dict, table: List[pd.DataFrame]):
    outputs = model(**inputs)
    predicted_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    predictions = []
    for i, coordinates in enumerate(predicted_coordinates):
        if len(coordinates) == 1:
            # only a single cell:
            predictions.append(table[i].iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table[i].iat[coordinate])
            if predicted_aggregation_indices[i] == SUM:
                aggregated_prediction = str(round(sum([float(c) for c in cell_values]), 2))
                predictions.append(AGGREGATION_OPS[SUM] + " > " + ", ".join(cell_values) + " > " + aggregated_prediction)
            elif predicted_aggregation_indices[i] == COUNT:
                aggregated_prediction = str(len(cell_values))
                predictions.append(AGGREGATION_OPS[COUNT] + " > " + aggregated_prediction)
            else:
                predictions.append(AGGREGATION_OPS[NONE] + " > " + ", ".join(cell_values))
    
    return predicted_coordinates, predictions

def score(test_dataloader: DataLoader, model: nn.Module):
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
config = TapasConfig(num_aggregation_labels=4, answer_loss_cutoff=1e5)
model = TapasForQuestionAnswering.from_pretrained(PRETRAINED_MODEL, config=config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

generate_merchants(save_file='test-merchants.csv')
generate_users(num=10 if args.mode == 'score' else 1, file='test-merchants.csv', save_file='test-users.csv')

if args.mode == 'score':
    generate_questions(file='test-users.csv', save_file=TEST_FILE)

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

    while True:
        query = prompt({
            'type': 'input',
            'name': 'query',
            'message': 'Input a desired query for the above table. (Enter q to quit)'
        })['query']

        if query == 'q':
            break

        inputs = tokenizer(
            table=transactions,
            queries=query,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        predicted_coordinates, predictions = predict(model, inputs, [transactions])
        
        print(f"Query: {query}\nCoordinates: {predicted_coordinates[0]}\nPrediction: {predictions[0]}")

