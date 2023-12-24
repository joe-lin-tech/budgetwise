from transformers import AutoTokenizer, TapasTokenizer, TapasForQuestionAnswering, TapasConfig
import pandas as pd
import json
import torch

tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
users = pd.read_csv("users.csv")
users['transactions'] = users['transactions'].apply(lambda t: json.loads(t))
transactions = pd.DataFrame.from_dict(users.iloc[0]['transactions']).astype(str)

data = pd.read_csv("data.csv")

questions = data['question'].to_list()
answer_coordinates = [json.loads(c) for c in data['answer_coordinates']]
answer_text = data['answer_text'].to_list()

inputs = tokenizer(
    table=transactions.astype(str),
    queries=questions,
    answer_coordinates=answer_coordinates,
    answer_text=answer_text,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
inputs['aggregation_labels'] = torch.tensor(data['aggregation_labels'])

config = TapasConfig(num_aggregation_labels=4)
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", config=config)
outputs = model(**inputs)
loss = outputs.loss
loss.backward()