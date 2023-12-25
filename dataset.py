import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import numpy as np
from typing import Literal

class TableDataset(Dataset):
    def __init__(self, data_path: str, users_path: str, tokenizer: AutoTokenizer, mode: Literal['train', 'val', 'test'] = 'train'):
        self.data = pd.read_csv(data_path)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        if mode == 'train':
            self.data = self.data.iloc[indices[:int(0.9 * len(self.data))]]
        else:
            self.data = self.data.iloc[indices[int(0.9 * len(self.data)):]]
        self.data = self.data.reset_index(drop=True)
        self.users = pd.read_csv(users_path)
        self.users['transactions'] = self.users['transactions'].apply(lambda t: json.loads(t))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]
        table = pd.DataFrame.from_dict(self.users.iloc[int(query['id'])]['transactions']).astype(str)
        encoding = self.tokenizer(
            table=table,
            queries=query['question'],
            answer_coordinates=json.loads(query['answer_coordinates']),
            answer_text=query['answer_text'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = { key: val.squeeze(0) for key, val in encoding.items() }
        encoding['aggregation_labels'] = torch.tensor(int(query['aggregation_labels']))
        return encoding