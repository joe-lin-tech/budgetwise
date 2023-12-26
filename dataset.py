import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import numpy as np
from typing import Literal
from params import *

class TableDataset(Dataset):
    def __init__(self, data_path: str, users_path: str, tokenizer: AutoTokenizer, mode: Literal['train', 'val', 'test'] = 'train'):
        self.data = pd.read_csv(data_path)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.mode = mode
        if self.mode == 'train':
            self.data = self.data.iloc[indices[:int(0.9 * len(self.data))]]
        elif self.mode == 'val':
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

        params = dict(
            table=table,
            queries=query['question'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            # max_length=1024 TODO - increase max length for larger tables
        )

        if self.mode != 'test':
            params['answer_coordinates'] = json.loads(query['answer_coordinates'])
            params['answer_text'] = query['answer_text']
        
        encoding = self.tokenizer(**params)
        # remove the batch dimension which the tokenizer adds by default
        encoding = { key: val.squeeze(0) for key, val in encoding.items() }

        if self.mode != 'test':
            encoding['aggregation_labels'] = torch.tensor(int(query['aggregation_labels']))

        if self.mode == 'test':
            reference = dict(query=query, table=table)
            return encoding, reference
        
        return encoding