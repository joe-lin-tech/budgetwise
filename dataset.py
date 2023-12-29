import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import numpy as np
from typing import Literal
from params import *

class TableDataset(Dataset):
    """Dataset of generated Table Question Answering queries in a SQA-similar format.

    Attributes:
        data: Pandas dataframe containing queries
        users: Pandas dataframe containing user metadata and transactions
        mode: Literal string representing the type of dataset split
        tokenizer: A transformers Tokenizer for preprocessing input queries and table entries
    """
    def __init__(self, data_path: str, users_path: str, tokenizer: AutoTokenizer, mode: Literal['train', 'val', 'test'] = 'train'):
        """Initializes a dataset instance according to the provided source paths, tokenizer, and mode.

        Args:
            data_path: Path to file with generated queries
            users_path: Path to file with associated user metadata and transaction history
            tokenizer: A transformers Tokenizer for preprocessing input queries and table entries
            mode: Literal string representing the type of dataset split (default is train)
        """
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
        """Computes the length of this dataset instance.

        Returns:
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a dataset element and preprocesses it with the tokenizer.

        If the dataset is in test mode, additionally return the input query and table as a reference.

        Returns:
            Dictionary with input encodings formatted properly for TAPAS model processing.
        """
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