import csv
import pandas as pd
import json


def load_merchants(file: str = 'merchants.csv'):
    """Loads a dictionary of merchants given a merchant file.

    Args:
        file (str): Path to list of merchants (default is merchants.csv)
    
    Returns:
        merchants (dict): a dictionary of saved merchants
    """
    merchants = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            merchants[row[0]] = row[1]
    return merchants

def load_users(file: str = 'users.csv'):
    """Loads a dataframe of users given a users file.

    Args:
        file (str): Path to list of users (default is users.csv)
    
    Returns:
        users (pd.DataFrame): a Pandas dataframe of saved users
    """
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                users = pd.DataFrame(columns=row)
                continue
            users.loc[len(users.index)] = row
            users.loc[len(users.index) - 1]['transactions'] = json.loads(users.loc[len(users.index) - 1]['transactions'])
    return users