import csv
import pandas as pd
import json


def load_merchants(file: str):
    merchants = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            merchants[row[0]] = row[1]
    return merchants

def load_users(file: str):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                users = pd.DataFrame(columns=row)
                continue
            users.loc[len(users.index)] = row
            users.loc[len(users.index) - 1]['transactions'] = json.loads(users.loc[len(users.index) - 1]['transactions'])
    return users