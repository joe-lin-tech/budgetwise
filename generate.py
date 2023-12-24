from faker import Factory
from argparse import ArgumentParser
import csv
import random
import json
import pandas as pd
from datetime import datetime, timedelta
from load import load_merchants, load_users


CATEGORIES = ['agricultural', 'contracted', 'transportation', 'utility', 'retail', 'clothing',
              'misc', 'business', 'government', 'airlines', 'lodging', 'professional']

SUM, COUNT, AVERAGE, NONE = 0, 1, 2, 3

def generate_merchants(num: int = 1000, save_file: str = None):
    fake = Factory.create('en_US')
    with open(save_file if save_file else 'merchants.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'category'])
        for i in range(num):
            writer.writerow([fake.company(), random.choice(CATEGORIES)])

def generate_transactions(num: int = 64, delta: int = 365, save_file: str = None, file: str = None):
    fake = Factory.create('en_US')
    merchants = load_merchants(file)
    transactions = []
    for i in range(num):
        date = fake.date_time_between_dates(datetime.now() - timedelta(days=delta), datetime.now())
        amount = round(random.uniform(1, 250), 2) # TODO - fix amount range
        merchant = random.choice(list(merchants.keys()))
        category = merchants[merchant]
        transactions.append(dict(date=date, amount=amount, merchant=merchant, category=category))

    transactions = sorted(transactions, key=lambda t: t['date'])

    if save_file:
        with open('transactions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'amount', 'merchant', 'category'])
            for t in transactions:
                writer.writerow(list(t.values()))
    else:
        return transactions

def generate_users(num: int = 10, save_file: str = None, file: str = None):
    fake = Factory.create('en_US')
    users = []
    for i in range(num):
        profile = fake.profile()
        transactions = generate_transactions(file=file) # TODO - add transaction amount parameter
        users.append({ **profile, 'transactions': json.dumps(transactions, default=str) })
    with open(save_file if save_file else 'users.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(list(users[0].keys()))
        for u in users:
            writer.writerow(list(u.values()))

def generate_questions(file: str = None):
    users = load_users(file)
    for id in range(len(users)):
        user = users.iloc[id]
        transactions = pd.DataFrame.from_dict(user['transactions'])
        queries = []

        for i in range(10):
            date_idxs = sorted(random.choices(range(len(transactions)), k=2))
            start_date, end_date = transactions.iloc[date_idxs[0]]['date'], transactions.iloc[date_idxs[1]]['date']
            # start_date, end_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S.%f'), datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S.%f')
            question = f"Between {start_date} and {end_date}, how many transactions were made?"
            answer_coordinates = [(i, transactions.columns.get_loc('date')) for i in range(date_idxs[0], date_idxs[1] + 1)]
            answer_float = date_idxs[1] - date_idxs[0] + 1
            queries.append({
                "id": id,
                "question": question,
                "answer_coordinates": json.dumps(answer_coordinates),
                "answer_text": str(answer_float),
                "aggregation_labels": COUNT
            })

        category = transactions.category.mode()[0]
        answer_coordinates = [[i, transactions.columns.get_loc('category')] for i in range(len(transactions)) if transactions.iloc[i]['category'] == category]
        question = "What category did I spend the most on overall?"
        queries.append({
            "id": id,
            "question": question,
            "answer_coordinates": json.dumps(answer_coordinates),
            "answer_text": category,
            "aggregation_labels": COUNT
        })
    
        with open('data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'question', 'answer_coordinates', 'answer_text', 'aggregation_labels'])
            for q in queries:
                writer.writerow(list(q.values()))


parser = ArgumentParser(prog='generate.py', description='Script for generating financial relevant data.')
parser.add_argument('-t', '--type', choices=['merchants', 'transactions', 'users', 'questions'], help='type of data to generate')
parser.add_argument('-n', '--num', type=int, help='number of data points')
parser.add_argument('-d', '--delta', type=int, help='how far to look back in transaction history (in days)')
parser.add_argument('-s', '--save-file', type=str, help='path to save file')
parser.add_argument('-f', '--file', type=str, help='path to data source file')

args = parser.parse_args()
params = { arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None }
params.pop('type')

if args.type == 'merchants':
    generate_merchants(**params)
elif args.type == 'transactions':
    generate_transactions(**params)
elif args.type == 'users':
    generate_users(**params)
elif args.type == 'questions':
    generate_questions(**params)