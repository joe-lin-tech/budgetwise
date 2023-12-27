from faker import Factory
from argparse import ArgumentParser
import csv
import random
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import random
from load import load_merchants, load_users
from collections import defaultdict
from params import *


def generate_merchants(num: int = 1000, save_file: str = None):
    """Generates a list of fake merchants and saves their name and corresponding category in a file.

    If the argument `save_file` is None, the list is saved as merchants.csv in the current working directory.

    Args:
        num (int): Number of fake merchants to generate (default is 1000)
        save_file (str): Path to save file (default is None)
    """
    fake = Factory.create('en_US')
    with open(save_file if save_file else 'merchants.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'category'])
        for i in range(num):
            writer.writerow([fake.company(), random.choice(CATEGORIES)])

def generate_transactions(num: int = 48, delta: int = 365, file: str = 'merchants.csv', save_file: str = None):
    """Generates a list of fake transactions (from `delta` days in the past to now) given a list of merchants.

    If the argument `save_file` is None, the list of transactions is returned instead.

    Args:
        num (int): Number of fake transactions to generate (default is 48)
        delta (int): Number of days in the past to generate transactions from (default is 365)
        file (str): Path to list of merchants (default is merchants.csv)
        save_file (str): Path to save file (default is None)

    Returns:
        transactions (list): a list of generated transactions
    """
    fake = Factory.create('en_US')
    merchants = load_merchants(file)
    transactions = []
    for i in range(num):
        date = fake.date_time_between_dates(datetime.now() - timedelta(days=delta), datetime.now())
        date = int(time.mktime(date.timetuple()))
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

def generate_users(num: int = 10, file: str = 'merchants.csv', save_file: str = None):
    """Generates a list of fake users and saves their profile and transactions in a file.

    If the argument `save_file` is None, the list is saved as users.csv in the current working directory.

    Args:
        num (int): Number of fake users to generate (default is 10)
        file (str): Path to list of merchants for generating transactions from (default is merchants.csv)
        save_file (str): Path to save file (default is None)
    """
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

def generate_questions(file: str = 'users.csv', save_file: str = None):
    """Generates a set of questions in a SQA-similar format.

    If the argument `save_file` is None, the list is saved as data.csv in the current working directory.

    Args:
        num (int): Number of fake users to generate (default is 10)
        file (str): Path to list of users (default is users.csv)
        save_file (str): Path to save file (default is None)
    """
    users = load_users(file)
    queries = []
    for id in range(len(users)):
        user = users.iloc[id]
        transactions = pd.DataFrame.from_dict(user['transactions'])

        # generate 20 questions of the following forms:
        # (1) Between (start_date) and (end_date), how many transactions were made?
        # (2) How many transactions were made between (start_date) and (end_date)?
        for i in range(20):
            date_idxs = sorted(random.choices(range(len(transactions)), k=2))
            start_date, end_date = transactions.iloc[date_idxs[0]]['date'], transactions.iloc[date_idxs[1]]['date']
            question = random.choice([
                f"Between {start_date} and {end_date}, how many transactions were made?",
                f"How many transactions were made between {start_date} and {end_date}?"
            ])
            answer_coordinates = [(i, transactions.columns.get_loc('date')) for i in range(date_idxs[0], date_idxs[1] + 1)]
            answer_float = date_idxs[1] - date_idxs[0] + 1
            queries.append({
                "id": id,
                "question": question,
                "answer_coordinates": json.dumps(answer_coordinates),
                "answer_text": str(answer_float),
                "aggregation_labels": COUNT
            })

        # generate question with one of the following forms:
        # (1) What category did I spend most frequently on?
        # (2) What was my biggest spending category?
        category = transactions['category'].mode()[0]
        answer_coordinates = [(i, transactions.columns.get_loc('category')) for i in range(len(transactions)) if transactions.iloc[i]['category'] == category]
        question = random.choice([
            "What category did I spend most frequently on?",
            "What was my biggest spending category?"
        ])
        queries.append({
            "id": id,
            "question": question,
            "answer_coordinates": json.dumps(answer_coordinates),
            "answer_text": category,
            "aggregation_labels": NONE
        })

        # generate question with one of the following forms:
        # (1) What spending category am I most inconsistent in?
        # (2) What category do I spend the least on?
        category = transactions['category'].value_counts().index[-1]
        answer_coordinates = [(i, transactions.columns.get_loc('category')) for i in range(len(transactions)) if transactions.iloc[i]['category'] == category]
        question = random.choice([
            "What spending category am I most inconsistent in?",
            "What category do I spend the least on?"
        ])
        queries.append({
            "id": id,
            "question": question,
            "answer_coordinates": json.dumps(answer_coordinates),
            "answer_text": category,
            "aggregation_labels": NONE
        })

        # generate 20⋅(1 + len(CATEGORIES)) questions of the following forms:
        # (1) What is the total amount I've spent from (ref_timestamp) to (now_timestamp)?
        # (2) What is the total amount I've spent from (num_days) days ago to today?
        # (3) In the last (num_days) days, how much did I spend in total?
        # (4) What is the total amount I've spent in (category) from (ref_date) to (now_timestamp)?
        # (5) What is the total amount I've spent in (category) from (num_days) days ago to today?
        max_days = (datetime.now() - datetime.fromtimestamp(transactions.iloc[0]['date'])).days
        num_days = sorted(random.choices(range(max_days), k=20))
        i = 0
        curr_amt = defaultdict(float)
        for j in range(len(transactions) - 1, -1, -1):
            while i < len(num_days):
                ref_date = datetime.now() - timedelta(days=num_days[i])
                curr_date = datetime.fromtimestamp(transactions.iloc[j]['date'])

                if ref_date > curr_date:
                    ref_timestamp = int(time.mktime(ref_date.timetuple()))
                    now_timestamp = int(time.mktime(datetime.now()))
                    question = random.choice([
                        f"What is the total amount I've spent from {ref_timestamp} to {now_timestamp}?",
                        f"What is the total amount I've spent from {num_days[i]} days ago to today?",
                        f"In the last {num_days[i]} days, how much did I spend in total?"
                    ])
                    answer_coordinates = [(i, transactions.columns.get_loc('amount')) for i in range(len(transactions) - 1, j, -1)]
                    answer_float = round(curr_amt['total'], 2)
                    if answer_float > 0 and len(answer_coordinates) > 0:
                        queries.append({
                            "id": id,
                            "question": question,
                            "answer_coordinates": json.dumps(answer_coordinates),
                            "answer_text": str(answer_float),
                            "aggregation_labels": SUM
                        })
                    for category in CATEGORIES:
                        question = random.choice([
                            f"What is the total amount I've spent in {category} from {ref_timestamp} to {now_timestamp}?",
                            f"What is the total amount I've spent in {category} from {num_days[i]} days ago to today?"
                        ])
                        answer_float = round(curr_amt[category], 2)
                        answer_coordinates = [(i, transactions.columns.get_loc('amount')) for i in range(len(transactions) - 1, j, -1) if transactions.iloc[i]['category'] == category]
                        if answer_float > 0 and len(answer_coordinates) > 0:
                            queries.append({
                                "id": id,
                                "question": question,
                                "answer_coordinates": json.dumps(answer_coordinates),
                                "answer_text": str(answer_float),
                                "aggregation_labels": SUM
                            })
                    i += 1
                else:
                    curr_amt['total'] += transactions.iloc[j]['amount']
                    curr_amt[transactions.iloc[j]['category']] += transactions.iloc[j]['amount']
                    break

        # generate following question for each spending category:
        # (1) What is the total amount I've spent in (category)?
        for category in CATEGORIES:
            total = transactions[transactions['category'] == category]['amount'].sum()
            question = f"What is the total amount I've spent in {category}?"
            answer_coordinates = [(i, transactions.columns.get_loc('amount')) for i in range(len(transactions)) if transactions.iloc[i]['category'] == category]
            answer_float = round(total, 2)
            if answer_float > 0 and len(answer_coordinates) > 0:
                queries.append({
                    "id": id,
                    "question": question,
                    "answer_coordinates": json.dumps(answer_coordinates),
                    "answer_text": str(answer_float),
                    "aggregation_labels": SUM
                })
        
        # generate following question for each merchant:
        # (1) What is the total amount I've spent at (merchant)?
        for merchant in transactions['merchant'].unique():
            total = transactions[transactions['merchant'] == merchant]['amount'].sum()
            question = f"What is the total amount I've spent at {merchant}?"
            answer_coordinates = [(i, transactions.columns.get_loc('amount')) for i in range(len(transactions)) if transactions.iloc[i]['merchant'] == merchant]
            answer_float = round(total, 2)
            queries.append({
                "id": id,
                "question": question,
                "answer_coordinates": json.dumps(answer_coordinates),
                "answer_text": str(answer_float),
                "aggregation_labels": SUM
            })

        # generate following question:
        # (1) What merchant did I spend the most on?
        merchant = transactions['merchant'].mode()[0]
        answer_coordinates = [(i, transactions.columns.get_loc('amount')) for i in range(len(transactions)) if transactions.iloc[i]['merchant'] == merchant]
        question = "What merchant did I spend the most on?"
        queries.append({
            "id": id,
            "question": question,
            "answer_coordinates": json.dumps(answer_coordinates),
            "answer_text": merchant,
            "aggregation_labels": NONE
        })
    
    # write queries to save file
    with open(save_file if save_file else 'data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'answer_coordinates', 'answer_text', 'aggregation_labels'])
        for q in queries:
            writer.writerow(list(q.values()))


if __name__ == "__main__":
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