# Budgetwise Machine Learning Engineer Intern Challenge
> Generate fake user bank and credit card transaction data and run relevant queries with a NLP model.

## Overview
This repository contains two major components:
- a script that generates synthetic bank and credit card transaction data for a fictional user dating back to the last 24 months
    - auto-generated user profile
    - transactions include date, amount, merchant, and spending category
- a training and inference script for a NLP model that can process natural language queries related to the user’s transaction history

## Usage
Clone the repository.
```shell
git clone git@github.com:joe-lin-tech/budgetwise.git
cd budgetwise
```

Create and activate a virtual environment. (Alternatively, use an existing environment of your choosing.)
```shell
python3 -m venv venv
source venv/bin/activate
```

Install required pip packages and dependencies.
```shell
python3 -m pip install -r requirements.txt
```

Login to a wandb account if you plan to run the training script and would like to view train logs.
```shell
wandb login
```

Your local environment should now be suitable to run the scripts in this repository.

<details open>
<summary>Generation Script</summary>

As a summary, this script can generate (1) a list of merchants, (2) a transaction history, (3) a database of users, (4) relevant queries as training data.

1. **Generate Merchants.** The following command generates a list of 100 merchants and saves it to merchants.csv:

    ```shell
    python3 generate.py -t merchants -n 100 -s merchants.csv
    ```

2. **Generate Transactions.** The following command generates a history of 48 transactions from the past year, given a list of merchants:

    ```shell
    python3 generate.py -t transactions -n 48 -f merchants.csv
    ```

3. **Generate Users.** The following command generates a database of 10 users with 48 transactions from the past year, given a list of merchants:

    ```shell
    python3 generate.py -t users -n 10 -f merchants.csv -s users.csv
    ```

4. **Generate Queries.** The following command generates a set of queries based on provided user + transaction data.

    ```shell
    python3 generate.py -t queries
    ```

For more function API details, expand the following:
<details>
<summary>Generation Script API Details</summary>

```shell
generate_merchants(num: int = 1000, save_file: str = None)
    Generates a list of fake merchants and saves their name and corresponding category in a file.
    
    If the argument `save_file` is None, the list is saved as merchants.csv in the current working directory.
    
    Args:
        num (int): Number of fake merchants to generate (default is 1000)
        save_file (str): Path to save file (default is None)
```

```shell
generate_transactions(num: int = 48, delta: int = 365, file: str = 'merchants.csv', save_file: str = None)
    Generates a list of fake transactions (from `delta` days in the past to now) given a list of merchants.
    
    If the argument `save_file` is None, the list of transactions is returned instead.
    
    Args:
        num (int): Number of fake transactions to generate (default is 48)
        delta (int): Number of days in the past to generate transactions from (default is 365)
        file (str): Path to list of merchants (default is merchants.csv)
        save_file (str): Path to save file (default is None)
    
    Returns:
        transactions (list): a list of generated transactions
```

```shell
generate_users(num: int = 10, file: str = 'merchants.csv', save_file: str = None)
    Generates a list of fake users and saves their profile and transactions in a file.
    
    If the argument `save_file` is None, the list is saved as users.csv in the current working directory.
    
    Args:
        num (int): Number of fake users to generate (default is 10)
        file (str): Path to list of merchants for generating transactions from (default is merchants.csv)
        save_file (str): Path to save file (default is None)
```

```shell
generate_queries(file: str = 'users.csv', save_file: str = None)
    Generates a set of queries in a SQA-similar format.
    
    If the argument `save_file` is None, the list is saved as data.csv in the current working directory.
    
    Args:
        file (str): Path to list of users (default is users.csv)
        save_file (str): Path to save file (default is None)
```
</details>
</details>

<br>
<details open>
<summary>Training + Inference Script</summary>

To train the model, simply run the following:
```shell
python3 train.py
```

For inference, there are two modes to run under: (1) score and (2) input:
```shell
python3 predict.py -m score
```

```shell
python3 predict.py -m input
```
</details>