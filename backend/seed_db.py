"""
Utility script to initialise the transactions table with sample data.

This script reads the generated training dataset from
`dataset/training_data.csv`, trains a logistic regression model on the
`amount` feature and computes a risk score and classification for each
transaction. It then inserts these records into the configured
database. Existing records in the table are left untouched.

Usage:
    $ python backend/seed_db.py

Ensure that the `.env` file is configured appropriately for your
database. By default it will use an SQLite database `fraud.db` in the
project root.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LogisticRegression

from app import Transaction, THRESH_APPROVE, THRESH_FLAG


# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
DB_URI = os.getenv('DB_URI', 'sqlite:///fraud.db')

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


def main():
    # Read training dataset
    training_csv = os.path.join(DATASET_DIR, 'training_data.csv')
    if not os.path.exists(training_csv):
        raise FileNotFoundError(f"Training data not found at {training_csv}. Did you run prepare_data.py?")
    df = pd.read_csv(training_csv)
    
    # Fit a simple logistic regression model on amount
    X = df[['amount']].values.reshape(-1, 1)
    y = df['label'].astype(int).values
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    # Compute risk scores for each row and determine classification
    probs = model.predict_proba(X)[:, 1]
    classifications = []
    for p in probs:
        if p < THRESH_APPROVE:
            classifications.append('Approved')
        elif p < THRESH_FLAG:
            classifications.append('Flagged')
        else:
            classifications.append('Fraud')
    df['risk_score'] = probs
    df['result'] = classifications

    # Convert timestamp column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Initialise the database session
    engine = create_engine(DB_URI, echo=False, future=True)
    SessionLocal = sessionmaker(bind=engine)
    # Create table if not exists
    Transaction.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        # Check if table already has data
        existing_count = session.query(Transaction).count()
        if existing_count > 0:
            print(f"Found {existing_count} existing records. Seeding skipped.")
            return
        # Bulk insert
        objs = []
        for _, row in df.iterrows():
            obj = Transaction(
                user_id=int(row['user_id']),
                amount=float(row['amount']),
                merchant=str(row['merchant']),
                timestamp=row['timestamp'],
                risk_score=float(row['risk_score']),
                result=str(row['result'])
            )
            objs.append(obj)
        session.bulk_save_objects(objs)
        session.commit()
        print(f"Inserted {len(objs)} training records into the database.")
    finally:
        session.close()


if __name__ == '__main__':
    main()