import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# Paths to the raw credit card dataset and output directory
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'creditcard_part01.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def generate_dataset(num_samples: int = 5000, seed: int = 42):
    """
    Generate a synthetic dataset for fraud detection using the provided
    credit card dataset. A subset of the original data is sampled and
    augmented with random user IDs, merchants and timestamps.

    Parameters
    ----------
    num_samples : int
        Number of rows to include in the generated dataset. If the
        underlying dataset has more rows than required, a random subset
        will be drawn. Otherwise, sampling with replacement is used.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the synthetic dataset with columns
        ['user_id', 'amount', 'merchant', 'timestamp', 'label'] where
        label is 1 for fraudulent transactions and 0 otherwise.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load the raw dataset. Only read necessary columns to save memory.
    data = pd.read_csv(RAW_DATA_PATH, usecols=['Amount', 'Class'])
    total_rows = len(data)

    # Determine sampling strategy based on available data
    if total_rows >= num_samples:
        sampled_df = data.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    else:
        sampled_df = data.sample(n=num_samples, replace=True, random_state=seed).reset_index(drop=True)

    # Define list of merchants
    merchants = [
        'Amazon', 'Walmart', 'Target', 'Starbucks', 'Uber', 'Netflix',
        'Apple', 'Google Play', 'McDonalds', 'Costco'
    ]

    num_users = 50  # Number of distinct users to simulate

    # Add synthetic fields
    sampled_df['user_id'] = np.random.randint(1, num_users + 1, size=num_samples)
    sampled_df['merchant'] = np.random.choice(merchants, size=num_samples)

    # Generate timestamps uniformly over the past 90 days
    now = datetime.utcnow()
    max_days = 90
    timestamps = []
    for _ in range(num_samples):
        # Random offset in seconds
        offset_sec = np.random.randint(0, max_days * 24 * 60 * 60)
        ts = now - timedelta(seconds=int(offset_sec))
        timestamps.append(ts.isoformat())
    sampled_df['timestamp'] = timestamps

    # Rename columns to generic names
    sampled_df.rename(columns={'Amount': 'amount', 'Class': 'label'}, inplace=True)

    # Rearrange columns
    df = sampled_df[['user_id', 'amount', 'merchant', 'timestamp', 'label']].copy()
    return df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.7, test_ratio: float = 0.15, seed: int = 42):
    """
    Split the dataset into training, testing and simulation sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to split.
    train_ratio : float
        Proportion of the data to use for training.
    test_ratio : float
        Proportion of the data to use for testing. The remainder will
        be used for simulation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_df, test_df, sim_df : tuple of pandas.DataFrame
        DataFrames corresponding to the training, testing and simulation
        portions of the data.
    """
    np.random.seed(seed)
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))
    train_df = df_shuffled.iloc[:train_end].reset_index(drop=True)
    test_df = df_shuffled.iloc[train_end:test_end].reset_index(drop=True)
    sim_df = df_shuffled.iloc[test_end:].reset_index(drop=True)
    return train_df, test_df, sim_df


def main():
    """Generate the dataset and write the splits to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_dataset(num_samples=5000)
    train_df, test_df, sim_df = split_dataset(df)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'training_data.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False)
    sim_df.to_csv(os.path.join(OUTPUT_DIR, 'simulation_data.csv'), index=False)
    print(f"Generated dataset with {len(df)} rows.\n"
          f"Training: {len(train_df)}, Test: {len(test_df)}, Simulation: {len(sim_df)}")


if __name__ == '__main__':
    main()