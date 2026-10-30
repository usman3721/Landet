"""Step 1: load the raw CSV, drop unused columns, split into train/val/test."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILE_PATH = "data/raw/new_merged_languages.csv"
PROCESSED_DIR = "data/preprocessed"


def load_data(file_path=RAW_FILE_PATH):
    """Load raw dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def clean_data(df):
    """Drop unnecessary columns."""
    drop_cols = ["new_id", "old_id", "data_source", "label"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"Columns dropped: {drop_cols}")
    return df


def split_data(df):
    """Split dataset into train (80%), val (10%), and test (10%)."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["language"],  # preserve class distribution
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["language"],
        random_state=42,
    )

    print(f"Split: train {train_df.shape}, val {val_df.shape}, test {test_df.shape}")
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir=PROCESSED_DIR):
    """Save train, val, and test data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Saved train/val/test to {output_dir}/")


def main():
    df = clean_data(load_data())
    train_df, val_df, test_df = split_data(df)
    save_processed_data(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
