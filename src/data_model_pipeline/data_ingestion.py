import pandas as pd
from sklearn.model_selection import train_test_split
import os
import mlflow

RAW_FILE_PATH = "data/raw/new_merged_languages.csv"
PROCESSED_DIR = "data/preprocessed"

mlflow.sklearn.autolog()

def load_data(file_path=RAW_FILE_PATH):
    """Load raw dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df):
    """Drop unnecessary columns."""
    drop_cols = ['new_id', 'old_id', 'data_source', 'label']
    df = df.drop(columns=drop_cols, errors='ignore')
    print(f"🧹 Data cleaned. Columns dropped: {drop_cols}")
    return df


def split_data(df):
    """Split dataset into train (80%), val (10%), and test (10%)."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['language'],  # preserve class distribution
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['language'],
        random_state=42
    )

    print(f"📊 Data split completed:")
    print(f"  • Train: {train_df.shape}")
    print(f"  • Val:   {val_df.shape}")
    print(f"  • Test:  {test_df.shape}")
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir=PROCESSED_DIR):
    """Save processed train, val, and test data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"💾 Saved processed datasets to '{output_dir}' successfully!")


def main():
    mlflow.set_experiment("language_detector_pipeline")

    with mlflow.start_run(run_name="data_ingestion"):

        df = load_data()
        df = clean_data(df)
        train_df, val_df, test_df = split_data(df)
        save_processed_data(train_df, val_df, test_df)

        # logging parameters and artifacts
        mlflow.log_param("train_row",len(train_df))
        mlflow.log_param("val_row",len(val_df))
        mlflow.log_param("test_row",len(test_df))

        print("🏁 Data ingestion process completed. Saved to",PROCESSED_DIR)


if __name__ == "__main__":
    main()
