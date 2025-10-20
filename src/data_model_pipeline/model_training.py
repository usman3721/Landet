import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from src.data_model_pipeline import data_processing
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_model_pipeline.data_processing import process_language_data




def train_language_model(train_df, val_df, test_df, model_dir="models"):
    """Trains a Multinomial Naive Bayes model on processed language data."""

    # Process the data
    processed_data = process_language_data(train_df, val_df, test_df)

    X_train = processed_data["X_train_tfidf"]
    y_train = processed_data["y_train_encoded"]

    # Initialize and train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{model_dir}/language_model.pkl")

    return model


if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv("data/preprocessed/train.csv")
    val_df = pd.read_csv("data/preprocessed/val.csv")
    test_df = pd.read_csv("data/preprocessed/test.csv")

    # Train and save the model
    train_language_model(train_df, val_df, test_df)