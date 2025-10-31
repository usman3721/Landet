import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from src.data_model_pipeline import data_processing
import os
import sys
import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_model_pipeline.data_processing import process_language_data

MODELS_DIR = "models"


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
    model_path = os.path.join(MODELS_DIR, "language_detector.joblib")
    joblib.dump(model, model_path)
    

    return {"model_path": model_path}


if __name__ == "__main__":
    mlflow.set_experiment("language_detector_pipeline")
    with mlflow.start_run(run_name="model_training"):

        try:
            import subprocess, json
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            mlflow.set_tag("git_commit", git_sha)
        except Exception:
            pass

        mlflow.log_param("model_type", "MultinomialNB")

        train_df = pd.read_csv("data/preprocessed/train.csv")
        val_df = pd.read_csv("data/preprocessed/val.csv")
        test_df = pd.read_csv("data/preprocessed/test.csv")

        model=train_language_model(train_df, val_df, test_df)
        mlflow.log_artifact(model["model_path"], artifact_path="models")
        mlflow.sklearn.log_model(model, "model") 
    # Load datasets


    # Train and save the model
    