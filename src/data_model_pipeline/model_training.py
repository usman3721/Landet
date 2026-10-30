"""Step 3: train the classifier on the training set."""

import os

import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from src.data_model_pipeline.data_processing import MODELS_DIR, PROCESSED_DIR, clean_text


def train_language_model(train_df, vectorizer, label_encoder):
    """Trains a Multinomial Naive Bayes model on the training data."""
    X_train = vectorizer.transform(train_df["prompt"].apply(clean_text).values)
    y_train = label_encoder.transform(train_df["language"].values)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def save_model(model, directory=MODELS_DIR):
    """Save the trained model."""
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, "language_detector.joblib")
    joblib.dump(model, model_path)
    return model_path


def main():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    model = train_language_model(train_df, vectorizer, label_encoder)
    print(f"Saved model to {save_model(model)}")


if __name__ == "__main__":
    main()
