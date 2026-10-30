"""Step 2: learn how to turn text into numbers (TF-IDF) and labels into numbers."""

import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

PROCESSED_DIR = "data/preprocessed"
MODELS_DIR = "models"


def clean_text(text):
    """Remove what says nothing about the language. Training and the API both use this."""
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # remove URLs, mentions
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text.lower().strip()


def fit_preprocessors(train_df):
    """Fit the vectorizer and label encoder on the training data only."""
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        analyzer="char_wb",  # character patterns identify a language better than whole words
        min_df=10,
    )
    vectorizer.fit(train_df["prompt"].apply(clean_text).values)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["language"].values)

    return vectorizer, label_encoder


def save_preprocessors(vectorizer, label_encoder, directory=MODELS_DIR):
    """Save the TF-IDF vectorizer and label encoder for later use."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(directory, "vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(directory, "label_encoder.joblib"))


def main():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    vectorizer, label_encoder = fit_preprocessors(train_df)
    save_preprocessors(vectorizer, label_encoder)
    print(f"Saved vectorizer and label encoder to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
