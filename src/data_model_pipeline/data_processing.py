import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def process_language_data(train_df, val_df, test_df):
    """Cleans and vectorizes text data, then encodes labels."""

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # remove URLs, mentions
        text = re.sub(r"\d+", "", text)  # remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        return text.lower().strip()

    # Clean text data
    X_train = train_df["prompt"].apply(clean_text).values
    X_val = val_df["prompt"].apply(clean_text).values
    X_test = test_df["prompt"].apply(clean_text).values

    # Target labels
    y_train = train_df["language"].values
    y_val = val_df["language"].values
    y_test = test_df["language"].values

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        analyzer="char_wb",
        min_df=10
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    return {
        "X_train_tfidf": X_train_tfidf,
        "X_val_tfidf": X_val_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "y_train_encoded": y_train_encoded,
        "y_val_encoded": y_val_encoded,
        "y_test_encoded": y_test_encoded,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder
    }


def dump_processed_data(processed_data, directory="models"):
    """Saves TF-IDF vectorizer and label encoder for later use."""
    os.makedirs(directory, exist_ok=True)

    joblib.dump(processed_data["vectorizer"], os.path.join(directory, "vectorizer.joblib"))
    joblib.dump(processed_data["label_encoder"], os.path.join(directory, "label_encoder.joblib"))

    print(f"✅ Saved vectorizer and label encoder to {directory}/")


def main():
    # Load datasets
    train_df = pd.read_csv("data/preprocessed/train.csv")
    val_df = pd.read_csv("data/preprocessed/val.csv")
    test_df = pd.read_csv("data/preprocessed/test.csv")

    # Process and dump
    processed_data = process_language_data(train_df, val_df, test_df)
    dump_processed_data(processed_data)


if __name__ == "__main__":
    main()
