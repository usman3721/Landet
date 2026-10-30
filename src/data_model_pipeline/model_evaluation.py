"""Step 4: measure the model on data it never saw while training."""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_model_pipeline.data_processing import MODELS_DIR, PROCESSED_DIR, clean_text

REPORTS_DIR = "reports"

# Use "val" while experimenting. Use "test" once, at the very end, for your final score:
# if you keep tuning against the test set, it stops telling you about unseen data.
EVAL_SPLIT = "val"


def evaluate_model(model, vectorizer, label_encoder, df, report_dir=REPORTS_DIR):
    """Score the model on df and save a JSON report and a confusion matrix plot."""
    X = vectorizer.transform(df["prompt"].apply(clean_text).values)
    y_true = label_encoder.transform(df["language"].values)
    y_pred = model.predict(X)

    report_data = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            digits=3,
            output_dict=True,
            zero_division=0,
        ),
    }

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), fmt="d", cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return report_data, report_path, cm_path


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{EVAL_SPLIT}.csv"))
    model = joblib.load(os.path.join(MODELS_DIR, "language_detector.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    report_data, report_path, cm_path = evaluate_model(model, vectorizer, label_encoder, df)
    print(f"{EVAL_SPLIT} accuracy: {report_data['accuracy']:.4f}")
    print(f"Saved {report_path} and {cm_path}")


if __name__ == "__main__":
    main()
