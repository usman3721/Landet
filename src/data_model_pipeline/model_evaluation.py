import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import mlflow
from src.data_model_pipeline.data_processing import process_language_data
from src.data_model_pipeline.model_training import train_language_model
import seaborn as sns
MODELS_DIR = "models"
REPORTS_DIR = "reports"

def evaluate_model(model, X_val_tfidf, y_val_encoded, label_encoder, show_samples=True, save_report=True, report_dir="reports"):

    y_val_pred = model.predict(X_val_tfidf)


    accuracy = accuracy_score(y_val_encoded, y_val_pred)
    precision = precision_score(y_val_encoded, y_val_pred, average="weighted")
    recall = recall_score(y_val_encoded, y_val_pred, average="weighted")
    f1 = f1_score(y_val_encoded, y_val_pred, average="weighted")


    class_report = classification_report(
        y_val_encoded,
        y_val_pred,
        target_names=label_encoder.classes_,
        digits=3,
        output_dict=True  # ✅ For saving to JSON
    )

        # confusion matrix plot
    cm = confusion_matrix(y_val_encoded, y_val_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, fmt='d', cmap='Blues')
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    # ------------------------

    # Save Report
    # ------------------------
    if save_report:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "report.json")

        report_data = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": class_report
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)

        print(f"\n📝 Evaluation report saved at: {report_path}")

    return report_data,cm_path


if __name__ == "__main__":


    # Load datasets
    train_df = pd.read_csv("data/preprocessed/train.csv")
    val_df = pd.read_csv("data/preprocessed/val.csv")
    test_df = pd.read_csv("data/preprocessed/test.csv")

    # Process the data
    processed_data = process_language_data(train_df, val_df, test_df)

    X_val_tfidf = processed_data["X_val_tfidf"]
    y_val_encoded = processed_data["y_val_encoded"]
    label_encoder = processed_data["label_encoder"]

    model_path = os.path.join(MODELS_DIR, "language_detector.joblib")
    
    # Load the trained model
    model = joblib.load("models\language_detector.joblib")
    mlflow.set_experiment("language_detector_pipeline")
    with mlflow.start_run(run_name="model_evaluation"):
        try:
            import subprocess
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            mlflow.set_tag("git_commit", git_sha)
        except Exception:
            pass

    # Evaluate the model
        report,cm_path=evaluate_model(
            model,
            X_val_tfidf,
            y_val_encoded,
            label_encoder,
            show_samples=True,
            save_report=True,
            report_dir="reports"
        )

        mlflow.log_metric("test_accuracy", report["accuracy"])
        mlflow.log_artifact(os.path.join(REPORTS_DIR, "report.json"), artifact_path="reports")
        mlflow.log_artifact(cm_path, artifact_path="reports")
        print("Evaluation finished and logged to MLflow.")
