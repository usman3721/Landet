import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def evaluate_model(model, X_val_tfidf, y_val_encoded, label_encoder, show_samples=True, save_report=True, report_dir="reports"):
    """
    Evaluate a trained model on validation data and optionally save a JSON report.

    Parameters
    ----------
    model : trained model (e.g., MultinomialNB)
        The trained model to evaluate.
    X_val_tfidf : sparse matrix
        TF-IDF transformed validation features.
    y_val_encoded : array-like
        Encoded true labels for validation set.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Used to decode label integers to class names.
    show_samples : bool
        Whether to print sample predictions for inspection.
    save_report : bool
        Whether to save the evaluation metrics to JSON.
    report_dir : str
        Directory where report.json will be saved.
    """

    # ------------------------
    # Predictions
    # ------------------------
    y_val_pred = model.predict(X_val_tfidf)

    # ------------------------
    # Metrics
    # ------------------------
    accuracy = accuracy_score(y_val_encoded, y_val_pred)
    precision = precision_score(y_val_encoded, y_val_pred, average="weighted")
    recall = recall_score(y_val_encoded, y_val_pred, average="weighted")
    f1 = f1_score(y_val_encoded, y_val_pred, average="weighted")

    # ------------------------
    # Print Summary
    # ------------------------
    print("\n🚀 MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print(f"📚 Number of Languages: {len(label_encoder.classes_)}")
    print(f"🧩 Validation Samples:  {len(y_val_encoded)}")
    print("=" * 50)

    # ------------------------
    # Detailed Report
    # ------------------------
    class_report = classification_report(
        y_val_encoded,
        y_val_pred,
        target_names=label_encoder.classes_,
        digits=3,
        output_dict=True  # ✅ For saving to JSON
    )

    # ------------------------
    # Show Random Samples
    # ------------------------
    if show_samples:
        print("\n🔍 Sample Predictions:")
        sample_indices = np.random.choice(len(y_val_encoded), 5, replace=False)
        for idx in sample_indices:
            actual = label_encoder.inverse_transform([y_val_encoded[idx]])[0]
            predicted = label_encoder.inverse_transform([y_val_pred[idx]])[0]
            correct = "✅" if actual == predicted else "❌"
            print(f"   {correct} Actual: {actual:15} → Predicted: {predicted}")

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    import pandas as pd
    import joblib
    from src.data_model_pipeline.data_processing import process_language_data
    from src.data_model_pipeline.model_training import train_language_model

    # Load datasets
    train_df = pd.read_csv("data/preprocessed/train.csv")
    val_df = pd.read_csv("data/preprocessed/val.csv")
    test_df = pd.read_csv("data/preprocessed/test.csv")

    # Process the data
    processed_data = process_language_data(train_df, val_df, test_df)

    X_val_tfidf = processed_data["X_val_tfidf"]
    y_val_encoded = processed_data["y_val_encoded"]
    label_encoder = processed_data["label_encoder"]

    # Load the trained model
    model = joblib.load("models/language_model.pkl")

    # Evaluate the model
    evaluate_model(
        model,
        X_val_tfidf,
        y_val_encoded,
        label_encoder,
        show_samples=True,
        save_report=True,
        report_dir="reports"
    )
