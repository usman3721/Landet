import joblib

from src.data_model_pipeline.data_ingestion import split_data
from src.data_model_pipeline.data_processing import fit_preprocessors, save_preprocessors
from src.data_model_pipeline.model_evaluation import evaluate_model
from src.data_model_pipeline.model_training import save_model, train_language_model


def test_pipeline_runs_end_to_end(sample_df, tmp_path):
    """Run every pipeline step on tiny data and check the saved model works."""
    train_df, val_df, _ = split_data(sample_df)

    vectorizer, label_encoder = fit_preprocessors(train_df)
    save_preprocessors(vectorizer, label_encoder, directory=tmp_path)
    model = train_language_model(train_df, vectorizer, label_encoder)
    model_path = save_model(model, directory=tmp_path)

    report, _, _ = evaluate_model(
        joblib.load(model_path), vectorizer, label_encoder, val_df, report_dir=tmp_path
    )

    assert report["accuracy"] > 0.9
    assert {path.name for path in tmp_path.iterdir()} == {
        "vectorizer.joblib",
        "label_encoder.joblib",
        "language_detector.joblib",
        "report.json",
        "confusion_matrix.png",
    }
