import joblib
import pandas as pd
import mlflow
import subprocess
from app.core.config import settings
from app.cache.redis_cache import set_cached_prediction, get_cached_prediction

# ✅ Load model, vectorizer, and encoder
model = joblib.load(settings.MODEL_PATH)
vectorizer = joblib.load(settings.VECTOR_PATH)
encoder = joblib.load(settings.ENCODER_PATH)

mlflow.set_tracking_uri("file:./mlruns")  # local MLflow tracking

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def predict_language(text: str):
    """
    Predict the language of the given text using vectorizer + model + encoder.
    """
    cache_key = text.lower().strip()
    cached = get_cached_prediction(cache_key)
    if cached:
        return cached

    # ✅ Step 1 — Vectorize input text
    X = vectorizer.transform([text])

    # ✅ Step 2 — Predict (numeric label)
    y_pred = model.predict(X)[0]  # This should work if model has predict method

    # ✅ Step 3 — Decode predicted label to language name
    language = encoder.inverse_transform([y_pred])[0]

    # ✅ Cache the result
    set_cached_prediction(cache_key, language)

    # ✅ Log in MLflow - FIXED: Don't log the actual objects, just metadata
    with mlflow.start_run(run_name="language_detection"):
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_param("model_path", settings.MODEL_PATH)
        mlflow.log_param("vectorizer_path", settings.VECTOR_PATH)
        mlflow.log_param("encoder_path", settings.ENCODER_PATH)
        mlflow.log_param("input_text_sample", text[:100])  # Log sample of input
        mlflow.log_metric("input_length", len(text))
        mlflow.log_metric("is_cached", 0)
        mlflow.log_metric("predicted_label", y_pred)
        mlflow.log_param("predicted_language", language)
        
        # Only log file paths as artifacts, not the loaded objects
        # mlflow.log_artifact(settings.MODEL_PATH)  # Remove or fix this
        # mlflow.log_artifact(settings.VECTOR_PATH)  # Remove or fix this  
        # mlflow.log_artifact(settings.ENCODER_PATH)  # Remove or fix this

    return language