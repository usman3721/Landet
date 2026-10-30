from functools import lru_cache

import joblib

from src.app.cache.redis_cache import get_cached_prediction, set_cached_prediction
from src.app.core.config import settings
from src.data_model_pipeline.data_processing import clean_text


@lru_cache
def load_artifacts():
    """Load the model once, on the first prediction, then keep it in memory."""
    model = joblib.load(settings.MODEL_PATH)
    vectorizer = joblib.load(settings.VECTOR_PATH)
    encoder = joblib.load(settings.ENCODER_PATH)
    return model, vectorizer, encoder


def predict_language(text: str):
    """Predict the language of the given text using vectorizer + model + encoder."""
    # Text must be cleaned exactly as it was in training, or the model sees different input.
    cleaned = clean_text(text)

    cached = get_cached_prediction(cleaned)
    if cached:
        return cached

    model, vectorizer, encoder = load_artifacts()

    # Step 1: turn text into numbers
    X = vectorizer.transform([cleaned])

    # Step 2: predict (numeric label)
    y_pred = model.predict(X)

    # Step 3: decode the numeric label back to a language name
    language = str(encoder.inverse_transform(y_pred)[0])

    set_cached_prediction(cleaned, language)
    return language
