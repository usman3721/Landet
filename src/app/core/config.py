import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME = 'Language Detector API'
    MODEL_PATH = 'models\language_detector.joblib'
    VECTOR_PATH = 'models/vectorizer.joblib'
    ENCODER_PATH = 'models\label_encoder.joblib'
    API_KEY = os.getenv('API_KEY', 'demo-key')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'secret')
    JWT_ALGORITHM = 'HS256'
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')



settings = Settings()