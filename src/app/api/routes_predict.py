from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.dependencies import get_api_key, get_current_user
from app.services.language_service import predict_language

router = APIRouter()

# ✅ Define the input schema
class TextInput(BaseModel):
    text: str

# ✅ Define the endpoint
@router.post("/predict")
def detect_language(payload: TextInput, user=Depends(get_current_user), _=Depends(get_api_key)):
    """
    Detects the language of the given text using the trained ML model.
    """
    language = predict_language(payload.text)
    return {"detected_language": language}
