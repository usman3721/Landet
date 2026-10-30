from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.app.core.security import create_token

router = APIRouter()


class AuthInput(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(auth: AuthInput):
    if (auth.username == "admin") and (auth.password == "admin"):
        token = create_token({"sub": auth.username})
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Invalid Credentials")
