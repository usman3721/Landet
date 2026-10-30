from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt

from src.app.core.config import settings


def create_token(data: dict, expire_minutes=30):
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str):
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None
