from fastapi.testclient import TestClient

from src.app.core.security import create_token, verify_token
from src.app.main import app

client = TestClient(app)


def test_token_round_trip():
    assert verify_token(create_token({"sub": "admin"}))["sub"] == "admin"


def test_tampered_token_is_rejected():
    assert verify_token(create_token({"sub": "admin"}) + "x") is None


def test_expired_token_is_rejected():
    assert verify_token(create_token({"sub": "admin"}, expire_minutes=-1)) is None


def test_login_with_good_credentials_returns_token():
    response = client.post("/login", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert verify_token(response.json()["access_token"])


def test_login_with_bad_credentials_is_rejected():
    response = client.post("/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401
