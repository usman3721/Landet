import pytest
from fastapi.testclient import TestClient

from src.app.api import routes_predict
from src.app.core.config import settings
from src.app.core.security import create_token
from src.app.main import app

client = TestClient(app)


@pytest.fixture
def headers():
    return {"token": create_token({"sub": "admin"}), "api-key": settings.API_KEY}


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """API tests check the routes, so they don't need the real 100 MB model."""
    monkeypatch.setattr(routes_predict, "predict_language", lambda text: "English")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_language(headers):
    response = client.post("/predict", json={"text": "hello"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"detected_language": "English"}


def test_predict_requires_auth_headers():
    assert client.post("/predict", json={"text": "hello"}).status_code == 422


def test_predict_rejects_wrong_api_key(headers):
    headers["api-key"] = "wrong"
    assert client.post("/predict", json={"text": "hello"}, headers=headers).status_code == 403


def test_predict_rejects_bad_token(headers):
    headers["token"] = "wrong"
    assert client.post("/predict", json={"text": "hello"}, headers=headers).status_code == 401


def test_predict_requires_text(headers):
    assert client.post("/predict", json={}, headers=headers).status_code == 422
