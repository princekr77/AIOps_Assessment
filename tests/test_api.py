import pytest
from fastapi.testclient import TestClient
from app import app, model

client = TestClient(app)


def test_root_endpoint():
    """Verify that the root endpoint returns metadata and status 200."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "documentation" in data


def test_health_endpoint():
    """Verify that the health check endpoint returns status: ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_spam():
    """Verify that an obvious spam text is correctly classified as 'spam'."""
    payload = {"text": "WINNER! You have won a guaranteed 1000 cash prize. Call 09061701461 to claim now!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "spam"
    assert 0.5 <= data["confidence"] <= 1.0


def test_predict_ham():
    """Verify that a legitimate message is correctly classified as 'ham'."""
    payload = {"text": "Hi Mark, let's schedule our project review meeting for tomorrow at 10 AM. Thanks!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "ham"
    assert 0.5 <= data["confidence"] <= 1.0


def test_predict_empty_text():
    """Verify that empty or whitespace strings return an HTTP error."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code in (400, 422)


def test_model_pipeline_steps():
    """Verify that the loaded pipeline contains the expected steps."""
    assert "tfidf" in model.named_steps
    assert "clf" in model.named_steps
