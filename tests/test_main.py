from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API"}

def test_prediction():
    response = client.post("/predict/", json=[[0.2, 0.1, 0.4, 0.6, 0.9]])
    assert response.status_code == 200
    assert "fraud" in response.json()

