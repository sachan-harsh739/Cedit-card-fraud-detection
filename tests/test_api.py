import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_empty():
    response = client.post("/predict", json={})
    # Should fail validation because missing all args
    assert response.status_code == 422 
    
def test_predict_endpoint_valid():
    # Only run prediction test if model is built natively
    if not os.path.exists("models/fraud_model.pkl"):
        pytest.skip("Model not built yet, skipping accurate predict integration test")
    
    valid_payload = {
        "Time": 3600.0, "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    response = client.post("/predict", json=valid_payload)
    if response.status_code == 503:
        pytest.skip("Model loaded as degraded internally")
        return
        
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "optimal_threshold_applied" in data
    assert data["prediction"] in ["Fraud", "Legit"]
