from fastapi.testclient import TestClient

from ml_system.api.main import app

client = TestClient(app)


def test_api_not_loaded():
    # Since models aren't loaded in test isolenv w/o files, expect 503
    response = client.post("/predict", json={"features": [{"test": 1}]})
    assert response.status_code == 503
