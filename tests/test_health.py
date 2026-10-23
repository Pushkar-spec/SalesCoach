from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root_status():
    """Verify that GET / returns HTTP 200 OK."""
    response = client.get("/")
    assert response.status_code == 200


def test_read_root_payload():
    """Verify that GET / returns expected Hello World message."""
    response = client.get("/")
    assert response.json() == {"message": "Hello World"}


def test_health_check_status():
    """Verify that GET /health returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_payload():
    """Verify that GET /health returns expected service name and status."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "sales-copilot-api"
    assert "environment" in data
