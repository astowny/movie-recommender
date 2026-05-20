from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_missing_model(monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main, "MODEL", None)
    response = client.get("/recommend/1")
    assert response.status_code == 200
    assert response.json()["error"] == "model not trained"
