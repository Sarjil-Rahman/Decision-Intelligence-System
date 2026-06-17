from fastapi.testclient import TestClient

from api.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_metrics():
    c = TestClient(app)
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_public_business_pack_endpoint_removed_by_default(tmp_path):
    c = TestClient(app)
    r = c.post("/business-pack", json={"data_dir": str(tmp_path)})
    assert r.status_code == 404
