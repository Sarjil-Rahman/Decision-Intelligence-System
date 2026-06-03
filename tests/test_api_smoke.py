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


def test_business_pack_endpoint_returns_controlled_error_without_upstream_outputs(tmp_path):
    c = TestClient(app)
    r = c.post("/business-pack", json={"data_dir": str(tmp_path)})
    assert r.status_code == 400
    assert "price_optimization_results.csv not found" in r.json()["error"]
