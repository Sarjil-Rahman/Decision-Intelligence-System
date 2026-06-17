from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.datasets import resolve_dataset_dir, resolve_dataset_file, validate_required_dataset_files


def _write_dataset(root: Path, dataset_id: str = "demo") -> Path:
    data_dir = root / dataset_id
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["x"],
            "item_id": ["i"],
            "dept_id": ["d"],
            "cat_id": ["c"],
            "store_id": ["s"],
            "state_id": ["st"],
            "d_1": [1],
        }
    ).to_csv(data_dir / "sales_train_validation.csv", index=False)
    pd.DataFrame(
        {
            "d": ["d_1"],
            "date": ["2020-01-01"],
            "wm_yr_wk": [1],
            "weekday": ["Wed"],
            "wday": [1],
            "month": [1],
            "year": [2020],
            "event_name_1": [None],
            "event_type_1": [None],
            "event_name_2": [None],
            "event_type_2": [None],
            "snap_CA": [0],
            "snap_TX": [0],
            "snap_WI": [0],
        }
    ).to_csv(data_dir / "calendar.csv", index=False)
    pd.DataFrame(
        {"store_id": ["s"], "item_id": ["i"], "wm_yr_wk": [1], "sell_price": [1.0]}
    ).to_csv(data_dir / "sell_prices.csv", index=False)
    pd.DataFrame({"id": ["x"], "F1": [0.0]}).to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


def test_dataset_resolver_rejects_traversal_absolute_and_missing_files(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path)
    assert resolve_dataset_dir(tmp_path, "demo") == data_dir.resolve()
    with pytest.raises(ValueError):
        resolve_dataset_dir(tmp_path, "../demo")
    with pytest.raises(ValueError):
        resolve_dataset_file(data_dir, "../secret.csv")
    with pytest.raises(ValueError):
        resolve_dataset_file(data_dir, str(Path(data_dir).resolve() / "x.csv"))
    missing = tmp_path / "missing_required"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_required_dataset_files(missing)


def test_symlink_escape_is_rejected_where_supported(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    link = data_dir / "link.csv"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")
    with pytest.raises(ValueError):
        resolve_dataset_file(data_dir, "link.csv")


def test_required_dataset_symlink_escape_is_rejected_where_supported(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path)
    outside = tmp_path / "external_sell_prices.csv"
    outside.write_text("store_id,item_id,wm_yr_wk,sell_price\ns,i,1,1.0\n", encoding="utf-8")
    target = data_dir / "sell_prices.csv"
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")
    with pytest.raises(ValueError, match="outside"):
        validate_required_dataset_files(data_dir)


def test_job_submission_requires_api_key_in_production(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    from api.settings import get_settings

    get_settings.cache_clear()
    from api.main import app

    client = TestClient(app)
    denied = client.post("/v1/jobs/forecast", json={"dataset_id": "demo", "params": {}})
    assert denied.status_code == 401
    accepted = client.post(
        "/v1/jobs/forecast",
        json={"dataset_id": "demo", "params": {}},
        headers={"X-API-Key": "secret"},
    )
    assert accepted.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert "/v1/jobs/" in accepted.json()["status_url"]


def test_public_job_params_reject_unknown_and_path_overrides(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    from api.settings import get_settings

    get_settings.cache_clear()
    from api.main import app

    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    for params in [{"unknown": 1}, {"data_dir": str(tmp_path)}, {"out_submission": "x.csv"}]:
        response = client.post(
            "/v1/jobs/forecast",
            json={"dataset_id": "demo", "params": params},
            headers=headers,
        )
        assert response.status_code == 422

    endpoint_params = [
        ("/v1/jobs/price-actions", {"submission_path": str(tmp_path / "submission.csv")}),
        ("/v1/jobs/price-actions", {"out_path": "elsewhere.csv"}),
        ("/v1/jobs/promo-selection", {"input_path": "../price_optimization_results.csv"}),
        ("/v1/jobs/promo-selection", {"out_path": "/tmp/promo.csv"}),
        ("/v1/jobs/business-pack", {"reports_subdir": "../reports"}),
        ("/v1/jobs/business-pack", {"data_dir": str(tmp_path)}),
    ]
    for endpoint, params in endpoint_params:
        response = client.post(
            endpoint,
            json={"dataset_id": "demo", "params": params},
            headers=headers,
        )
        assert response.status_code == 422

    response = client.post(
        "/v1/jobs/forecast",
        json={"dataset_id": "demo", "params": {"max_series": 1, "n_jobs": 1}},
        headers=headers,
    )
    assert response.status_code == 202


def test_worker_revalidates_payload_and_sanitizes_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    from api.settings import get_settings
    from worker import tasks

    get_settings.cache_clear()
    payload = {
        "dataset_id": "demo",
        "params": {"data_dir": r"C:\secret\m5", "out_path": "/tmp/leak.csv"},
    }
    result = tasks._task_wrapper(tasks._forecast_impl, payload)
    assert result["status"] == "failed"
    assert "C:\\secret" not in result["message"]
    assert "/tmp/leak.csv" not in result["message"]
    assert "Traceback" not in result["message"]
    sanitized = tasks._sanitize_public_error(
        "boom redis://:pw@localhost:6379/0 postgresql://u:p@host/db api_key=abc123"
    )
    assert "redis://" not in sanitized
    assert "postgresql://" not in sanitized
    assert "abc123" not in sanitized


def test_status_lookup_uses_configured_celery_app(monkeypatch) -> None:
    from api import job_routes

    calls = []

    class FakeResult:
        state = "SUCCESS"
        result = {
            "status": "succeeded",
            "started_at": "2020-01-01T00:00:00+00:00",
            "finished_at": "2020-01-01T00:00:01+00:00",
            "result": {"ok": True},
        }

    class FakeCeleryApp:
        def AsyncResult(self, job_id):
            calls.append(job_id)
            return FakeResult()

    monkeypatch.setattr(job_routes, "celery_app", FakeCeleryApp())
    response = job_routes.get_job_status("abc123")
    assert calls == ["abc123"]
    assert response.status == "succeeded"
    assert response.result == {"ok": True}


def test_sync_endpoint_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_SYNC_ENDPOINTS", raising=False)
    from api.settings import get_settings

    get_settings.cache_clear()
    from api.main import app

    client = TestClient(app)
    response = client.post("/internal/business-pack", json={"data_dir": "data"})
    assert response.status_code == 404
