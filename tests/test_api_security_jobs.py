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


def test_sync_endpoint_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_SYNC_ENDPOINTS", raising=False)
    from api.settings import get_settings

    get_settings.cache_clear()
    from api.main import app

    client = TestClient(app)
    response = client.post("/internal/business-pack", json={"data_dir": "data"})
    assert response.status_code == 404
