from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from api.datasets import resolve_dataset_dir, resolve_dataset_file, validate_required_dataset_files
from api.settings import get_settings
from m5_pipeline.business_outputs import BusinessPackConfig, generate_business_pack
from m5_pipeline.m5_forecasting import ForecastConfig, run_forecast
from m5_pipeline.m5_price_optimization import PriceOptConfig, run_price_optimization
from m5_pipeline.m5_promo_selection import PromoSelectionConfig, run_promo_selection
from worker.celery_app import celery_app


def _safe_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": type(exc).__name__,
        "message": str(exc).replace(str(Path.cwd()), "<workspace>"),
    }


def _run_dataset_job(dataset_id: str, func: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    settings = get_settings()
    dataset_dir = resolve_dataset_dir(settings.data_root, dataset_id)
    validate_required_dataset_files(dataset_dir)
    started = datetime.now(timezone.utc).isoformat()
    result = func(dataset_dir)
    return {
        "status": "succeeded",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }


def _forecast_impl(payload: dict[str, Any]) -> dict[str, Any]:
    def run(dataset_dir: Path) -> dict[str, Any]:
        cfg = ForecastConfig(data_dir=str(dataset_dir), **payload.get("params", {}))
        res = run_forecast(cfg)
        return {
            "winner": res.get("winner"),
            "selected_baseline": res.get("selected_baseline"),
            "promotion": res.get("promotion"),
            "submission_filename": Path(str(res.get("submission_path"))).name,
        }

    return _run_dataset_job(payload["dataset_id"], run)


def _price_actions_impl(payload: dict[str, Any]) -> dict[str, Any]:
    def run(dataset_dir: Path) -> dict[str, Any]:
        params = dict(payload.get("params", {}))
        if "submission_path" in params:
            params["submission_path"] = resolve_dataset_file(
                dataset_dir, params["submission_path"]
            ).name
        cfg = PriceOptConfig(data_dir=str(dataset_dir), **params)
        res = run_price_optimization(cfg)
        return {"n": res.get("n"), "summary": res.get("summary", {})}

    return _run_dataset_job(payload["dataset_id"], run)


def _promo_selection_impl(payload: dict[str, Any]) -> dict[str, Any]:
    def run(dataset_dir: Path) -> dict[str, Any]:
        params = dict(payload.get("params", {}))
        if "input_path" in params:
            params["input_path"] = resolve_dataset_file(dataset_dir, params["input_path"]).name
        cfg = PromoSelectionConfig(data_dir=str(dataset_dir), **params)
        res = run_promo_selection(cfg)
        return {
            "method": res.get("method"),
            "solver_status": res.get("solver_status"),
            "constraint_report": res.get("constraint_report", {}),
        }

    return _run_dataset_job(payload["dataset_id"], run)


def _business_pack_impl(payload: dict[str, Any]) -> dict[str, Any]:
    def run(dataset_dir: Path) -> dict[str, Any]:
        res = generate_business_pack(BusinessPackConfig(data_dir=str(dataset_dir)))
        return {
            "readiness": res.get("readiness"),
            "exports": sorted(res.get("dashboard_exports", {})),
        }

    return _run_dataset_job(payload["dataset_id"], run)


def _task_wrapper(
    func: Callable[[dict[str, Any]], dict[str, Any]], payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        return func(payload)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        return _safe_failure(exc)


if celery_app is not None:

    @celery_app.task(name="worker.tasks.forecast_task", bind=True)
    def forecast_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.update_state(state="STARTED", meta={"stage": "running"})
        return _task_wrapper(_forecast_impl, payload)

    @celery_app.task(name="worker.tasks.price_actions_task", bind=True)
    def price_actions_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.update_state(state="STARTED", meta={"stage": "running"})
        return _task_wrapper(_price_actions_impl, payload)

    @celery_app.task(name="worker.tasks.promo_selection_task", bind=True)
    def promo_selection_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.update_state(state="STARTED", meta={"stage": "running"})
        return _task_wrapper(_promo_selection_impl, payload)

    @celery_app.task(name="worker.tasks.business_pack_task", bind=True)
    def business_pack_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.update_state(state="STARTED", meta={"stage": "running"})
        return _task_wrapper(_business_pack_impl, payload)

else:

    class _EagerTask:
        def __init__(self, func: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
            self.func = func
            self.name = func.__name__

        def delay(self, payload: dict[str, Any]):
            class Result:
                id = "eager-local"
                status = "SUCCESS"

                def get(self, timeout: int | None = None):
                    return _task_wrapper(self_func, payload)

            self_func = self.func
            return Result()

    forecast_task = _EagerTask(_forecast_impl)
    price_actions_task = _EagerTask(_price_actions_impl)
    promo_selection_task = _EagerTask(_promo_selection_impl)
    business_pack_task = _EagerTask(_business_pack_impl)
