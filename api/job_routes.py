from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.datasets import resolve_dataset_dir, validate_required_dataset_files
from api.security import require_api_key
from api.settings import ApiSettings, get_settings
from worker import tasks
from worker.celery_app import celery_app

try:
    from celery.result import AsyncResult
except ImportError:  # pragma: no cover
    AsyncResult = None  # type: ignore


router = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])
_EAGER_RESULTS: dict[str, dict[str, Any]] = {}


class JobSubmitRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    submitted_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    failure_message: str | None = None


def _validate_submission(req: JobSubmitRequest, settings: ApiSettings) -> None:
    dataset_dir = resolve_dataset_dir(settings.data_root, req.dataset_id)
    validate_required_dataset_files(dataset_dir)


def _submit(task, req: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    settings = get_settings()
    try:
        _validate_submission(req, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    payload = {"dataset_id": req.dataset_id, "params": req.params}
    submitted_at = datetime.now(timezone.utc).isoformat()
    if settings.celery_task_always_eager:
        impl_by_name = {
            "worker.tasks.forecast_task": tasks._forecast_impl,
            "worker.tasks.price_actions_task": tasks._price_actions_impl,
            "worker.tasks.promo_selection_task": tasks._promo_selection_impl,
            "worker.tasks.business_pack_task": tasks._business_pack_impl,
            "forecast_task": tasks._forecast_impl,
            "price_actions_task": tasks._price_actions_impl,
            "promo_selection_task": tasks._promo_selection_impl,
            "business_pack_task": tasks._business_pack_impl,
        }
        job_id = f"eager-{uuid4().hex}"
        impl = impl_by_name.get(getattr(task, "name", ""))
        _EAGER_RESULTS[job_id] = (
            tasks._task_wrapper(impl, payload) if impl is not None else {"status": "failed"}
        )
        _EAGER_RESULTS[job_id]["submitted_at"] = submitted_at
        return JobSubmitResponse(
            job_id=job_id,
            status="queued",
            status_url=str(request.url_for("get_job_status", job_id=job_id)),
            submitted_at=submitted_at,
        )
    if celery_app is not None:
        celery_app.conf.task_always_eager = bool(settings.celery_task_always_eager)
        celery_app.conf.task_store_eager_result = True
    result = task.delay(payload)
    return JobSubmitResponse(
        job_id=str(result.id),
        status="queued",
        status_url=str(request.url_for("get_job_status", job_id=str(result.id))),
        submitted_at=submitted_at,
    )


@router.post("/forecast", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_forecast(req: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    return _submit(tasks.forecast_task, req, request)


@router.post(
    "/price-actions", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED
)
def submit_price_actions(req: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    return _submit(tasks.price_actions_task, req, request)


@router.post(
    "/promo-selection", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED
)
def submit_promo_selection(req: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    return _submit(tasks.promo_selection_task, req, request)


@router.post(
    "/business-pack", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED
)
def submit_business_pack(req: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    return _submit(tasks.business_pack_task, req, request)


@router.get("/{job_id}", response_model=JobStatusResponse, name="get_job_status")
def get_job_status(job_id: str) -> JobStatusResponse:
    if job_id in _EAGER_RESULTS:
        payload = _EAGER_RESULTS[job_id]
        if payload.get("status") == "failed":
            return JobStatusResponse(
                job_id=job_id,
                status="failed",
                submitted_at=payload.get("submitted_at"),
                finished_at=payload.get("finished_at"),
                failure_message=payload.get("message", "Task failed."),
            )
        return JobStatusResponse(
            job_id=job_id,
            status="succeeded",
            submitted_at=payload.get("submitted_at"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            result=payload.get("result"),
        )
    if AsyncResult is None:
        if job_id == "eager-local":
            return JobStatusResponse(job_id=job_id, status="succeeded")
        raise HTTPException(status_code=404, detail="Job backend is unavailable.")
    result = AsyncResult(job_id)
    state = str(result.state).upper()
    if state in {"PENDING", "RECEIVED"}:
        return JobStatusResponse(job_id=job_id, status="queued")
    if state in {"STARTED", "RETRY"}:
        meta = result.info if isinstance(result.info, dict) else {}
        return JobStatusResponse(job_id=job_id, status="running", stage=meta.get("stage"))
    if state == "SUCCESS":
        payload = result.result if isinstance(result.result, dict) else {}
        if payload.get("status") == "failed":
            return JobStatusResponse(
                job_id=job_id,
                status="failed",
                finished_at=payload.get("finished_at"),
                failure_message=payload.get("message"),
            )
        return JobStatusResponse(
            job_id=job_id,
            status="succeeded",
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            result=payload.get("result"),
        )
    return JobStatusResponse(job_id=job_id, status="failed", failure_message="Task failed.")
