from __future__ import annotations

from api.settings import get_settings

try:
    from celery import Celery
except ImportError:  # pragma: no cover - lets unit tests run before optional deps are installed
    Celery = None  # type: ignore


settings = get_settings()

if Celery is not None:
    celery_app = Celery(
        "sql_forecasting",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["worker.tasks"],
    )
    celery_app.conf.task_always_eager = bool(settings.celery_task_always_eager)
    celery_app.conf.task_store_eager_result = True
else:
    celery_app = None
