from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status

from api.settings import ApiSettings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: ApiSettings = Depends(get_settings),
) -> None:
    env = settings.api_env.lower()
    if env in {"development", "dev", "test", "testing"} and not settings.api_key:
        return
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key is required outside development/test mode.",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
