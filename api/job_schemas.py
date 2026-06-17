from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastJobParams(_StrictModel):
    max_series: int = Field(default=0, ge=0, le=100_000)
    start_d: str = Field(default="d_1500", pattern=r"^d_[0-9]+$")
    last_train_d: str = Field(default="d_1913", pattern=r"^d_[0-9]+$")
    horizon: int = Field(default=28, ge=1, le=56)
    objective: Literal["tweedie", "poisson"] = "tweedie"
    tweedie_variance_power: float = Field(default=1.1, ge=1.0, le=1.9)
    two_stage: bool = True
    split_strategy: Literal["rolling_origin", "last_window"] = "rolling_origin"
    n_backtests: int = Field(default=3, ge=1, le=10)
    backtest_stride: int = Field(default=28, ge=7, le=56)
    validate_inputs: bool = True
    save_artifacts: bool = True
    n_jobs: int = Field(default=1, ge=1, le=8)
    n_estimators: int | None = Field(default=None, ge=10, le=4_000)
    classifier_n_estimators: int = Field(default=500, ge=10, le=2_000)
    regressor_n_estimators: int | None = Field(default=None, ge=10, le=4_000)
    classifier_early_stopping_rounds: int = Field(default=50, ge=0, le=500)
    regressor_early_stopping_rounds: int | None = Field(default=None, ge=0, le=500)
    lightgbm_verbosity: int = Field(default=-1, ge=-1, le=2)
    random_state: int = Field(default=42, ge=0, le=2_147_483_647)


class PriceActionsJobParams(_StrictModel):
    last_train_d: str = Field(default="d_1913", pattern=r"^d_[0-9]+$")
    margin: float = Field(default=0.30, ge=0.0, le=0.99)
    max_series: int = Field(default=0, ge=0, le=100_000)
    lookback_days: int = Field(default=365, ge=30, le=2_000)
    elasticity_clip_low: float = Field(default=-5.0, ge=-20.0, lt=0.0)
    elasticity_clip_high: float = Field(default=-0.1, gt=-20.0, lt=0.0)
    max_abs_price_change_pct: float = Field(default=0.20, ge=0.0, le=0.80)
    max_demand_mult: float = Field(default=3.0, ge=1.0, le=20.0)
    suspicious_profit_gain_pct: float = Field(default=400.0, ge=50.0, le=5_000.0)
    suspicious_demand_gain_pct: float = Field(default=500.0, ge=50.0, le=5_000.0)


class PromoSelectionJobParams(_StrictModel):
    max_price_changes_total: int | None = Field(default=5_000, ge=0, le=100_000)
    max_price_changes_per_store: int | None = Field(default=800, ge=0, le=100_000)
    max_price_changes_per_cat: int | None = Field(default=1_200, ge=0, le=100_000)
    budget: float | None = Field(default=None, ge=0.0)
    forbid_price_increase: bool = True
    max_abs_price_change_pct: float = Field(default=0.20, ge=0.0, le=0.80)
    max_demand_mult: float = Field(default=3.0, ge=1.0, le=20.0)
    objective: Literal["profit", "demand"] = "profit"
    require_price_change: bool = True
    promo_discount_grid: tuple[float, ...] = Field(default=(-0.20, -0.10, -0.05), min_length=1)
    solver_time_limit_seconds: int = Field(default=60, ge=1, le=600)
    allow_greedy_fallback: bool = False


class BusinessPackJobParams(_StrictModel):
    pass


class _JobSubmitRequest(_StrictModel):
    dataset_id: str = Field(..., min_length=1, max_length=64)


class ForecastJobSubmitRequest(_JobSubmitRequest):
    params: ForecastJobParams = Field(default_factory=ForecastJobParams)


class PriceActionsJobSubmitRequest(_JobSubmitRequest):
    params: PriceActionsJobParams = Field(default_factory=PriceActionsJobParams)


class PromoSelectionJobSubmitRequest(_JobSubmitRequest):
    params: PromoSelectionJobParams = Field(default_factory=PromoSelectionJobParams)


class BusinessPackJobSubmitRequest(_JobSubmitRequest):
    params: BusinessPackJobParams = Field(default_factory=BusinessPackJobParams)


def payload_from_request(req: _JobSubmitRequest) -> dict[str, Any]:
    params = getattr(req, "params", None)
    return {
        "dataset_id": req.dataset_id,
        "params": params.model_dump(exclude_none=True) if params is not None else {},
    }
