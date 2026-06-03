from __future__ import annotations
import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    data_dir: str = Field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    max_series: int = Field(default=0, ge=0)
    start_d: str = Field(default="d_1500")
    last_train_d: str = Field(default="d_1913")
    horizon: int = Field(default=28, ge=1, le=56)

    # Model config
    objective: Literal["tweedie", "poisson"] = "tweedie"
    tweedie_variance_power: float = Field(default=1.1, ge=1.0, le=1.9)
    two_stage: bool = True

    # Production evaluation config
    split_strategy: Literal["rolling_origin", "last_window"] = "rolling_origin"
    n_backtests: int = Field(default=3, ge=1, le=10)
    backtest_stride: int = Field(default=28, ge=7, le=56)

    # Safety / ops
    validate_inputs: bool = True
    save_artifacts: bool = True


class ForecastResponse(BaseModel):
    submission_path: str
    winner: str
    backtests: List[Dict[str, Any]]

    # Convenience: latest residual quantiles for confidence intervals
    residual_q10: float
    residual_q50: float
    residual_q90: float

    artifacts: Dict[str, Any]


class PriceActionsRequest(BaseModel):
    data_dir: str = Field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    submission_path: str = Field(default="submission.csv")
    last_train_d: str = Field(default="d_1913")

    margin: float = Field(default=0.30, ge=0.0, le=0.99)
    unit_econ_path: Optional[str] = Field(
        default=None, description="Optional costs/margins CSV relative to data-dir"
    )

    max_series: int = Field(default=0, ge=0)
    lookback_days: int = Field(default=365, ge=30, le=2000)

    elasticity_clip_low: float = Field(default=-5.0)
    elasticity_clip_high: float = Field(default=-0.1)

    max_abs_price_change_pct: float = Field(default=0.20, ge=0.0, le=0.80)
    max_demand_mult: float = Field(default=3.0, ge=1.0, le=20.0)

    # suspicious uplift flags
    suspicious_profit_gain_pct: float = Field(default=400.0, ge=50.0, le=5000.0)
    suspicious_demand_gain_pct: float = Field(default=500.0, ge=50.0, le=5000.0)


class PriceActionsResponse(BaseModel):
    opt_path: str
    n: int
    summary: Dict[str, Any] = Field(default_factory=dict)
    reports: Dict[str, Any]
    limitations: List[str]


class PromoSelectionRequest(BaseModel):
    data_dir: str = Field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    input_path: str = Field(default="price_optimization_results.csv")

    max_price_changes_total: Optional[int] = Field(default=5000, ge=0)
    max_price_changes_per_store: Optional[int] = Field(default=800, ge=0)
    max_price_changes_per_cat: Optional[int] = Field(default=1200, ge=0)
    budget: Optional[float] = Field(default=None, ge=0.0)

    forbid_price_increase: bool = True

    max_abs_price_change_pct: float = Field(default=0.20, ge=0.0, le=0.80)
    max_demand_mult: float = Field(default=3.0, ge=1.0, le=20.0)

    objective: Literal["profit", "demand"] = "profit"
    require_price_change: bool = True
    promo_discount_grid: List[float] = Field(default_factory=lambda: [-0.20, -0.10, -0.05])


class PromoSelectionResponse(BaseModel):
    promo_path: str
    method: str
    n: int
    constraint_report: Dict[str, Any]
    summary: Dict[str, Any] = Field(default_factory=dict)
    reports: Dict[str, Any]


class ABTestSimRequest(BaseModel):
    price_actions_csv: str = Field(
        ..., description="Path to price_actions.csv produced by optimizer"
    )
    out_report_json: str = Field(
        "./reports/ab_test_simulation.json", description="Where to write the simulation report"
    )
    treatment_share: float = Field(0.5, ge=0.05, le=0.95)
    noise_sigma: float = Field(0.10, ge=0.0, le=1.0)
    elasticity_col: str = Field("elasticity")
    n_boot: int = Field(500, ge=50, le=5000)
    seed: int = Field(42)


class ABTestSimResponse(BaseModel):
    report: dict


class BusinessPackRequest(BaseModel):
    data_dir: str = Field(default_factory=lambda: os.getenv("DATA_DIR", "data"))


class BusinessPackResponse(BaseModel):
    executive_kpi_summary_json: str
    reason_coded_action_recommendations_csv: str
    scenario_comparison_csv: str
    dashboard_exports: Dict[str, str]
    docs: Dict[str, str]
    readiness: Optional[str] = None
