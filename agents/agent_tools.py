from __future__ import annotations

import time
from typing import Any, Dict, Optional, List

# These imports assume your project package layout:
# api/services.py, m5_pipeline/validation.py, m5_pipeline/m5_price_optimization.py, drift_check.py
from api.services import forecast_point_or_quantiles, price_actions, promo_selection
from m5_pipeline.validation import ValidationConfig, validate_m5_inputs
from m5_pipeline.m5_price_optimization import backtest_price_uplift


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    return out, dt_ms


def run_validate(data_dir: str) -> tuple[Dict[str, Any], int]:
    return _timed(
        validate_m5_inputs, ValidationConfig(data_dir=data_dir, strict=True, write_reports=True)
    )


def run_forecast(data_dir: str, params: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    return _timed(
        forecast_point_or_quantiles,
        data_dir=data_dir,
        max_series=int(params.get("max_series", 0)),
        start_d=str(params.get("start_d", "d_1500")),
        last_train_d=str(params.get("last_train_d", "d_1913")),
        horizon=int(params.get("horizon", 28)),
        objective=str(params.get("objective", "tweedie")),
        tweedie_variance_power=float(params.get("tweedie_variance_power", 1.1)),
        two_stage=bool(params.get("two_stage", True)),
        split_strategy=str(params.get("split_strategy", "rolling_origin")),
        n_backtests=int(params.get("n_backtests", 3)),
        backtest_stride=int(params.get("backtest_stride", 28)),
        validate_inputs=False,
        save_artifacts=bool(params.get("save_artifacts", True)),
    )


def run_price_actions(data_dir: str, params: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    return _timed(
        price_actions,
        data_dir=data_dir,
        submission_path=str(params.get("submission_path", "submission.csv")),
        last_train_d=str(params.get("last_train_d", "d_1913")),
        margin=float(params.get("margin", 0.30)),
        unit_econ_path=params.get("unit_econ_path"),
        max_series=int(params.get("max_series", 0)),
        lookback_days=int(params.get("lookback_days", 365)),
        elasticity_clip_low=float(params.get("elasticity_clip_low", -5.0)),
        elasticity_clip_high=float(params.get("elasticity_clip_high", -0.1)),
        max_abs_price_change_pct=float(params.get("max_abs_price_change_pct", 0.20)),
        max_demand_mult=float(params.get("max_demand_mult", 3.0)),
        suspicious_profit_gain_pct=float(params.get("suspicious_profit_gain_pct", 400.0)),
        suspicious_demand_gain_pct=float(params.get("suspicious_demand_gain_pct", 500.0)),
    )


def run_promo_selection(data_dir: str, params: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    return _timed(
        promo_selection,
        data_dir=data_dir,
        input_path=str(params.get("input_path", "price_optimization_results.csv")),
        max_price_changes_total=params.get("max_price_changes_total", 5000),
        max_price_changes_per_store=params.get("max_price_changes_per_store", 800),
        max_price_changes_per_cat=params.get("max_price_changes_per_cat", 1200),
        budget=params.get("budget"),
        forbid_price_increase=bool(params.get("forbid_price_increase", True)),
        max_abs_price_change_pct=float(params.get("max_abs_price_change_pct", 0.20)),
        max_demand_mult=float(params.get("max_demand_mult", 3.0)),
        objective=str(params.get("promo_objective", "profit")),
        require_price_change=bool(params.get("require_price_change", True)),
        promo_discount_grid=list(params.get("promo_discount_grid", [-0.2, -0.1, -0.05])),
    )


def run_uplift_backtest(
    data_dir: str, params: Dict[str, Any], cutoffs: Optional[List[str]] = None
) -> tuple[Dict[str, Any], int]:
    if cutoffs is None:
        cutoffs = list(params.get("uplift_cutoffs", [])) or ["d_1800", "d_1856", "d_1912"]
    return _timed(
        backtest_price_uplift,
        data_dir=data_dir,
        cutoffs=cutoffs,
        horizon=int(params.get("horizon", 28)),
        max_series=int(params.get("max_series", 0)),
        margin=float(params.get("margin", 0.30)),
        unit_econ_path=params.get("unit_econ_path"),
        lookback_days=int(params.get("lookback_days", 365)),
        elasticity_clip=(
            float(params.get("elasticity_clip_low", -5.0)),
            float(params.get("elasticity_clip_high", -0.1)),
        ),
        max_abs_price_change_pct=float(params.get("max_abs_price_change_pct", 0.20)),
        max_demand_mult=float(params.get("max_demand_mult", 3.0)),
    )
