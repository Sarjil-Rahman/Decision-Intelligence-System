from __future__ import annotations

from typing import Dict, Any, Optional, List

from m5_pipeline.m5_forecasting import ForecastConfig, run_forecast
from m5_pipeline.m5_price_optimization import PriceOptConfig, run_price_optimization
from m5_pipeline.m5_promo_selection import PromoSelectionConfig, run_promo_selection
from m5_pipeline.validation import ValidationConfig, validate_m5_inputs
from m5_pipeline.business_outputs import BusinessPackConfig, generate_business_pack


def forecast_point_or_quantiles(
    data_dir: str,
    *,
    max_series: int = 0,
    start_d: str = "d_1500",
    last_train_d: str = "d_1913",
    horizon: int = 28,
    objective: str = "tweedie",
    tweedie_variance_power: float = 1.1,
    two_stage: bool = True,
    split_strategy: str = "rolling_origin",
    n_backtests: int = 3,
    backtest_stride: int = 28,
    validate_inputs: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    if validate_inputs:
        validate_m5_inputs(ValidationConfig(data_dir=data_dir, strict=True, write_reports=True))

    cfg = ForecastConfig(
        data_dir=data_dir,
        max_series=max_series,
        start_d=start_d,
        last_train_d=last_train_d,
        horizon=horizon,
        objective=objective,  # type: ignore[arg-type]
        tweedie_variance_power=tweedie_variance_power,
        two_stage=two_stage,
        split_strategy=split_strategy,  # type: ignore[arg-type]
        n_backtests=n_backtests,
        backtest_stride=backtest_stride,
        save_artifacts=save_artifacts,
        validate_inputs=False,  # already validated above
        out_submission="submission.csv",
    )

    fres = run_forecast(cfg)

    ci = {
        "q10": float(fres.get("residual_q10", 0.0)),
        "q50": float(fres.get("residual_q50", 0.0)),
        "q90": float(fres.get("residual_q90", 0.0)),
    }

    return {
        "submission_path": fres["submission_path"],
        "winner": fres["winner"],
        "selected_baseline": fres.get("selected_baseline"),
        "promotion": fres.get("promotion", {}),
        "prediction_intervals": fres.get("prediction_intervals", {}),
        "backtests": fres["backtests"],
        "residual_quantiles": ci,
        "artifacts": fres.get("artifacts", {}),
    }


def price_actions(
    data_dir: str,
    *,
    submission_path: str = "submission.csv",
    last_train_d: str = "d_1913",
    margin: float = 0.30,
    unit_econ_path: Optional[str] = None,
    max_series: int = 0,
    lookback_days: int = 365,
    elasticity_clip_low: float = -5.0,
    elasticity_clip_high: float = -0.1,
    max_abs_price_change_pct: float = 0.20,
    max_demand_mult: float = 3.0,
    suspicious_profit_gain_pct: float = 400.0,
    suspicious_demand_gain_pct: float = 500.0,
) -> Dict[str, Any]:
    cfg = PriceOptConfig(  # ✅ fixed name
        data_dir=data_dir,
        submission_path=submission_path,
        last_train_d=last_train_d,
        margin=margin,
        unit_econ_path=unit_econ_path,
        max_series=max_series,
        lookback_days=lookback_days,
        elasticity_clip=(float(elasticity_clip_low), float(elasticity_clip_high)),
        max_abs_price_change_pct=float(max_abs_price_change_pct),
        max_demand_mult=float(max_demand_mult),
        suspicious_profit_gain_pct=float(suspicious_profit_gain_pct),
        suspicious_demand_gain_pct=float(suspicious_demand_gain_pct),
        write_reports=True,
    )
    return run_price_optimization(cfg)


def promo_selection(
    data_dir: str,
    *,
    input_path: str = "price_optimization_results.csv",
    max_price_changes_total: Optional[int] = 5000,
    max_price_changes_per_store: Optional[int] = 800,
    max_price_changes_per_cat: Optional[int] = 1200,
    budget: Optional[float] = None,
    forbid_price_increase: bool = True,
    max_abs_price_change_pct: float = 0.20,
    max_demand_mult: float = 3.0,
    objective: str = "profit",
    require_price_change: bool = True,
    promo_discount_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    cfg = PromoSelectionConfig(
        data_dir=data_dir,
        input_path=input_path,
        max_price_changes_total=max_price_changes_total,
        max_price_changes_per_store=max_price_changes_per_store,
        max_price_changes_per_cat=max_price_changes_per_cat,
        budget=budget,
        forbid_price_increase=forbid_price_increase,
        max_abs_price_change_pct=float(max_abs_price_change_pct),
        max_demand_mult=float(max_demand_mult),
        objective=objective,  # type: ignore[arg-type]
        require_price_change=require_price_change,
        promo_discount_grid=(
            tuple(promo_discount_grid) if promo_discount_grid is not None else (-0.20, -0.10, -0.05)
        ),
        write_reports=True,
    )
    return run_promo_selection(cfg)


def simulate_ab_test(
    price_actions_csv: str,
    out_report_json: str = "./reports/ab_test_simulation.json",
    treatment_share: float = 0.5,
    noise_sigma: float = 0.10,
    elasticity_col: str = "elasticity",
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """
    Deprecated internal offline counterfactual simulator wrapper.

    Reads the optimizer output CSV and generates an uplift distribution + bootstrap CI.
    """
    from m5_pipeline.ab_testing import ABTestSimConfig, simulate_offline_counterfactual

    cfg = ABTestSimConfig(
        price_actions_csv=price_actions_csv,
        out_report_json=out_report_json,
        treatment_share=treatment_share,
        noise_sigma=noise_sigma,
        elasticity_col=elasticity_col,
        n_boot=n_boot,
        seed=seed,
    )
    return simulate_offline_counterfactual(cfg)


def generate_business_reporting_pack(data_dir: str) -> Dict[str, Any]:
    return generate_business_pack(BusinessPackConfig(data_dir=data_dir))
