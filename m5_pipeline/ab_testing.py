from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np
import pandas as pd


@dataclass
class ABTestSimConfig:
    """
    Offline A/B-style simulator for pricing changes.

    This is NOT a real online experiment. It is a paired counterfactual simulation to:
    - sanity-check pricing recommendations
    - quantify expected rollout uplift
    - estimate uncertainty bands via bootstrap

    Inputs are typically the optimiser outputs:
    - a CSV with baseline vs recommended price and baseline demand
    """

    price_actions_csv: str
    out_report_json: str = "./reports/ab_test_simulation.json"

    # Experiment design metadata
    treatment_share: float = 0.5
    unit: str = "store_item"

    # Demand response model
    noise_sigma: float = 0.10
    elasticity_col: str = "elasticity"

    # Metrics
    n_boot: int = 500
    seed: int = 42


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Present columns: {list(df.columns)}"
        )


def _safe_pct_uplift(treatment_value: float, control_value: float) -> float:
    treatment_value = float(treatment_value)
    control_value = float(control_value)
    if abs(treatment_value - control_value) <= 1e-12:
        return 0.0
    if abs(control_value) <= 1e-12:
        return float("nan")
    return float((treatment_value / control_value - 1.0) * 100.0)


def _ci_from_samples(samples: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(samples, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "p05": float(np.quantile(vals, 0.05)),
        "p50": float(np.quantile(vals, 0.50)),
        "p95": float(np.quantile(vals, 0.95)),
    }


def simulate_price_ab_test(cfg: ABTestSimConfig) -> Dict[str, Any]:
    """
    Simulate control and treatment outcomes for the SAME units using shared demand shocks.

    Why this estimator is used:
    - the optimiser recommends actions per row/unit, so a paired counterfactual is much
      more stable than randomly splitting a small heterogeneous set of rows into arms
    - percent uplift reflects the expected full-rollout effect
    - treatment_share is still reported and used to derive a simple exposure-scaled lift
      for a partial experiment, but it does not distort the core uplift estimate
    """
    rng = np.random.default_rng(cfg.seed)

    df = pd.read_csv(cfg.price_actions_csv)

    col_map = {
        "base_price": "price",
        "new_price": "best_price",
        "base_demand": "base_demand_28d",
    }
    for want, have in col_map.items():
        if want not in df.columns and have in df.columns:
            df[want] = df[have]

    if "cost" not in df.columns:
        df["cost"] = 0.7 * df["base_price"]

    if cfg.elasticity_col not in df.columns:
        df[cfg.elasticity_col] = -1.5

    _require_cols(df, ("base_price", "new_price", "base_demand", "cost", cfg.elasticity_col))

    n = int(len(df))
    if n < 2:
        raise ValueError("A/B simulation needs at least 2 rows.")

    base_price = df["base_price"].astype(np.float64).to_numpy()
    new_price = df["new_price"].astype(np.float64).to_numpy()
    cost = df["cost"].astype(np.float64).to_numpy()
    base_demand = df["base_demand"].astype(np.float64).to_numpy()
    elasticity = df[cfg.elasticity_col].astype(np.float64).to_numpy()

    # Expected/noiseless point estimate, then bootstrap uncertainty with fresh shared shocks.
    shared_mult = np.ones(n, dtype=np.float64)

    q_control = np.clip(base_demand * shared_mult, 0.0, None)
    treat_ratio = np.divide(
        new_price,
        base_price,
        out=np.ones_like(new_price, dtype=np.float64),
        where=np.abs(base_price) > 1e-12,
    )
    treat_ratio = np.where(
        np.isclose(new_price, base_price, atol=1e-12, rtol=0.0), 1.0, treat_ratio
    )
    q_treatment = np.clip(base_demand * np.power(treat_ratio, elasticity) * shared_mult, 0.0, None)

    revenue_control = base_price * q_control
    revenue_treatment = new_price * q_treatment
    profit_control = (base_price - cost) * q_control
    profit_treatment = (new_price - cost) * q_treatment

    point = {
        "demand": {
            "control": float(q_control.sum()),
            "treatment": float(q_treatment.sum()),
        },
        "revenue": {
            "control": float(revenue_control.sum()),
            "treatment": float(revenue_treatment.sum()),
        },
        "profit": {
            "control": float(profit_control.sum()),
            "treatment": float(profit_treatment.sum()),
        },
    }

    uplift_pct = {
        metric: _safe_pct_uplift(vals["treatment"], vals["control"])
        for metric, vals in point.items()
    }
    uplift_absolute = {
        metric: float(vals["treatment"] - vals["control"]) for metric, vals in point.items()
    }
    exposure_scaled_absolute = {
        metric: float(float(cfg.treatment_share) * val) for metric, val in uplift_absolute.items()
    }

    base_matrix = np.column_stack(
        [
            base_price,
            new_price,
            cost,
            base_demand,
            treat_ratio,
            elasticity,
        ]
    )

    pct_samples = {"demand": [], "revenue": [], "profit": []}
    abs_samples = {"demand": [], "revenue": [], "profit": []}

    for _ in range(int(cfg.n_boot)):
        idx = rng.integers(0, n, size=n)
        boot = base_matrix[idx]
        bp = boot[:, 0]
        np_ = boot[:, 1]
        c = boot[:, 2]
        bd = boot[:, 3]
        tr = boot[:, 4]
        el = boot[:, 5]

        noise = rng.normal(loc=0.0, scale=float(cfg.noise_sigma), size=len(boot))
        mult = np.clip(1.0 + noise, 0.0, None)

        q_c_arr = np.clip(bd * mult, 0.0, None)
        q_t_arr = np.clip(bd * np.power(tr, el) * mult, 0.0, None)

        q_c, q_t = q_c_arr.sum(), q_t_arr.sum()
        r_c, r_t = (bp * q_c_arr).sum(), (np_ * q_t_arr).sum()
        p_c, p_t = ((bp - c) * q_c_arr).sum(), ((np_ - c) * q_t_arr).sum()

        pct_samples["demand"].append(_safe_pct_uplift(q_t, q_c))
        pct_samples["revenue"].append(_safe_pct_uplift(r_t, r_c))
        pct_samples["profit"].append(_safe_pct_uplift(p_t, p_c))

        abs_samples["demand"].append(float(q_t - q_c))
        abs_samples["revenue"].append(float(r_t - r_c))
        abs_samples["profit"].append(float(p_t - p_c))

    report = {
        "design": {
            "treatment_share": float(cfg.treatment_share),
            "unit": cfg.unit,
            "noise_sigma": float(cfg.noise_sigma),
            "elasticity_col": cfg.elasticity_col,
            "n_rows": n,
            "estimator": "paired_counterfactual_bootstrap",
        },
        "aggregate": {
            "control": {
                "n_units": n,
                "demand": point["demand"]["control"],
                "revenue": point["revenue"]["control"],
                "profit": point["profit"]["control"],
                "avg_per_unit": {
                    "demand": float(q_control.mean()),
                    "revenue": float(revenue_control.mean()),
                    "profit": float(profit_control.mean()),
                },
            },
            "treatment": {
                "n_units": n,
                "demand": point["demand"]["treatment"],
                "revenue": point["revenue"]["treatment"],
                "profit": point["profit"]["treatment"],
                "avg_per_unit": {
                    "demand": float(q_treatment.mean()),
                    "revenue": float(revenue_treatment.mean()),
                    "profit": float(profit_treatment.mean()),
                },
            },
            "exposure_scaled_absolute_uplift": exposure_scaled_absolute,
        },
        "uplift_pct": {
            **uplift_pct,
            "demand_ci_bootstrap": _ci_from_samples(
                np.asarray(pct_samples["demand"], dtype=np.float64)
            ),
            "revenue_ci_bootstrap": _ci_from_samples(
                np.asarray(pct_samples["revenue"], dtype=np.float64)
            ),
            "profit_ci_bootstrap": _ci_from_samples(
                np.asarray(pct_samples["profit"], dtype=np.float64)
            ),
        },
        "uplift_absolute": {
            **uplift_absolute,
            "demand_ci_bootstrap": _ci_from_samples(
                np.asarray(abs_samples["demand"], dtype=np.float64)
            ),
            "revenue_ci_bootstrap": _ci_from_samples(
                np.asarray(abs_samples["revenue"], dtype=np.float64)
            ),
            "profit_ci_bootstrap": _ci_from_samples(
                np.asarray(abs_samples["profit"], dtype=np.float64)
            ),
        },
        "notes": [
            "This is an OFFLINE simulation, not an online randomized experiment.",
            "Control and treatment are simulated for the same units using shared demand shocks, which reduces composition noise.",
            "Percent uplift reflects the expected full-rollout effect implied by the recommended prices.",
            "Exposure-scaled absolute uplift applies treatment_share to the full-rollout lift as a rough partial-rollout proxy.",
            "Interpret uplift as directional and for risk screening; validate online before roll-out.",
        ],
    }

    out_path = Path(cfg.out_report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
