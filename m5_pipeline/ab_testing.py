from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Any, Tuple, Sequence, Optional, Literal

import json
import hashlib
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


OfflineCounterfactualSimConfig = ABTestSimConfig


def simulate_offline_counterfactual(cfg: ABTestSimConfig) -> Dict[str, Any]:
    """Compatibility-safe public name for the non-causal offline scenario simulator."""
    report = simulate_price_ab_test(cfg)
    report["design"]["evidence_type"] = "offline_counterfactual_simulation"
    report["design"]["causal_validated"] = False
    report["design"]["requires_randomized_validation"] = True
    return report


def assign_stratified_treatment(
    df: pd.DataFrame,
    *,
    strata_cols: Sequence[str] = ("store_id", "cat_id", "baseline_demand_band"),
    treatment_share: float = 0.5,
    seed: int = 42,
    unit_col: str = "unit_id",
) -> pd.DataFrame:
    """Deterministically assign treatment/control within provided strata."""
    _require_cols(df, tuple([unit_col, *strata_cols]))
    if not 0 < float(treatment_share) < 1:
        raise ValueError("treatment_share must be between 0 and 1")
    out = df.copy()
    scores = []
    for row in out[[unit_col, *strata_cols]].astype(str).itertuples(index=False, name=None):
        payload = "|".join(row) + f"|{int(seed)}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        scores.append(int(digest[:12], 16) / float(0xFFFFFFFFFFFF))
    out["_assignment_score"] = scores
    out["assignment"] = "control"
    for _, idx in out.groupby(list(strata_cols), dropna=False, sort=False).groups.items():
        part = out.loc[idx].sort_values("_assignment_score")
        n_treat = int(round(len(part) * float(treatment_share)))
        n_treat = min(max(n_treat, 1 if len(part) > 1 else 0), max(len(part) - 1, 0))
        out.loc[part.head(n_treat).index, "assignment"] = "treatment"
    return out.drop(columns=["_assignment_score"])


def treatment_balance(
    df: pd.DataFrame,
    *,
    assignment_col: str = "assignment",
    covariate_cols: Sequence[str] = ("pre_metric",),
) -> Dict[str, Any]:
    _require_cols(df, tuple([assignment_col, *covariate_cols]))
    groups = set(df[assignment_col].dropna().astype(str))
    if not {"control", "treatment"} <= groups:
        raise ValueError("assignment must contain both control and treatment")
    rows = []
    for col in covariate_cols:
        control = pd.to_numeric(df.loc[df[assignment_col] == "control", col], errors="coerce")
        treatment = pd.to_numeric(df.loc[df[assignment_col] == "treatment", col], errors="coerce")
        pooled_sd = float(
            np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2.0)
            if len(control) > 1 and len(treatment) > 1
            else 0.0
        )
        diff = float(treatment.mean() - control.mean())
        rows.append(
            {
                "covariate": col,
                "control_mean": float(control.mean()),
                "treatment_mean": float(treatment.mean()),
                "standardized_mean_difference": diff / pooled_sd if pooled_sd > 1e-12 else 0.0,
            }
        )
    return {
        "n_control": int((df[assignment_col] == "control").sum()),
        "n_treatment": int((df[assignment_col] == "treatment").sum()),
        "covariates": rows,
    }


def approximate_sample_size_per_group(
    *,
    baseline_std: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sidedness: Literal["two-sided", "one-sided"] = "two-sided",
) -> Dict[str, Any]:
    if baseline_std <= 0 or minimum_detectable_effect <= 0:
        raise ValueError("baseline_std and minimum_detectable_effect must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < float(power) < 1.0:
        raise ValueError("power must be between 0 and 1")
    if sidedness not in {"two-sided", "one-sided"}:
        raise ValueError("sidedness must be 'two-sided' or 'one-sided'")
    normal = NormalDist()
    alpha_tail = float(alpha) / 2.0 if sidedness == "two-sided" else float(alpha)
    z_alpha = normal.inv_cdf(1.0 - alpha_tail)
    z_power = normal.inv_cdf(float(power))
    n = int(np.ceil(2 * ((z_alpha + z_power) * baseline_std / minimum_detectable_effect) ** 2))
    return {
        "sample_size_per_group": n,
        "assumptions": {
            "alpha": float(alpha),
            "power": float(power),
            "sidedness": sidedness,
            "baseline_std": float(baseline_std),
            "minimum_detectable_effect": float(minimum_detectable_effect),
            "method": "normal_approximation_two_sample_means",
        },
    }


def analyse_randomized_price_experiment(
    df: pd.DataFrame,
    *,
    assignment_col: str = "assignment",
    unit_col: str = "unit_id",
    pre_metric_col: str = "pre_metric",
    post_metric_col: str = "post_metric",
    guardrail_cols: Optional[Sequence[str]] = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    required = [assignment_col, unit_col, pre_metric_col, post_metric_col]
    if guardrail_cols:
        required.extend(guardrail_cols)
    _require_cols(df, tuple(required))
    groups = set(df[assignment_col].dropna().astype(str))
    if not {"control", "treatment"} <= groups:
        raise ValueError("Observed experiment data must include both treatment and control groups")

    work = df.copy()
    work["_change"] = pd.to_numeric(work[post_metric_col], errors="coerce") - pd.to_numeric(
        work[pre_metric_col], errors="coerce"
    )
    control = work.loc[work[assignment_col] == "control", "_change"].dropna().to_numpy(float)
    treatment = work.loc[work[assignment_col] == "treatment", "_change"].dropna().to_numpy(float)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("Both groups must have observed pre/post metric changes")
    estimate = float(treatment.mean() - control.mean())

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        c = rng.choice(control, size=control.size, replace=True)
        t = rng.choice(treatment, size=treatment.size, replace=True)
        boot.append(float(t.mean() - c.mean()))
    ci = {
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
    }

    combined = np.concatenate([control, treatment])
    n_t = treatment.size
    more_extreme = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(combined)
        diff = float(perm[:n_t].mean() - perm[n_t:].mean())
        if abs(diff) >= abs(estimate):
            more_extreme += 1
    p_value = float((more_extreme + 1) / (int(n_perm) + 1))

    guardrails = {}
    for col in guardrail_cols or []:
        guardrails[col] = treatment_balance(
            work, assignment_col=assignment_col, covariate_cols=[col]
        )

    return {
        "evidence_type": "randomized_experiment",
        "causal_validated": False,
        "causal_validation_note": (
            "Treatment/control labels are present, but this function cannot verify assignment "
            "integrity, interference, or experiment execution from labels alone."
        ),
        "estimator": "difference_in_mean_pre_post_changes",
        "estimate": estimate,
        "bootstrap_ci_95": ci,
        "permutation_p_value": p_value,
        "n_control": int(control.size),
        "n_treatment": int(treatment.size),
        "guardrails": guardrails,
    }
