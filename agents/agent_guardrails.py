from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _metric_value(
    metrics: Dict[str, Any], primary_key: str, legacy_key: Optional[str] = None
) -> Optional[float]:
    value = metrics.get(primary_key)
    if value is None and legacy_key is not None:
        value = metrics.get(legacy_key)
    if value is None:
        return None
    return float(value)


def assess_forecast_gate(forecast_res: Dict[str, Any]) -> Tuple[bool, str]:
    """Decide if downstream pricing should continue.

    Uses your existing backtest payload (`backtests[0]`) and winner field.
    """
    winner = str(forecast_res.get("winner", ""))
    backtests = forecast_res.get("backtests", []) or []
    if not backtests:
        return False, "Missing forecast backtests; cannot trust downstream pricing inputs."

    latest = backtests[0]
    wmape_lgbm = _metric_value(latest, "wmape_lgbm")
    if wmape_lgbm is None:
        return False, "Missing wmape_lgbm; cannot evaluate forecast gate."

    baseline_metrics = {
        "mean_28": _metric_value(latest, "wmape_baseline_mean_28"),
        "seasonal_7": _metric_value(latest, "wmape_baseline_seasonal_7", "wmape_baseline_seas_7"),
        "seasonal_364": _metric_value(
            latest, "wmape_baseline_seasonal_364", "wmape_baseline_seas_364"
        ),
    }
    baseline_candidates = [value for value in baseline_metrics.values() if value is not None]
    if not baseline_candidates:
        return False, "Missing forecast baseline WMAPE metrics; cannot evaluate forecast gate."

    best_baseline = min(baseline_candidates)

    if winner == "lgbm":
        return (
            True,
            f"Forecast gate passed: LGBM beats baselines (wmape={wmape_lgbm:.4f} < {best_baseline:.4f}).",
        )
    return True, (
        "Forecast winner is baseline. Continue allowed, but mark pricing outputs as lower-confidence "
        f"(model wmape={wmape_lgbm:.4f}, best baseline={best_baseline:.4f})."
    )


def assess_promo_constraints(promo_res: Dict[str, Any]) -> Tuple[bool, str]:
    rep = promo_res.get("constraint_report", {}) or {}
    if bool(rep.get("any_violation", False)):
        return False, f"Promo selection constraint violation detected: {rep}"
    return True, "Promo constraints satisfied."


def summarise_profit(pres: Dict[str, Any], sres: Dict[str, Any]) -> Dict[str, Any]:
    ps = (pres or {}).get("summary", {}) or {}
    ss = (sres or {}).get("summary", {}) or {}
    return {
        "unconstrained": {
            "base_profit_gbp": ps.get("base_profit_gbp"),
            "optimised_profit_gbp": ps.get("optimised_profit_gbp"),
            "uplift_gbp": (ps.get("uplift_gbp") if "uplift_gbp" in ps else None),
            "uplift_pct": ps.get("uplift_pct"),
        },
        "constrained": {
            "constrained_profit_gbp": ss.get("constrained_profit_gbp"),
            "delta_base_to_constrained_gbp": ss.get("delta_base_to_constrained_gbp"),
            "uplift_constrained_pct": ss.get("uplift_constrained_pct"),
        },
    }
