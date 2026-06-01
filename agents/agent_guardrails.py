from __future__ import annotations

from typing import Any, Dict, Tuple


def assess_forecast_gate(forecast_res: Dict[str, Any]) -> Tuple[bool, str]:
    """Decide if downstream pricing should continue.

    Uses your existing backtest payload (`backtests[0]`) and winner field.
    """
    winner = str(forecast_res.get("winner", ""))
    backtests = forecast_res.get("backtests", []) or []
    if not backtests:
        return False, "Missing forecast backtests; cannot trust downstream pricing inputs."

    latest = backtests[0]
    wmape_lgbm = float(latest.get("wmape_lgbm", float("inf")))
    baseline_candidates = [
        float(latest.get("wmape_baseline_mean_28", float("inf"))),
        float(latest.get("wmape_baseline_seas_7", float("inf"))),
        float(latest.get("wmape_baseline_seas_364", float("inf"))),
    ]
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
