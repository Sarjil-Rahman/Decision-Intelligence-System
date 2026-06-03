from __future__ import annotations

import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, stable_config_hash, write_json


@dataclass
class BusinessPackConfig:
    data_dir: str = "data"
    reports_subdir: str = "reports"
    dashboard_subdir: str = "dashboard_ready"
    docs_subdir: str = "docs"
    dashboard_mockup_path: str = "dashboards/power_bi_tableau/retail_decision_dashboard_mockup.png"


REASON_CODE_REFERENCE: List[Dict[str, Any]] = [
    {
        "reason_code": "APPROVED_PRICE_INCREASE_MARGIN_CAPTURE",
        "reason_group": "approved",
        "priority": "high",
        "meaning": "Raise price because the optimiser projects higher profit with manageable demand decline.",
    },
    {
        "reason_code": "APPROVED_DISCOUNT_VOLUME_DRIVE",
        "reason_group": "approved",
        "priority": "high",
        "meaning": "Discount selected because the constrained plan still improves profit through higher demand.",
    },
    {
        "reason_code": "APPROVED_HOLD_BASELINE",
        "reason_group": "approved",
        "priority": "low",
        "meaning": "Keep the current price because no credible upside cleared the rules.",
    },
    {
        "reason_code": "DEPRIORITISED_BY_CONSTRAINTS",
        "reason_group": "constraint",
        "priority": "medium",
        "meaning": "An action looked useful unconstrained but was not chosen after budgets or capacity limits were applied.",
    },
    {
        "reason_code": "REVIEW_BAD_UNIT_ECON",
        "reason_group": "review",
        "priority": "high",
        "meaning": "Base price is at or below cost proxy, so the recommendation needs human review.",
    },
    {
        "reason_code": "REVIEW_SUSPICIOUS_UPLIFT",
        "reason_group": "review",
        "priority": "high",
        "meaning": "Projected uplift breached credibility guardrails and should not be executed blindly.",
    },
    {
        "reason_code": "REVIEW_GLOBAL_ELASTICITY_FALLBACK",
        "reason_group": "review",
        "priority": "high",
        "meaning": "The item used the global fallback elasticity, so the recommendation is weaker and should be treated as directional.",
    },
    {
        "reason_code": "REVIEW_CATEGORY_ELASTICITY_FALLBACK",
        "reason_group": "review",
        "priority": "medium",
        "meaning": "The item used a category-level fallback elasticity rather than a directly estimated one.",
    },
    {
        "reason_code": "UNCONSTRAINED_PRICE_INCREASE_NOT_EXECUTED",
        "reason_group": "informational",
        "priority": "medium",
        "meaning": "The unconstrained optimiser preferred a price increase, but the execution plan did not apply it.",
    },
    {
        "reason_code": "UNCONSTRAINED_DISCOUNT_NOT_EXECUTED",
        "reason_group": "informational",
        "priority": "medium",
        "meaning": "The unconstrained optimiser preferred a discount, but the execution plan did not apply it.",
    },
    {
        "reason_code": "HOLD_NO_MATERIAL_GAIN",
        "reason_group": "hold",
        "priority": "low",
        "meaning": "No material uplift cleared the decision threshold, so the safest action is to hold.",
    },
]


def _read_json_if_exists(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv_if_exists(path: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _latest_forecast_artifacts_dir(data_dir: str) -> Optional[Path]:
    candidates = sorted(glob(os.path.join(data_dir, "artifacts", "forecast", "*")))
    if not candidates:
        return None
    return Path(candidates[-1])


def _latest_forecast_backtests(data_dir: str) -> List[Dict[str, Any]]:
    latest = _latest_forecast_artifacts_dir(data_dir)
    if latest is None:
        return []
    p = latest / "backtests.json"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def build_reason_coded_actions(
    price_df: pd.DataFrame, promo_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    df = price_df.copy()

    if promo_df is not None and len(promo_df):
        keep = [
            "id",
            "selected",
            "eligible",
            "is_change",
            "applied_price",
            "applied_demand",
            "applied_profit",
            "applied_is_change",
            "new_price",
            "new_demand",
            "new_profit",
            "constraint_violation",
        ]
        keep = [c for c in keep if c in promo_df.columns]
        df = df.merge(promo_df[keep], on="id", how="left")
    else:
        df["selected"] = np.nan
        df["eligible"] = np.nan
        df["applied_price"] = np.nan
        df["applied_profit"] = np.nan
        df["applied_is_change"] = np.nan
        df["constraint_violation"] = np.nan

    df["selected"] = df.get("selected", pd.Series(np.nan, index=df.index)).fillna(0).astype(int)
    if "eligible" in df.columns:
        df["eligible"] = df["eligible"].fillna(0).astype(int)
    else:
        df["eligible"] = 0
    if "applied_is_change" in df.columns:
        df["applied_is_change"] = df["applied_is_change"].fillna(0).astype(int)
    else:
        df["applied_is_change"] = 0

    df["final_price_recommendation"] = np.where(
        df["selected"] == 1,
        df.get("applied_price", df.get("best_price")),
        df.get("best_price", df.get("price")),
    )
    df["final_profit_projection"] = np.where(
        df["selected"] == 1,
        df.get("applied_profit", df.get("best_profit")),
        df.get("best_profit", df.get("base_profit")),
    )

    def classify(row: pd.Series) -> tuple[str, str, str, str, str]:
        price = _float(row.get("price"))
        best_price = _float(row.get("best_price"))
        applied_price = _float(row.get("applied_price"))
        final_price = (
            applied_price
            if row.get("selected", 0) == 1 and np.isfinite(applied_price)
            else best_price
        )
        profit_gain = _float(row.get("profit_gain"))

        if int(row.get("bad_unit_econ", 0)) == 1:
            return (
                "REVIEW_BAD_UNIT_ECON",
                "review",
                "Review cost inputs or margin assumptions before acting.",
                "Cost proxy implies weak or negative economics at the current price.",
                "High",
            )
        if int(row.get("suspicious_uplift", 0)) == 1:
            return (
                "REVIEW_SUSPICIOUS_UPLIFT",
                "review",
                "Treat this as a hypothesis and validate with a controlled test.",
                "Projected uplift tripped a credibility guardrail.",
                "High",
            )
        if str(row.get("elasticity_source", "")) == "global_fallback":
            return (
                "REVIEW_GLOBAL_ELASTICITY_FALLBACK",
                "review",
                "Do not auto-execute; gather better price-response evidence first.",
                "The action relies on a global fallback elasticity rather than item evidence.",
                "High",
            )
        if str(row.get("elasticity_source", "")) == "category_fallback":
            return (
                "REVIEW_CATEGORY_ELASTICITY_FALLBACK",
                "review",
                "Use this as directional guidance and monitor the first live results closely.",
                "The action relies on category-level price response rather than direct item evidence.",
                "Medium",
            )
        if int(row.get("selected", 0)) == 1 and int(row.get("applied_is_change", 0)) == 1:
            if np.isfinite(final_price) and np.isfinite(price) and final_price > price:
                return (
                    "APPROVED_PRICE_INCREASE_MARGIN_CAPTURE",
                    "approved",
                    "Prioritise this action in the execution list.",
                    "Higher price is projected to more than offset demand loss.",
                    "High",
                )
            if np.isfinite(final_price) and np.isfinite(price) and final_price < price:
                return (
                    "APPROVED_DISCOUNT_VOLUME_DRIVE",
                    "approved",
                    "Execute as a constrained promotion candidate.",
                    "Discount-driven demand lift is still expected to improve profit.",
                    "High",
                )
        if (
            int(row.get("eligible", 0)) == 1
            and int(row.get("selected", 0)) == 0
            and abs(_float(row.get("best_delta"))) > 1e-12
        ):
            return (
                "DEPRIORITISED_BY_CONSTRAINTS",
                "constraint",
                "Keep on the watchlist for the next budget or capacity cycle.",
                "The action had upside unconstrained but did not survive capacity or budget rules.",
                "Medium",
            )
        if (
            np.isfinite(best_price)
            and np.isfinite(price)
            and best_price > price
            and profit_gain > 0
        ):
            return (
                "UNCONSTRAINED_PRICE_INCREASE_NOT_EXECUTED",
                "informational",
                "Useful for scenario comparison, not for the final execution list.",
                "The unconstrained optimiser prefers a higher price than current.",
                "Medium",
            )
        if (
            np.isfinite(best_price)
            and np.isfinite(price)
            and best_price < price
            and profit_gain > 0
        ):
            return (
                "UNCONSTRAINED_DISCOUNT_NOT_EXECUTED",
                "informational",
                "Useful for scenario comparison, not for the final execution list.",
                "The unconstrained optimiser prefers a discount versus current price.",
                "Medium",
            )
        if abs(profit_gain) <= 1e-9 or abs(_float(row.get("best_delta"))) <= 1e-12:
            return (
                "HOLD_NO_MATERIAL_GAIN",
                "hold",
                "Hold current price and re-evaluate after new data arrives.",
                "No material gain cleared the decision threshold.",
                "Low",
            )
        return (
            "APPROVED_HOLD_BASELINE",
            "approved",
            "Keep the current price because it remains the safest option.",
            "The final plan does not justify a change.",
            "Low",
        )

    reason_cols = df.apply(classify, axis=1, result_type="expand")
    reason_cols.columns = [
        "reason_code",
        "reason_group",
        "action_recommendation",
        "business_rationale",
        "priority",
    ]
    df = pd.concat([df, reason_cols], axis=1)

    df["price_change_pct"] = (
        (df["final_price_recommendation"] - df["price"]) / (df["price"].abs() + 1e-9)
    ) * 100.0
    df["final_profit_uplift_gbp"] = df["final_profit_projection"] - df["base_profit"]
    df["final_profit_uplift_pct"] = (
        df["final_profit_uplift_gbp"] / (df["base_profit"].abs() + 1e-9)
    ) * 100.0
    df["portfolio_demo_warning"] = np.where(
        df["reason_code"].str.startswith("REVIEW_"),
        "Needs analyst review before execution.",
        "Directional action only; validate with live or historical control logic.",
    )

    cols = [
        c
        for c in [
            "id",
            "store_id",
            "item_id",
            "cat_id",
            "price",
            "best_price",
            "final_price_recommendation",
            "price_change_pct",
            "base_profit",
            "best_profit",
            "final_profit_projection",
            "profit_gain",
            "final_profit_uplift_gbp",
            "final_profit_uplift_pct",
            "elasticity",
            "elasticity_source",
            "reason_code",
            "reason_group",
            "priority",
            "action_recommendation",
            "business_rationale",
            "selected",
            "eligible",
            "applied_is_change",
            "portfolio_demo_warning",
        ]
        if c in df.columns
    ]
    return df[cols].copy()


def build_scenario_comparison(
    price_df: pd.DataFrame,
    promo_df: Optional[pd.DataFrame],
    backtests: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    base_profit = float(price_df["base_profit"].sum())
    opt_profit = float(price_df["best_profit"].sum())
    if promo_df is not None and "applied_profit" in promo_df.columns:
        constrained_profit = float(promo_df["applied_profit"].sum())
        selected_changes = int((promo_df.get("applied_is_change", 0) == 1).sum())
        selected_rows = int((promo_df.get("selected", 0) == 1).sum())
        spend_used = float(
            promo_df.get("promo_spend_proxy", pd.Series(0.0, index=promo_df.index))
            .where(promo_df.get("selected", 0) == 1, 0.0)
            .sum()
        )
    else:
        constrained_profit = float("nan")
        selected_changes = 0
        selected_rows = 0
        spend_used = 0.0

    latest = backtests[0] if backtests else {}
    best_baseline_wmape = np.nan
    if latest:
        best_baseline_wmape = min(
            _float(latest.get("wmape_baseline_mean_28")),
            _float(latest.get("wmape_baseline_seas_7")),
            _float(latest.get("wmape_baseline_seas_364")),
        )

    rows = [
        {
            "scenario": "baseline_current_price",
            "scenario_label": "Baseline",
            "profit_gbp": base_profit,
            "uplift_gbp": 0.0,
            "uplift_pct": 0.0,
            "candidate_actions": 0,
            "selected_actions": 0,
            "selected_price_changes": 0,
            "avg_price_change_pct": 0.0,
            "avg_profit_uplift_pct": 0.0,
            "budget_used_gbp": 0.0,
            "forecast_winner": (
                "baseline"
                if latest and _float(latest.get("wmape_lgbm")) > best_baseline_wmape
                else "lgbm"
            ),
            "latest_model_wmape": _float(latest.get("wmape_lgbm")),
            "latest_best_baseline_wmape": best_baseline_wmape,
        },
        {
            "scenario": "unconstrained_price_optimizer",
            "scenario_label": "Unconstrained price optimisation",
            "profit_gbp": opt_profit,
            "uplift_gbp": opt_profit - base_profit,
            "uplift_pct": ((opt_profit - base_profit) / (abs(base_profit) + 1e-9)) * 100.0,
            "candidate_actions": (
                int((price_df["best_delta"].abs() > 1e-12).sum())
                if "best_delta" in price_df.columns
                else int(len(price_df))
            ),
            "selected_actions": int(len(price_df)),
            "selected_price_changes": (
                int((price_df["best_price"] != price_df["price"]).sum())
                if "best_price" in price_df.columns
                else 0
            ),
            "avg_price_change_pct": (
                float(
                    (
                        (
                            (price_df["best_price"] - price_df["price"])
                            / (price_df["price"].abs() + 1e-9)
                        )
                        * 100.0
                    ).mean()
                )
                if "best_price" in price_df.columns
                else float("nan")
            ),
            "avg_profit_uplift_pct": float(
                (
                    (price_df["best_profit"] - price_df["base_profit"])
                    / (price_df["base_profit"].abs() + 1e-9)
                    * 100.0
                ).mean()
            ),
            "budget_used_gbp": 0.0,
            "forecast_winner": (
                "baseline"
                if latest and _float(latest.get("wmape_lgbm")) > best_baseline_wmape
                else "lgbm"
            ),
            "latest_model_wmape": _float(latest.get("wmape_lgbm")),
            "latest_best_baseline_wmape": best_baseline_wmape,
        },
        {
            "scenario": "constrained_execution_plan",
            "scenario_label": "Constrained execution plan",
            "profit_gbp": constrained_profit,
            "uplift_gbp": (
                constrained_profit - base_profit
                if np.isfinite(constrained_profit)
                else float("nan")
            ),
            "uplift_pct": (
                ((constrained_profit - base_profit) / (abs(base_profit) + 1e-9)) * 100.0
                if np.isfinite(constrained_profit)
                else float("nan")
            ),
            "candidate_actions": (
                int((promo_df.get("eligible", 0) == 1).sum()) if promo_df is not None else 0
            ),
            "selected_actions": selected_rows,
            "selected_price_changes": selected_changes,
            "avg_price_change_pct": (
                float(
                    (
                        (
                            (
                                promo_df.get("applied_price", promo_df.get("new_price"))
                                - promo_df["price"]
                            )
                            / (promo_df["price"].abs() + 1e-9)
                        )
                        * 100.0
                    ).mean()
                )
                if promo_df is not None and "price" in promo_df.columns
                else float("nan")
            ),
            "avg_profit_uplift_pct": (
                float(
                    (
                        (
                            (
                                promo_df.get("applied_profit", promo_df.get("new_profit"))
                                - promo_df["base_profit"]
                            )
                            / (promo_df["base_profit"].abs() + 1e-9)
                        )
                        * 100.0
                    ).mean()
                )
                if promo_df is not None and "base_profit" in promo_df.columns
                else float("nan")
            ),
            "budget_used_gbp": spend_used,
            "forecast_winner": (
                "baseline"
                if latest and _float(latest.get("wmape_lgbm")) > best_baseline_wmape
                else "lgbm"
            ),
            "latest_model_wmape": _float(latest.get("wmape_lgbm")),
            "latest_best_baseline_wmape": best_baseline_wmape,
        },
    ]
    return pd.DataFrame(rows)


def build_executive_kpi_summary(
    price_df: pd.DataFrame,
    reason_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    promo_df: Optional[pd.DataFrame],
    backtests: Optional[List[Dict[str, Any]]],
    uplift_backtest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    latest = backtests[0] if backtests else {}
    best_baseline_wmape = np.nan
    if latest:
        best_baseline_wmape = min(
            _float(latest.get("wmape_baseline_mean_28")),
            _float(latest.get("wmape_baseline_seas_7")),
            _float(latest.get("wmape_baseline_seas_364")),
        )
    model_wmape = _float(latest.get("wmape_lgbm"))
    forecast_winner = "baseline" if latest and model_wmape > best_baseline_wmape else "lgbm"
    model_uplift = (
        ((best_baseline_wmape - model_wmape) / (best_baseline_wmape + 1e-9) * 100.0)
        if latest
        else float("nan")
    )

    global_fallback_share = (
        float(
            (
                price_df.get("elasticity_source", pd.Series([], dtype=object)) == "global_fallback"
            ).mean()
        )
        if len(price_df)
        else float("nan")
    )
    category_fallback_share = (
        float(
            (
                price_df.get("elasticity_source", pd.Series([], dtype=object))
                == "category_fallback"
            ).mean()
        )
        if len(price_df)
        else float("nan")
    )
    selected_changes = (
        int((promo_df.get("applied_is_change", 0) == 1).sum())
        if promo_df is not None and len(promo_df)
        else 0
    )
    review_rows = int(reason_df["reason_group"].eq("review").sum()) if len(reason_df) else 0
    unique_stores = int(price_df["store_id"].nunique()) if "store_id" in price_df.columns else 0
    unique_categories = int(price_df["cat_id"].nunique()) if "cat_id" in price_df.columns else 0

    readiness = "portfolio_demo_with_caveats"
    truth_checks: List[str] = []
    if forecast_winner == "baseline":
        truth_checks.append(
            "Forecast model does not beat the best baseline on latest WMAPE, so baseline should remain the serving choice."
        )
    if global_fallback_share > 0.50:
        readiness = "directional_only"
        truth_checks.append(
            "More than half of price actions rely on the global fallback elasticity, so the price plan is directional rather than production-grade."
        )
    if unique_stores < 2 or unique_categories < 2:
        readiness = "directional_only"
        truth_checks.append(
            "The analysed slice is narrow, so store/category comparisons are not broad enough for a real commercial rollout story."
        )
    if review_rows > 0:
        truth_checks.append(f"{review_rows} rows are flagged for manual review before execution.")
    if promo_df is not None and selected_changes == 0:
        truth_checks.append(
            "The constrained execution plan currently selects zero live price changes, so the operational story is currently insight-led rather than execution-led."
        )
    if not truth_checks:
        truth_checks.append(
            "This is a credible portfolio-style decision system, but it still depends on simulated economics and should be framed as a decision-support prototype."
        )

    stability = uplift_backtest.get("stability", {}) if isinstance(uplift_backtest, dict) else {}

    summary = {
        "readiness": readiness,
        "headline": {
            "forecast_winner": forecast_winner,
            "latest_model_wmape": model_wmape,
            "latest_best_baseline_wmape": best_baseline_wmape,
            "forecast_wmape_uplift_pct": model_uplift,
            "baseline_profit_gbp": _float(
                scenario_df.loc[
                    scenario_df["scenario"] == "baseline_current_price", "profit_gbp"
                ].iloc[0]
            ),
            "unconstrained_profit_gbp": _float(
                scenario_df.loc[
                    scenario_df["scenario"] == "unconstrained_price_optimizer", "profit_gbp"
                ].iloc[0]
            ),
            "constrained_profit_gbp": _float(
                scenario_df.loc[
                    scenario_df["scenario"] == "constrained_execution_plan", "profit_gbp"
                ].iloc[0]
            ),
            "approved_price_changes": selected_changes,
            "review_rows": review_rows,
        },
        "portfolio_truth_checks": truth_checks,
        "coverage": {
            "rows_modelled": int(len(price_df)),
            "unique_stores": unique_stores,
            "unique_categories": unique_categories,
            "global_fallback_share": global_fallback_share,
            "category_fallback_share": category_fallback_share,
        },
        "stability": {
            "mean_uplift_pct": stability.get("mean_uplift_pct"),
            "std_uplift_pct": stability.get("std_uplift_pct"),
            "uplift_pct_p10": stability.get("uplift_pct_p10"),
            "uplift_pct_p50": stability.get("uplift_pct_p50"),
            "uplift_pct_p90": stability.get("uplift_pct_p90"),
        },
        "scenario_summary": scenario_df.to_dict(orient="records"),
        "reason_code_mix": reason_df.groupby(["reason_code", "reason_group"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("rows", ascending=False)
        .to_dict(orient="records"),
        "recruiter_positioning": "This repo now shows forecasting, commercial optimisation, constrained decision logic, KPI surfacing, and BI-ready reporting instead of stopping at model outputs.",
        "hash": stable_config_hash(
            {
                "headline": {
                    "forecast_winner": forecast_winner,
                    "baseline_profit_gbp": _float(
                        scenario_df.loc[
                            scenario_df["scenario"] == "baseline_current_price", "profit_gbp"
                        ].iloc[0]
                    ),
                    "unconstrained_profit_gbp": _float(
                        scenario_df.loc[
                            scenario_df["scenario"] == "unconstrained_price_optimizer", "profit_gbp"
                        ].iloc[0]
                    ),
                    "constrained_profit_gbp": _float(
                        scenario_df.loc[
                            scenario_df["scenario"] == "constrained_execution_plan", "profit_gbp"
                        ].iloc[0]
                    ),
                },
                "coverage": {
                    "rows_modelled": int(len(price_df)),
                    "global_fallback_share": global_fallback_share,
                },
            }
        ),
    }
    return summary


def _kpi_dictionary_rows() -> pd.DataFrame:
    rows = [
        (
            "latest_model_wmape",
            "Forecast",
            "Weighted mean absolute percentage error for the LightGBM forecast on the latest validation window.",
            "Lower is better",
        ),
        (
            "latest_best_baseline_wmape",
            "Forecast",
            "Best WMAPE across the baseline_mean_28, seasonal_7, and seasonal_364 baselines.",
            "Lower is better",
        ),
        (
            "forecast_wmape_uplift_pct",
            "Forecast",
            "Relative WMAPE improvement versus the best baseline. Positive means the model beats the baseline.",
            "Higher is better",
        ),
        (
            "baseline_profit_gbp",
            "Commercial",
            "Projected profit at current prices using the forecasted 28-day base demand.",
            "Context only",
        ),
        (
            "unconstrained_profit_gbp",
            "Commercial",
            "Projected profit if every unconstrained optimiser action were applied.",
            "Higher is better",
        ),
        (
            "constrained_profit_gbp",
            "Commercial",
            "Projected profit after execution constraints are applied.",
            "Higher is better",
        ),
        (
            "approved_price_changes",
            "Execution",
            "Count of live price changes approved by the constrained plan.",
            "Higher is better when credible",
        ),
        (
            "review_rows",
            "Governance",
            "Count of actions that need analyst review because of weak evidence or guardrails.",
            "Lower is better",
        ),
        (
            "global_fallback_share",
            "Governance",
            "Share of rows using the global default elasticity rather than item or category evidence.",
            "Lower is better",
        ),
        (
            "mean_uplift_pct",
            "Stability",
            "Average uplift across uplift backtest cutoffs.",
            "Higher is better",
        ),
        (
            "std_uplift_pct",
            "Stability",
            "Variation in uplift across uplift backtest cutoffs.",
            "Lower is better",
        ),
    ]
    return pd.DataFrame(rows, columns=["kpi_name", "kpi_group", "definition", "interpretation"])


def _write_markdown_files(cfg: BusinessPackConfig, executive: Dict[str, Any]) -> Dict[str, str]:
    docs_dir = Path(ensure_dir(os.path.join(cfg.data_dir, cfg.docs_subdir)))
    reports_dir = Path(ensure_dir(os.path.join(cfg.data_dir, cfg.reports_subdir)))

    kpi_dict_path = docs_dir / "kpi_dictionary.md"
    user_guide_path = docs_dir / "user_guide.md"
    exec_md_path = reports_dir / "executive_kpi_summary.md"

    kpi_md = [
        "# KPI Dictionary",
        "",
        "| KPI | Group | Definition | Interpretation |",
        "|---|---|---|---|",
    ]
    for row in _kpi_dictionary_rows().to_dict(orient="records"):
        kpi_md.append(
            f"| {row['kpi_name']} | {row['kpi_group']} | {row['definition']} | {row['interpretation']} |"
        )
    kpi_dict_path.write_text("\n".join(kpi_md) + "\n", encoding="utf-8")

    user_guide = f"""# User Guide

## What this pack does
This pack translates model outputs into decision-support artefacts for a recruiter, hiring manager, analyst, or BI consumer.

## Files to open first
1. `reports/executive_kpi_summary.json` for the machine-readable headline.
2. `reports/executive_kpi_summary.md` for the human-readable summary.
3. `reports/scenario_comparison.csv` for baseline vs unconstrained vs constrained outcomes.
4. `reports/reason_coded_action_recommendations.csv` for the action list with business reasons.
5. `reports/dashboard_ready/` for Power BI, Tableau, SQL, or Streamlit ingestion.

## How to explain this in an interview
- Start with the forecast winner and whether the model actually beats the best baseline.
- Move to the scenario comparison to show baseline, unconstrained upside, and constrained executable value.
- Use the reason-coded actions to show this is a decision system, not just a model notebook.
- Mention the governance layer: suspicious uplift flags, elasticity-source visibility, and review queues.
- Be honest that the price optimisation layer still relies on simulated economics unless real costs and live experimentation are provided.

## Truth-in-advertising guidance
Current readiness: **{executive.get('readiness', 'unknown')}**.
Do not claim this is a live production pricing engine unless you replace proxies with real commercial data and experiment design.
"""
    user_guide_path.write_text(user_guide, encoding="utf-8")

    headline = executive.get("headline", {})
    truth_checks = executive.get("portfolio_truth_checks", [])
    scenario_rows = executive.get("scenario_summary", [])
    lines = [
        "# Executive KPI Summary",
        "",
        f"- Readiness: **{executive.get('readiness', 'unknown')}**",
        f"- Forecast winner: **{headline.get('forecast_winner')}**",
        f"- Latest model WMAPE: **{headline.get('latest_model_wmape')}**",
        f"- Best baseline WMAPE: **{headline.get('latest_best_baseline_wmape')}**",
        f"- Baseline profit (GBP): **{headline.get('baseline_profit_gbp')}**",
        f"- Unconstrained profit (GBP): **{headline.get('unconstrained_profit_gbp')}**",
        f"- Constrained profit (GBP): **{headline.get('constrained_profit_gbp')}**",
        f"- Approved price changes: **{headline.get('approved_price_changes')}**",
        f"- Review rows: **{headline.get('review_rows')}**",
        "",
        "## Scenario comparison",
        "",
        "| Scenario | Profit (GBP) | Uplift (GBP) | Uplift (%) | Selected price changes |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scenario_rows:
        lines.append(
            f"| {row.get('scenario_label')} | {row.get('profit_gbp', float('nan')):.2f} | {row.get('uplift_gbp', float('nan')):.2f} | {row.get('uplift_pct', float('nan')):.2f} | {row.get('selected_price_changes', 0)} |"
        )
    lines.extend(["", "## Truth checks", ""])
    lines.extend([f"- {x}" for x in truth_checks])
    exec_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "kpi_dictionary_md": str(kpi_dict_path),
        "user_guide_md": str(user_guide_path),
        "executive_kpi_summary_md": str(exec_md_path),
    }


def _build_store_category_rollups(
    reason_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_store = reason_df.groupby("store_id", as_index=False).agg(
        total_rows=("id", "count"),
        approved_rows=("reason_group", lambda s: int((s == "approved").sum())),
        review_rows=("reason_group", lambda s: int((s == "review").sum())),
        expected_profit_uplift_gbp=("final_profit_uplift_gbp", "sum"),
    )
    by_cat = reason_df.groupby("cat_id", as_index=False).agg(
        total_rows=("id", "count"),
        approved_rows=("reason_group", lambda s: int((s == "approved").sum())),
        review_rows=("reason_group", lambda s: int((s == "review").sum())),
        expected_profit_uplift_gbp=("final_profit_uplift_gbp", "sum"),
    )
    reason_mix = (
        reason_df.groupby(["reason_code", "reason_group"], as_index=False)
        .agg(rows=("id", "count"), uplift_gbp=("final_profit_uplift_gbp", "sum"))
        .sort_values(["rows", "uplift_gbp"], ascending=[False, False])
    )
    return by_store, by_cat, reason_mix


def _write_dashboard_mockup(
    path: str | Path, executive: Dict[str, Any], scenario_df: pd.DataFrame, reason_mix: pd.DataFrame
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update({"figure.figsize": (14, 8)})
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        ax0.axis("off")
        headline = executive.get("headline", {})
        cards = [
            f"Readiness\n{executive.get('readiness', 'unknown')}",
            f"Forecast winner\n{headline.get('forecast_winner', 'n/a')}",
            f"Baseline profit\nGBP {headline.get('baseline_profit_gbp', float('nan')):,.2f}",
            f"Constrained profit\nGBP {headline.get('constrained_profit_gbp', float('nan')):,.2f}",
            f"Review rows\n{headline.get('review_rows', 0)}",
        ]
        for i, card in enumerate(cards):
            ax0.text(
                0.02 + i * 0.19,
                0.5,
                card,
                fontsize=14,
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.6", alpha=0.15),
            )
        ax0.text(
            0.02,
            0.95,
            "Retail Decision Intelligence Dashboard Mockup",
            fontsize=18,
            weight="bold",
            va="top",
        )

        s = scenario_df.copy()
        ax1.bar(s["scenario_label"], s["profit_gbp"])
        ax1.set_title("Scenario profit comparison")
        ax1.set_ylabel("Projected profit (GBP)")
        ax1.tick_params(axis="x", rotation=15)

        top_reason = reason_mix.head(8).copy()
        ax2.barh(top_reason["reason_code"], top_reason["rows"])
        ax2.set_title("Reason-code mix")
        ax2.set_xlabel("Rows")
        ax2.invert_yaxis()

        fig.tight_layout()
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(p)
    except Exception:
        return None


def generate_business_pack(cfg: BusinessPackConfig) -> Dict[str, Any]:
    logger = get_logger("business_pack")
    reports_dir = Path(ensure_dir(os.path.join(cfg.data_dir, cfg.reports_subdir)))
    dashboard_dir = Path(ensure_dir(os.path.join(reports_dir, cfg.dashboard_subdir)))

    price_df = _read_csv_if_exists(Path(cfg.data_dir) / "price_optimization_results.csv")
    if price_df is None or price_df.empty:
        raise FileNotFoundError(
            "price_optimization_results.csv not found. Run price optimisation before generating the business pack."
        )
    promo_df = _read_csv_if_exists(Path(cfg.data_dir) / "promo_selection_results.csv")
    backtests = _latest_forecast_backtests(cfg.data_dir)
    uplift_backtest = _read_json_if_exists(reports_dir / "uplift_backtest.json")

    reason_df = build_reason_coded_actions(price_df=price_df, promo_df=promo_df)
    scenario_df = build_scenario_comparison(
        price_df=price_df, promo_df=promo_df, backtests=backtests
    )
    executive = build_executive_kpi_summary(
        price_df=price_df,
        reason_df=reason_df,
        scenario_df=scenario_df,
        promo_df=promo_df,
        backtests=backtests,
        uplift_backtest=uplift_backtest,
    )
    by_store, by_cat, reason_mix = _build_store_category_rollups(reason_df)
    reason_ref = pd.DataFrame(REASON_CODE_REFERENCE)
    kpi_dict = _kpi_dictionary_rows()

    reason_path = reports_dir / "reason_coded_action_recommendations.csv"
    scenario_path = reports_dir / "scenario_comparison.csv"
    executive_json_path = reports_dir / "executive_kpi_summary.json"
    dashboard_files = {
        "fact_action_recommendations_csv": dashboard_dir / "fact_action_recommendations.csv",
        "fact_scenario_comparison_csv": dashboard_dir / "fact_scenario_comparison.csv",
        "agg_store_action_summary_csv": dashboard_dir / "agg_store_action_summary.csv",
        "agg_category_action_summary_csv": dashboard_dir / "agg_category_action_summary.csv",
        "agg_reason_code_mix_csv": dashboard_dir / "agg_reason_code_mix.csv",
        "dim_reason_codes_csv": dashboard_dir / "dim_reason_codes.csv",
        "dim_kpi_dictionary_csv": dashboard_dir / "dim_kpi_dictionary.csv",
    }

    reason_df.to_csv(reason_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    write_json(executive_json_path, executive)

    reason_df.to_csv(dashboard_files["fact_action_recommendations_csv"], index=False)
    scenario_df.to_csv(dashboard_files["fact_scenario_comparison_csv"], index=False)
    by_store.to_csv(dashboard_files["agg_store_action_summary_csv"], index=False)
    by_cat.to_csv(dashboard_files["agg_category_action_summary_csv"], index=False)
    reason_mix.to_csv(dashboard_files["agg_reason_code_mix_csv"], index=False)
    reason_ref.to_csv(dashboard_files["dim_reason_codes_csv"], index=False)
    kpi_dict.to_csv(dashboard_files["dim_kpi_dictionary_csv"], index=False)

    if uplift_backtest and isinstance(uplift_backtest, dict):
        rows = pd.DataFrame(uplift_backtest.get("cutoff_results", []))
        if len(rows):
            rows.to_csv(dashboard_dir / "fact_uplift_backtest.csv", index=False)
            dashboard_files["fact_uplift_backtest_csv"] = dashboard_dir / "fact_uplift_backtest.csv"

    markdown_paths = _write_markdown_files(cfg, executive)
    mockup_path = _write_dashboard_mockup(
        cfg.dashboard_mockup_path, executive, scenario_df, reason_mix
    )
    if mockup_path:
        dashboard_files["dashboard_mockup_png"] = Path(mockup_path)

    output = {
        "executive_kpi_summary_json": str(executive_json_path),
        "reason_coded_action_recommendations_csv": str(reason_path),
        "scenario_comparison_csv": str(scenario_path),
        "dashboard_exports": {k: str(v) for k, v in dashboard_files.items()},
        "docs": markdown_paths,
        "readiness": executive.get("readiness"),
    }
    logger.info("Generated business pack in %s", reports_dir)
    return output
