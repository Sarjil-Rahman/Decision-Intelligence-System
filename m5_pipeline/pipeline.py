from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from .utils import get_logger, require_files, write_json
from .validation import ValidationConfig, validate_m5_inputs
from .m5_forecasting import ForecastConfig, run_forecast
from .m5_price_optimization import PriceOptConfig, run_price_optimization, backtest_price_uplift
from .m5_promo_selection import PromoSelectionConfig, run_promo_selection
from .business_outputs import BusinessPackConfig, generate_business_pack


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M5: Validate → Forecast (LGBM vs baselines) → Price Opt → Constrained Promo Selection"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--max-series",
        type=int,
        default=0,
        help="0 = all series; use 5000/10000 for quicker dev runs",
    )
    p.add_argument("--start-d", default="d_1500")
    p.add_argument("--last-train-d", default="d_1913")
    p.add_argument("--horizon", type=int, default=28)

    # Forecast tuning
    p.add_argument("--objective", default="tweedie", choices=["tweedie", "poisson"])
    p.add_argument("--tweedie-power", type=float, default=1.1)
    p.add_argument(
        "--split-strategy", default="rolling_origin", choices=["rolling_origin", "last_window"]
    )
    p.add_argument("--n-backtests", type=int, default=3)
    p.add_argument("--backtest-stride", type=int, default=28)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--classifier-n-estimators", type=int, default=500)
    p.add_argument("--regressor-n-estimators", type=int, default=None)
    p.add_argument("--classifier-early-stopping-rounds", type=int, default=50)
    p.add_argument("--regressor-early-stopping-rounds", type=int, default=None)
    p.add_argument("--lightgbm-verbosity", type=int, default=-1)
    p.add_argument("--random-state", type=int, default=42)

    # Pricing
    p.add_argument("--skip-price-opt", action="store_true")
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument(
        "--unit-econ-path", default=None, help="Optional costs/margins csv relative to data-dir"
    )

    # Standard industry guardrails (pricing)
    p.add_argument(
        "--max-price-move-pct",
        type=float,
        default=0.20,
        help="Cap absolute price move, e.g. 0.2 = +/-20%",
    )
    p.add_argument("--elasticity-clip-low", type=float, default=-5.0)
    p.add_argument("--elasticity-clip-high", type=float, default=-0.1)
    p.add_argument(
        "--max-demand-mult", type=float, default=3.0, help="Cap demand uplift, q <= base * k"
    )

    # Constraints
    p.add_argument("--max-changes-total", type=int, default=5000)
    p.add_argument("--max-changes-per-store", type=int, default=800)
    p.add_argument("--max-changes-per-cat", type=int, default=1200)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument(
        "--allow-price-increase",
        action="store_true",
        help="Allow price increases in constrained promo selection. Default is discount-only.",
    )
    p.add_argument(
        "--allow-no-price-change",
        action="store_true",
        help="Allow explicit no-change baseline candidates in promo selection.",
    )

    # Uplift backtest
    p.add_argument(
        "--uplift-cutoffs",
        default=None,
        help="Comma-separated d_XXXX cutoffs (e.g. d_1700,d_1760,d_1820). If omitted, uses 3 cutoffs.",
    )

    return p


def _parse_cutoffs(s: Optional[str], last_train_d: str, stride: int, n: int) -> List[str]:
    if s:
        return [x.strip() for x in s.split(",") if x.strip()]
    # default: 3 rolling cutoffs before the end
    last = int(last_train_d.split("_")[1])
    out = [f"d_{last - stride * (i + 2)}" for i in range(n)]  # keep distance from end
    return out


def main() -> None:
    args = build_parser().parse_args()
    logger = get_logger("runner")

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"data-dir not found: {args.data_dir}")

    require_files(
        args.data_dir,
        ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv", "sample_submission.csv"],
    )

    # 0) Validate inputs + write data quality report (fail fast)
    vcfg = ValidationConfig(data_dir=args.data_dir, strict=True, write_reports=True)
    vrep = validate_m5_inputs(vcfg)
    logger.info("✅ Input validation passed.")

    # 1) Forecast + baseline comparison (includes rolling-origin backtests + residual quantiles)
    fc = ForecastConfig(
        data_dir=args.data_dir,
        max_series=args.max_series,
        start_d=args.start_d,
        last_train_d=args.last_train_d,
        horizon=args.horizon,
        out_submission="submission.csv",
        objective=args.objective,
        tweedie_variance_power=args.tweedie_power,
        n_jobs=args.n_jobs,
        n_estimators=args.n_estimators or ForecastConfig.n_estimators,
        classifier_n_estimators=args.classifier_n_estimators,
        regressor_n_estimators=args.regressor_n_estimators,
        classifier_early_stopping_rounds=args.classifier_early_stopping_rounds,
        regressor_early_stopping_rounds=args.regressor_early_stopping_rounds,
        lightgbm_verbosity=args.lightgbm_verbosity,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        n_backtests=args.n_backtests,
        backtest_stride=args.backtest_stride,
        save_artifacts=True,
        validate_inputs=False,  # already validated above
    )
    fres = run_forecast(fc)

    # Headline: WMAPE (overall) + segment WMAPE to prove robustness
    latest = fres["backtests"][0]
    best_baseline_wmape = min(
        latest["wmape_baseline_mean_28"],
        latest["wmape_baseline_seas_7"],
        latest["wmape_baseline_seas_364"],
    )
    lift_wmape = (best_baseline_wmape - latest["wmape_lgbm"]) / (best_baseline_wmape + 1e-9) * 100.0

    if fres["winner"] == "lgbm":
        logger.info(
            "✅ Forecast model beats best baseline by %.2f%% WMAPE (best baseline=%.4f → model=%.4f).",
            lift_wmape,
            best_baseline_wmape,
            latest["wmape_lgbm"],
        )
        logger.info(
            "Segment WMAPE: event=%.4f, non-event=%.4f, price-drop=%.4f, non-price-drop=%.4f",
            latest.get("wmape_event_days", float("nan")),
            latest.get("wmape_non_event_days", float("nan")),
            latest.get("wmape_price_drop_days", float("nan")),
            latest.get("wmape_non_price_drop_days", float("nan")),
        )
    else:
        logger.info(
            "⚠️ Model did not beat baselines on latest-split WMAPE (best baseline=%.4f, model=%.4f). Using baseline for serving.",
            best_baseline_wmape,
            latest["wmape_lgbm"],
        )

    if args.skip_price_opt:
        logger.info("Skipping price optimisation & promo selection.")
        return

    # 2) Price optimisation (unconstrained)
    pc = PriceOptConfig(
        data_dir=args.data_dir,
        last_train_d=args.last_train_d,
        submission_path="submission.csv",
        out_path="price_optimization_results.csv",
        margin=args.margin,
        unit_econ_path=args.unit_econ_path,
        max_series=args.max_series,
        elasticity_clip=(float(args.elasticity_clip_low), float(args.elasticity_clip_high)),
        max_abs_price_change_pct=float(args.max_price_move_pct),
        max_demand_mult=float(args.max_demand_mult),
        write_reports=True,
    )
    pres = run_price_optimization(pc)
    price_summary = pres.get("summary", {})
    df_opt = pd.read_csv(os.path.join(args.data_dir, "price_optimization_results.csv"))

    base_profit = float(df_opt["base_profit"].sum())
    opt_profit = float(df_opt["best_profit"].sum())
    uplift_opt = (opt_profit / (base_profit + 1e-9) - 1.0) * 100.0

    logger.info(
        "💷 Profit (unconstrained): base=%.2f → optimised=%.2f (uplift=%.2f%%). suspicious=%s",
        base_profit,
        opt_profit,
        uplift_opt,
        int(df_opt.get("suspicious_uplift", pd.Series([0])).sum()),
    )

    # Store uplift per store/category (business visibility)
    uplift_store = (
        df_opt.assign(profit_uplift=df_opt["best_profit"] - df_opt["base_profit"])
        .groupby("store_id", as_index=False)[["base_profit", "best_profit", "profit_uplift"]]
        .sum()
        .sort_values("profit_uplift", ascending=False)
    )
    uplift_cat = (
        df_opt.assign(profit_uplift=df_opt["best_profit"] - df_opt["base_profit"])
        .groupby("cat_id", as_index=False)[["base_profit", "best_profit", "profit_uplift"]]
        .sum()
        .sort_values("profit_uplift", ascending=False)
    )

    # 3) Constrained promo selection
    sc = PromoSelectionConfig(
        data_dir=args.data_dir,
        input_path="price_optimization_results.csv",
        out_path="promo_selection_results.csv",
        max_price_changes_total=args.max_changes_total,
        max_price_changes_per_store=args.max_changes_per_store,
        max_price_changes_per_cat=args.max_changes_per_cat,
        budget=args.budget,
        forbid_price_increase=not args.allow_price_increase,
        require_price_change=not args.allow_no_price_change,
        max_abs_price_change_pct=float(args.max_price_move_pct),
        max_demand_mult=float(args.max_demand_mult),
        write_reports=True,
    )
    sres = run_promo_selection(sc)
    promo_summary = sres.get("summary", {})
    df_sel = pd.read_csv(os.path.join(args.data_dir, "promo_selection_results.csv"))

    constrained_profit = float(df_sel["applied_profit"].sum())
    uplift_con = (constrained_profit / (base_profit + 1e-9) - 1.0) * 100.0
    n_changes = int(df_sel["applied_is_change"].sum())
    spend = float(df_sel.loc[df_sel["selected"] == 1, "promo_spend_proxy"].sum())

    logger.info(
        "🏷️ Constrained decisions (%s): profit=%.2f (uplift=%.2f%%), price_changes=%s, spend_proxy=%.2f",
        sres["method"],
        constrained_profit,
        uplift_con,
        n_changes,
        spend,
    )
    if sres.get("constraint_report", {}).get("any_violation"):
        logger.warning("Constraint violation detected! Report: %s", sres["constraint_report"])

    # 4) Profit uplift backtest on historical cutoffs (credibility layer)
    cutoffs = _parse_cutoffs(args.uplift_cutoffs, args.last_train_d, args.backtest_stride, 3)
    bt = backtest_price_uplift(
        data_dir=args.data_dir,
        cutoffs=cutoffs,
        horizon=args.horizon,
        max_series=args.max_series,
        margin=args.margin,
        unit_econ_path=args.unit_econ_path,
        lookback_days=365,
        elasticity_clip=(float(args.elasticity_clip_low), float(args.elasticity_clip_high)),
        max_abs_price_change_pct=float(args.max_price_move_pct),
        max_demand_mult=float(args.max_demand_mult),
    )
    logger.info("📉 Uplift backtest written: %s", bt.get("report_path", ""))

    # 5) Save KPI summary
    kpis = {
        "validation": vrep,
        "forecast": {
            "winner": fres["winner"],
            "selected_baseline": fres.get("selected_baseline"),
            "promotion": fres.get("promotion", {}),
            "prediction_intervals": fres.get("prediction_intervals", {}),
            "latest_split": latest.get("split", {}),
            "residual_q10": latest.get("residual_q10", float("nan")),
            "residual_q50": latest.get("residual_q50", float("nan")),
            "residual_q90": latest.get("residual_q90", float("nan")),
            "backtests": fres["backtests"],
            "artifacts": fres.get("artifacts", {}),
        },
        "pricing_guardrails": {
            "elasticity_clip": [float(args.elasticity_clip_low), float(args.elasticity_clip_high)],
            "max_price_move_pct": float(args.max_price_move_pct),
            "max_demand_mult": float(args.max_demand_mult),
            "forbid_price_increase": bool(not args.allow_price_increase),
            "require_price_change": bool(not args.allow_no_price_change),
            "budget": args.budget,
            "unit_econ_path": args.unit_econ_path,
        },
        "profit": {
            "base_profit_gbp": base_profit,
            "optimised_profit_gbp": opt_profit,
            "constrained_profit_gbp": constrained_profit,
            "delta_base_to_optimised_gbp": opt_profit - base_profit,
            "delta_base_to_constrained_gbp": constrained_profit - base_profit,
            "delta_optimised_to_constrained_gbp": constrained_profit - opt_profit,
            "uplift_unconstrained_pct": uplift_opt,
            "uplift_constrained_pct": uplift_con,
            # Unconstrained detail (distribution, contributors, risk, behaviour)
            "unconstrained_summary": price_summary,
            # Constrained detail (what you can actually deploy)
            "constrained_summary": promo_summary,
            # Convenience top tables (fallback to pipeline-computed if summary missing)
            "uplift_by_store_top10": price_summary.get(
                "uplift_by_store_top10", uplift_store.head(10).to_dict(orient="records")
            ),
            "uplift_by_cat_top10": price_summary.get(
                "uplift_by_category_top10", uplift_cat.head(10).to_dict(orient="records")
            ),
            "promo_selection_method": sres["method"],
            "constraint_report": sres.get("constraint_report", {}),
        },
        "uplift_backtest": bt,
        "limitations": pres.get("limitations", []),
    }
    write_json(os.path.join(args.data_dir, "kpis.json"), kpis)
    logger.info("📦 Saved KPI pack: %s", os.path.join(args.data_dir, "kpis.json"))

    business_pack = generate_business_pack(BusinessPackConfig(data_dir=args.data_dir))
    logger.info(
        "📊 Generated executive/business pack: %s",
        business_pack.get("executive_kpi_summary_json", ""),
    )


if __name__ == "__main__":
    main()
