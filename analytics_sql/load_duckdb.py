from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd


def _insert_df(con, table: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    con.register("tmp_table", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM tmp_table")
    con.unregister("tmp_table")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="analytics.duckdb")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--reports-dir", default="./data/reports")
    ap.add_argument("--price-actions-csv", default=None)
    ap.add_argument("--uplift-store-csv", default=None)
    ap.add_argument("--uplift-category-csv", default=None)
    ap.add_argument("--kpis-json", default=None)
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    dashboard_dir = reports / "dashboard_ready"
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    price_actions_csv = (
        Path(args.price_actions_csv) if args.price_actions_csv else (reports / "price_actions.csv")
    )
    if not price_actions_csv.exists():
        alt = reports.parent / "price_optimization_results.csv"
        if alt.exists():
            price_actions_csv = alt
    uplift_store_csv = (
        Path(args.uplift_store_csv) if args.uplift_store_csv else (reports / "uplift_by_store.csv")
    )
    uplift_cat_csv = (
        Path(args.uplift_category_csv)
        if args.uplift_category_csv
        else (reports / "uplift_by_category.csv")
    )
    kpis_json = Path(args.kpis_json) if args.kpis_json else (reports / "kpis.json")
    if not kpis_json.exists():
        alt = reports.parent / "kpis.json"
        if alt.exists():
            kpis_json = alt

    con = duckdb.connect(args.db)
    schema_path = Path(__file__).with_name("schema.sql")
    con.execute(schema_path.read_text(encoding="utf-8"))
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
        [run_id, datetime.utcnow(), f"Loaded from {reports}"],
    )

    if price_actions_csv.exists():
        df = pd.read_csv(price_actions_csv)
        if "price" in df.columns and "base_price" not in df.columns:
            df["base_price"] = df["price"]
        if "best_price" in df.columns and "new_price" not in df.columns:
            df["new_price"] = df["best_price"]
        if "base_demand_28d" in df.columns and "base_demand" not in df.columns:
            df["base_demand"] = df["base_demand_28d"]
        if (
            "uplift_profit" not in df.columns
            and "best_profit" in df.columns
            and "base_profit" in df.columns
        ):
            df["uplift_profit"] = df["best_profit"] - df["base_profit"]
        for c in [
            "store_id",
            "item_id",
            "base_price",
            "new_price",
            "base_demand",
            "elasticity",
            "base_profit",
            "best_profit",
            "uplift_profit",
        ]:
            if c not in df.columns:
                df[c] = None
        df2 = df[
            [
                "store_id",
                "item_id",
                "base_price",
                "new_price",
                "base_demand",
                "elasticity",
                "base_profit",
                "best_profit",
                "uplift_profit",
            ]
        ].copy()
        df2.insert(0, "run_id", run_id)
        _insert_df(con, "fact_price_actions", df2)

    if uplift_store_csv.exists():
        df = pd.read_csv(uplift_store_csv)
        if "optimised_profit" not in df.columns and "opt_profit" in df.columns:
            df["optimised_profit"] = df["opt_profit"]
        if "uplift_pct" not in df.columns and "uplift_opt_pct" in df.columns:
            df["uplift_pct"] = df["uplift_opt_pct"]
        for c in ["store_id", "base_profit", "optimised_profit", "uplift_pct"]:
            if c not in df.columns:
                df[c] = None
        df2 = df[["store_id", "base_profit", "optimised_profit", "uplift_pct"]].copy()
        df2.insert(0, "run_id", run_id)
        _insert_df(con, "agg_uplift_by_store", df2)

    if uplift_cat_csv.exists():
        df = pd.read_csv(uplift_cat_csv)
        if "optimised_profit" not in df.columns and "opt_profit" in df.columns:
            df["optimised_profit"] = df["opt_profit"]
        if "uplift_pct" not in df.columns and "uplift_opt_pct" in df.columns:
            df["uplift_pct"] = df["uplift_opt_pct"]
        for c in ["cat_id", "base_profit", "optimised_profit", "uplift_pct"]:
            if c not in df.columns:
                df[c] = None
        df2 = df[["cat_id", "base_profit", "optimised_profit", "uplift_pct"]].copy()
        df2.insert(0, "run_id", run_id)
        _insert_df(con, "agg_uplift_by_category", df2)

    if kpis_json.exists():
        k = json.loads(kpis_json.read_text(encoding="utf-8"))
        profit = k.get("profit", {}) if isinstance(k, dict) else {}
        row = {
            "run_id": run_id,
            "base_profit": profit.get("base_profit_gbp") or k.get("base_profit"),
            "optimised_profit": profit.get("optimised_profit_gbp")
            or k.get("optimised_profit")
            or k.get("opt_profit"),
            "constrained_profit": profit.get("constrained_profit_gbp")
            or k.get("constrained_profit"),
            "uplift_opt_pct": profit.get("uplift_unconstrained_pct")
            or k.get("uplift_opt_pct")
            or k.get("uplift_opt"),
            "uplift_con_pct": profit.get("uplift_constrained_pct")
            or k.get("uplift_con_pct")
            or k.get("uplift_con"),
        }
        con.execute(
            "INSERT INTO fact_kpis VALUES (?, ?, ?, ?, ?, ?)",
            [
                row["run_id"],
                row["base_profit"],
                row["optimised_profit"],
                row["constrained_profit"],
                row["uplift_opt_pct"],
                row["uplift_con_pct"],
            ],
        )

    dashboard_map = {
        "fact_scenario_comparison.csv": (
            "fact_scenario_comparison",
            [
                "scenario",
                "scenario_label",
                "profit_gbp",
                "uplift_gbp",
                "uplift_pct",
                "candidate_actions",
                "selected_actions",
                "selected_price_changes",
                "avg_price_change_pct",
                "avg_profit_uplift_pct",
                "budget_used_gbp",
                "forecast_winner",
                "latest_model_wmape",
                "latest_best_baseline_wmape",
            ],
        ),
        "fact_action_recommendations.csv": (
            "fact_action_recommendations",
            [
                "id",
                "store_id",
                "item_id",
                "cat_id",
                "price",
                "final_price_recommendation",
                "final_profit_projection",
                "final_profit_uplift_gbp",
                "final_profit_uplift_pct",
                "elasticity",
                "elasticity_source",
                "reason_code",
                "reason_group",
                "priority",
                "selected",
                "eligible",
                "applied_is_change",
            ],
        ),
        "agg_reason_code_mix.csv": (
            "agg_reason_code_mix",
            ["reason_code", "reason_group", "rows", "uplift_gbp"],
        ),
        "agg_store_action_summary.csv": (
            "agg_store_action_summary",
            [
                "store_id",
                "total_rows",
                "approved_rows",
                "review_rows",
                "expected_profit_uplift_gbp",
            ],
        ),
        "agg_category_action_summary.csv": (
            "agg_category_action_summary",
            ["cat_id", "total_rows", "approved_rows", "review_rows", "expected_profit_uplift_gbp"],
        ),
        "fact_uplift_backtest.csv": (
            "fact_uplift_backtest",
            [
                "cutoff_d",
                "n",
                "base_profit_gbp",
                "optimised_profit_gbp",
                "uplift_gbp",
                "uplift_pct",
            ],
        ),
        "dim_reason_codes.csv": (
            "dim_reason_codes",
            ["reason_code", "reason_group", "priority", "meaning"],
        ),
        "dim_kpi_dictionary.csv": (
            "dim_kpi_dictionary",
            ["kpi_name", "kpi_group", "definition", "interpretation"],
        ),
    }

    for filename, (table, keep_cols) in dashboard_map.items():
        path = dashboard_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
        df2 = df[keep_cols].copy()
        if table.startswith("dim_"):
            _insert_df(con, table, df2)
        else:
            df2.insert(0, "run_id", run_id)
            _insert_df(con, table, df2)

    print(f"Loaded run_id={run_id} into {args.db}")
    print('Try: duckdb analytics.duckdb -c "SELECT * FROM fact_scenario_comparison;"')


if __name__ == "__main__":
    main()
