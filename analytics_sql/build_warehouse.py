from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

try:
    from analytics_sql.anomaly_detection import write_kpi_anomalies
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from anomaly_detection import write_kpi_anomalies


DEFAULT_MARGIN_RATE = 0.30


def _existing_path(data_dir: Path, filename: str) -> Path | None:
    for candidate in [data_dir / filename, data_dir / filename.replace(".csv", "_backup.csv")]:
        if candidate.exists():
            return candidate
    return None


def _register_insert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.register("tmp_insert_df", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM tmp_insert_df")
    con.unregister("tmp_insert_df")


def cleanup_run_tables(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """Remove rows owned by a run before rebuilding it."""
    run_scoped_tables = [
        "fact_daily_sales",
        "fact_sell_price",
        "fact_retail_daily_kpis",
        "fact_price_actions",
        "agg_uplift_by_store",
        "agg_uplift_by_category",
        "fact_kpis",
        "fact_scenario_comparison",
        "fact_action_recommendations",
        "agg_reason_code_mix",
        "agg_store_action_summary",
        "agg_category_action_summary",
        "fact_uplift_backtest",
        "fact_kpi_anomalies",
        "runs",
        "dim_run",
    ]
    for table in run_scoped_tables:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])


def upsert_dimension_from_df(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
    key_columns: list[str],
) -> None:
    """Replace matching dimension keys without weakening primary-key constraints."""
    if df.empty:
        return
    deduped = df.drop_duplicates(subset=key_columns, keep="last").copy()
    con.register("tmp_dimension_df", deduped)
    predicates = " AND ".join(
        [f"{table_name}.{column} = tmp_dimension_df.{column}" for column in key_columns]
    )
    con.execute(f"DELETE FROM {table_name} USING tmp_dimension_df WHERE {predicates}")
    con.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_dimension_df")
    con.unregister("tmp_dimension_df")


def _normalise_day(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("d_"):
        text = text[2:]
    return int(text)


def _read_calendar(data_dir: Path, start_d: str | None, end_d: str | None) -> pd.DataFrame:
    path = _existing_path(data_dir, "calendar.csv")
    if path is None:
        return pd.DataFrame()
    calendar = pd.read_csv(path)
    start = _normalise_day(start_d)
    end = _normalise_day(end_d)
    calendar["d_num"] = calendar["d"].str.replace("d_", "", regex=False).astype(int)
    if start is not None:
        calendar = calendar[calendar["d_num"] >= start]
    if end is not None:
        calendar = calendar[calendar["d_num"] <= end]
    keep = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    for col in keep:
        if col not in calendar.columns:
            calendar[col] = None
    calendar = calendar[keep].copy()
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.date
    return calendar.sort_values("d").reset_index(drop=True)


def _read_sales(data_dir: Path, day_cols: list[str], max_series: int | None) -> pd.DataFrame:
    path = _existing_path(data_dir, "sales_train_validation.csv")
    if path is None or not day_cols:
        return pd.DataFrame()
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    usecols = meta_cols + day_cols
    sales = pd.read_csv(path, usecols=lambda col: col in usecols)
    sales = sales.sort_values("id")
    if max_series is not None and max_series > 0:
        sales = sales.head(max_series)
    return sales.reset_index(drop=True)


def _read_prices(data_dir: Path, products: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    path = _existing_path(data_dir, "sell_prices.csv")
    if path is None or products.empty or calendar.empty:
        return pd.DataFrame()
    weeks = sorted(calendar["wm_yr_wk"].dropna().astype(int).unique().tolist())
    prices = pd.read_csv(path)
    prices = prices[
        prices["store_id"].isin(products["store_id"].unique())
        & prices["item_id"].isin(products["item_id"].unique())
        & prices["wm_yr_wk"].isin(weeks)
    ].copy()
    return prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]].reset_index(drop=True)


def _build_daily_sales(sales: pd.DataFrame, day_cols: list[str], run_id: str) -> pd.DataFrame:
    if sales.empty or not day_cols:
        return pd.DataFrame(columns=["run_id", "id", "item_id", "store_id", "d", "units_sold"])
    fact = sales.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )
    fact.insert(0, "run_id", run_id)
    fact["units_sold"] = pd.to_numeric(fact["units_sold"], errors="coerce").fillna(0.0)
    return fact[["run_id", "id", "item_id", "store_id", "d", "units_sold"]]


def _build_daily_kpis(
    daily_sales: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    run_id: str,
    default_margin_rate: float = DEFAULT_MARGIN_RATE,
) -> pd.DataFrame:
    if daily_sales.empty or calendar.empty:
        return pd.DataFrame()
    df = daily_sales.merge(products[["id", "cat_id"]], on="id", how="left")
    df = df.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    if prices.empty:
        df["sell_price"] = 0.0
    else:
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].ffill()
        df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].bfill()
        df["sell_price"] = df["sell_price"].fillna(0.0)
    df["revenue_gbp"] = df["units_sold"] * df["sell_price"]
    df["gross_margin_proxy_gbp"] = df["revenue_gbp"] * default_margin_rate
    kpis = (
        df.groupby(["run_id", "d", "date", "store_id", "cat_id"], as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            revenue_gbp=("revenue_gbp", "sum"),
            gross_margin_proxy_gbp=("gross_margin_proxy_gbp", "sum"),
            active_items=("id", "nunique"),
            zero_sales_items=("units_sold", lambda s: int((s == 0).sum())),
        )
        .sort_values(["date", "store_id", "cat_id"])
    )
    kpis["avg_selling_price"] = kpis["revenue_gbp"] / kpis["units_sold"].where(
        kpis["units_sold"].abs() > 1e-9
    )
    return kpis[
        [
            "run_id",
            "d",
            "date",
            "store_id",
            "cat_id",
            "units_sold",
            "revenue_gbp",
            "gross_margin_proxy_gbp",
            "avg_selling_price",
            "active_items",
            "zero_sales_items",
        ]
    ]


def _load_business_pack(con: duckdb.DuckDBPyConnection, reports_dir: Path, run_id: str) -> None:
    dashboard_dir = reports_dir / "dashboard_ready"
    csv_map = {
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
    for filename, (table, columns) in csv_map.items():
        path = dashboard_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns].copy()
        if not table.startswith("dim_") and table not in {"dim_reason_codes", "dim_kpi_dictionary"}:
            df.insert(0, "run_id", run_id)
        if table == "dim_reason_codes":
            upsert_dimension_from_df(con, table, df, ["reason_code"])
        elif table == "dim_kpi_dictionary":
            upsert_dimension_from_df(con, table, df, ["kpi_name"])
        else:
            _register_insert(con, table, df)

    price_path = reports_dir.parent / "price_optimization_results.csv"
    if not price_path.exists():
        price_path = reports_dir / "price_actions.csv"
    if price_path.exists():
        price_df = pd.read_csv(price_path)
        rename = {
            "price": "base_price",
            "best_price": "new_price",
            "base_demand_28d": "base_demand",
            "profit_gain": "uplift_profit",
        }
        price_df = price_df.rename(columns={k: v for k, v in rename.items() if k in price_df})
        if "uplift_profit" not in price_df.columns and {"best_profit", "base_profit"} <= set(
            price_df.columns
        ):
            price_df["uplift_profit"] = price_df["best_profit"] - price_df["base_profit"]
        cols = [
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
        for col in cols:
            if col not in price_df.columns:
                price_df[col] = None
        out = price_df[cols].copy()
        out.insert(0, "run_id", run_id)
        _register_insert(con, "fact_price_actions", out)


def build_warehouse(
    *,
    data_dir: str | Path = "data",
    reports_dir: str | Path = "data/reports",
    db: str | Path = "analytics.duckdb",
    run_id: str | None = None,
    max_series: int | None = None,
    start_d: str | None = None,
    end_d: str | None = None,
    default_margin_rate: float = DEFAULT_MARGIN_RATE,
    anomaly_threshold: float = 3.5,
) -> dict[str, object]:
    run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    data_path = Path(data_dir)
    reports_path = Path(reports_dir)
    db_path = Path(db)
    con = duckdb.connect(str(db_path))
    schema_path = Path(__file__).with_name("schema.sql")
    con.execute(schema_path.read_text(encoding="utf-8"))

    cleanup_run_tables(con, run_id)
    created_at = datetime.utcnow()
    note = f"Local DuckDB warehouse; default gross margin proxy={default_margin_rate:.2%}"
    con.execute("INSERT INTO runs VALUES (?, ?, ?)", [run_id, created_at, note])
    con.execute(
        "INSERT INTO dim_run VALUES (?, ?, ?, ?)",
        [run_id, created_at, "analytics_sql.build_warehouse", note],
    )

    calendar = _read_calendar(data_path, start_d, end_d)
    day_cols = calendar["d"].tolist() if not calendar.empty else []
    sales = _read_sales(data_path, day_cols, max_series)
    products = (
        sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
        if not sales.empty
        else pd.DataFrame(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    )
    prices = _read_prices(data_path, products, calendar)
    daily_sales = _build_daily_sales(sales, day_cols, run_id)
    daily_kpis = _build_daily_kpis(
        daily_sales, products, calendar, prices, run_id, default_margin_rate
    )

    if not calendar.empty:
        upsert_dimension_from_df(con, "dim_date", calendar, ["d"])
    if not products.empty:
        upsert_dimension_from_df(
            con,
            "dim_item",
            products[["item_id", "dept_id", "cat_id"]].drop_duplicates(),
            ["item_id"],
        )
        upsert_dimension_from_df(
            con, "dim_store", products[["store_id", "state_id"]].drop_duplicates(), ["store_id"]
        )
        upsert_dimension_from_df(con, "dim_product_store", products.drop_duplicates(), ["id"])
    if not daily_sales.empty:
        _register_insert(con, "fact_daily_sales", daily_sales)
    if not prices.empty:
        prices = prices.copy()
        prices.insert(0, "run_id", run_id)
        _register_insert(con, "fact_sell_price", prices)
    if not daily_kpis.empty:
        _register_insert(con, "fact_retail_daily_kpis", daily_kpis)

    _load_business_pack(con, reports_path, run_id)
    anomalies = write_kpi_anomalies(con, run_id=run_id, threshold=anomaly_threshold)

    return {
        "db": str(db_path),
        "run_id": run_id,
        "dim_date_rows": len(calendar),
        "dim_product_store_rows": len(products),
        "fact_daily_sales_rows": len(daily_sales),
        "fact_retail_daily_kpis_rows": len(daily_kpis),
        "fact_kpi_anomalies_rows": len(anomalies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reports-dir", default="data/reports")
    parser.add_argument("--db", default="analytics.duckdb")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--start-d", default=None)
    parser.add_argument("--end-d", default=None)
    parser.add_argument("--default-margin-rate", type=float, default=DEFAULT_MARGIN_RATE)
    parser.add_argument("--anomaly-threshold", type=float, default=3.5)
    args = parser.parse_args()

    result = build_warehouse(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        db=args.db,
        run_id=args.run_id,
        max_series=args.max_series,
        start_d=args.start_d,
        end_d=args.end_d,
        default_margin_rate=args.default_margin_rate,
        anomaly_threshold=args.anomaly_threshold,
    )
    print(
        "Built DuckDB warehouse "
        f"db={result['db']} run_id={result['run_id']} "
        f"daily_sales_rows={result['fact_daily_sales_rows']} "
        f"kpi_rows={result['fact_retail_daily_kpis_rows']} "
        f"anomaly_rows={result['fact_kpi_anomalies_rows']}"
    )


if __name__ == "__main__":
    main()
