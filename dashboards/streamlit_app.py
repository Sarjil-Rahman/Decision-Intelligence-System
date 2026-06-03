from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - Streamlit environment issue
    duckdb = None


st.set_page_config(page_title="Retail Decision Intelligence Dashboard", layout="wide")
st.title("Retail Decision Intelligence Dashboard")
st.caption(
    "DuckDB warehouse marts with CSV fallback. Finance KPIs are proxy portfolio metrics, "
    "not real company financials."
)

db_path = Path(st.sidebar.text_input("DuckDB path", value="analytics.duckdb"))
reports_root = Path(st.sidebar.text_input("Reports folder", value="data/reports"))
dashboard_dir = reports_root / "dashboard_ready"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def query_duckdb(db: str, sql: str) -> pd.DataFrame:
    if duckdb is None or not Path(db).exists():
        return pd.DataFrame()
    with duckdb.connect(db, read_only=True) as con:
        return con.execute(sql).fetchdf()


def table_or_info(df: pd.DataFrame, message: str) -> None:
    if df.empty:
        st.info(message)
    else:
        st.dataframe(df, use_container_width=True)


use_duckdb = duckdb is not None and db_path.exists()
if use_duckdb:
    st.sidebar.success("Reading DuckDB marts")
else:
    st.sidebar.warning("DuckDB unavailable; using CSV fallback")


if use_duckdb:
    executive_df = query_duckdb(str(db_path), "SELECT * FROM mart_executive_finance_kpis")
    store_df = query_duckdb(str(db_path), "SELECT * FROM mart_store_finance_kpis")
    category_df = query_duckdb(str(db_path), "SELECT * FROM mart_category_finance_kpis")
    bridge_df = query_duckdb(str(db_path), "SELECT * FROM mart_price_action_profit_bridge")
    readiness_df = query_duckdb(str(db_path), "SELECT * FROM mart_execution_readiness")
    anomalies_df = query_duckdb(str(db_path), "SELECT * FROM fact_kpi_anomalies")
    daily_df = query_duckdb(
        str(db_path),
        """
        SELECT date, SUM(revenue_gbp) AS revenue_gbp, SUM(gross_margin_proxy_gbp) AS margin_gbp
        FROM fact_retail_daily_kpis
        GROUP BY date
        ORDER BY date
        """,
    )
else:
    executive = load_json(reports_root / "executive_kpi_summary.json")
    executive_df = pd.DataFrame([executive.get("headline", {})]) if executive else pd.DataFrame()
    store_df = load_csv(dashboard_dir / "agg_store_action_summary.csv")
    category_df = load_csv(dashboard_dir / "agg_category_action_summary.csv")
    bridge_df = load_csv(reports_root / "scenario_comparison.csv")
    readiness_df = load_csv(dashboard_dir / "agg_reason_code_mix.csv")
    anomalies_df = pd.DataFrame()
    daily_df = pd.DataFrame()


tab_names = [
    "Executive Summary",
    "Finance KPIs",
    "Store Performance",
    "Category Performance",
    "Price Action Bridge",
    "Anomalies",
    "Data/Model Readiness",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.subheader("Executive Summary")
    if executive_df.empty:
        st.info("Build the DuckDB warehouse or generate the business pack to populate this view.")
    else:
        latest = executive_df.sort_values("run_id").tail(1).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Revenue proxy", f"GBP {latest.get('revenue_gbp', 0) or 0:,.0f}")
        c2.metric("Margin proxy", f"GBP {latest.get('gross_margin_proxy_gbp', 0) or 0:,.0f}")
        c3.metric("Units", f"{latest.get('units_sold', 0) or 0:,.0f}")
        c4.metric("Uplift proxy", f"GBP {latest.get('uplift_gbp', 0) or 0:,.0f}")
        c5.metric("Review rows", f"{latest.get('review_rows', 0) or 0:,.0f}")
        table_or_info(executive_df, "No executive KPI rows found.")
    if not daily_df.empty:
        st.line_chart(daily_df.set_index("date")[["revenue_gbp", "margin_gbp"]])

with tabs[1]:
    st.subheader("Finance KPIs")
    st.caption(
        "Gross margin and profit values are proxy KPIs using available optimisation outputs or an assumed margin rate."
    )
    table_or_info(executive_df, "No finance KPI mart rows found.")

with tabs[2]:
    st.subheader("Store Performance")
    if not store_df.empty and "gross_margin_proxy_gbp" in store_df.columns:
        store_df = store_df.sort_values("gross_margin_proxy_gbp", ascending=False)
    table_or_info(store_df, "No store performance rows found.")

with tabs[3]:
    st.subheader("Category Performance")
    if not category_df.empty and "gross_margin_proxy_gbp" in category_df.columns:
        category_df = category_df.sort_values("gross_margin_proxy_gbp", ascending=False)
    table_or_info(category_df, "No category performance rows found.")

with tabs[4]:
    st.subheader("Price Action Bridge")
    table_or_info(bridge_df, "No price action bridge rows found.")

with tabs[5]:
    st.subheader("Anomalies")
    if anomalies_df.empty:
        st.info("No KPI anomalies found or anomaly detection has not been run.")
    else:
        severity = st.multiselect(
            "Severity", sorted(anomalies_df["severity"].dropna().unique()), default=None
        )
        filtered = anomalies_df
        if severity:
            filtered = filtered[filtered["severity"].isin(severity)]
        st.dataframe(
            filtered.sort_values(["date", "anomaly_score"], ascending=[False, False]),
            use_container_width=True,
        )

with tabs[6]:
    st.subheader("Data/Model Readiness")
    if not readiness_df.empty:
        table_or_info(readiness_df, "No readiness rows found.")
    else:
        st.info(
            "Business-pack readiness tables are optional; build the business pack for this view."
        )
