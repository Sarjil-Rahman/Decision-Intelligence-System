# SQL Analytics Case Study

## Business Problem

This project uses M5-style retail data to show how forecasting and price optimisation outputs can be turned into an analytics warehouse for commercial decision support. The SQL layer answers executive questions about stores, categories, proxy profit bridges, review queues, elasticity risk, and anomalous KPI movements.

The warehouse is local-first and intentionally lightweight: DuckDB, pandas, and SQL files. It is not presented as a live enterprise warehouse.

## Warehouse Design

`analytics_sql/build_warehouse.py` creates a DuckDB database from local M5 inputs and optional business-pack outputs.

Core star schema:

- `dim_date`: calendar, week, event, and SNAP fields keyed by `d`.
- `dim_item`: item, department, and category.
- `dim_store`: store and state.
- `dim_product_store`: the M5 product-store series grain.
- `dim_run`: build/run metadata.
- `fact_daily_sales`: daily units by product-store series.
- `fact_sell_price`: weekly sell price by store and item.
- `fact_retail_daily_kpis`: daily store/category KPI fact.

Optional ML/business-output tables are loaded when business-pack files exist, including scenario comparison, action recommendations, uplift backtests, reason-code mix, store/category action summaries, reason-code reference data, and the KPI dictionary.

The warehouse still builds when these files are missing, so CI and local samples do not depend on running the full ML pipeline first.

## KPI Definitions

M5 does not include real COGS, inventory, or audited company finance data. Finance fields are therefore proxy KPIs.

- `revenue_gbp`: `units_sold * sell_price`.
- `gross_margin_proxy_gbp`: revenue multiplied by the default gross margin assumption, currently 30%, unless richer optimisation outputs are available.
- `baseline_profit_gbp`: projected profit at current price from optimisation outputs.
- `optimised_profit_gbp`: projected unconstrained optimiser profit.
- `constrained_profit_gbp`: projected profit after execution constraints.
- `uplift_gbp`: constrained or optimised profit minus baseline profit.
- `uplift_pct`: uplift divided by absolute baseline profit.
- `global_fallback_elasticity_share`: share of recommendations using the global fallback elasticity.
- `action_coverage_pct`: selected recommendation share.

These values are useful for portfolio analytics and interview discussion, but they are not real company financial statements.

## KPI Marts

The schema creates reusable DuckDB marts:

- `mart_executive_finance_kpis`
- `mart_store_finance_kpis`
- `mart_category_finance_kpis`
- `mart_price_action_profit_bridge`
- `mart_forecast_vs_baseline_summary`
- `mart_execution_readiness`

These marts connect raw M5 sales and price data to ML outputs such as scenario comparison, recommendation review queues, elasticity fallback risk, and forecast readiness.

## Anomaly Detection

`analytics_sql/anomaly_detection.py` creates `fact_kpi_anomalies` from `fact_retail_daily_kpis`.

The method is deliberately explainable:

- group by store/day and category/day;
- compute rolling median as the expected value;
- use median absolute deviation, with standard deviation fallback, as the scale;
- flag rows whose robust score exceeds the configured threshold.

Detected metrics include revenue, units, gross margin proxy, and zero-sales item share. Each anomaly includes run, grain, entity, day/date, metric, observed value, expected value, score, severity, and a plain-English reason.

## SQL Query Pack

Reusable queries live in `analytics_sql/queries/`, including executive KPI summary, store/category profit leaderboards, daily revenue and margin trends, price action bridge, selected versus reviewed recommendations, elasticity fallback risk, anomaly summaries, and readiness checks.

Example:

```sql
SELECT *
FROM mart_executive_finance_kpis
ORDER BY run_id DESC;
```

## Dashboard Connection

`dashboards/streamlit_app.py` can read directly from DuckDB when `analytics.duckdb` or another DB path is provided. If DuckDB is unavailable, it falls back to the existing CSV business-pack files.

Dashboard tabs include Executive Summary, Finance KPIs, Store Performance, Category Performance, Price Action Bridge, Anomalies, and Data/Model Readiness.

## Run Locally

Build a small local warehouse:

```bash
python analytics_sql/build_warehouse.py --data-dir data --reports-dir data/reports --db analytics.duckdb --run-id demo --max-series 100 --start-d d_1 --end-d d_90
```

Run anomaly detection separately if needed:

```bash
python analytics_sql/anomaly_detection.py --db analytics.duckdb --run-id demo --threshold 3.5
```

Open the Streamlit dashboard:

```bash
streamlit run dashboards/streamlit_app.py
```

Run focused tests:

```bash
pytest tests/test_analytics_warehouse.py tests/test_anomaly_detection.py
```

## What Is Real vs Proxy

Real:

- M5 sales/calendar/price structure.
- DuckDB star schema and SQL marts.
- Deterministic local build path.
- Business-pack ingestion when generated outputs exist.
- Explainable anomaly detection over warehouse KPIs.

Proxy:

- GBP labels are portfolio-friendly naming, not audited financial currency conversion.
- Gross margin uses an assumed default rate where real COGS is unavailable.
- Optimised/constrained profit depends on the existing price optimisation simulation outputs.
- Readiness and elasticity risk are decision-support signals, not production approval gates.

## Limitations

- No real COGS, inventory, supplier funding, or promotion cost accounting.
- No live experiment design or causal inference layer.
- No orchestration tool; runs are local commands.
- No claim that price recommendations are safe to execute without commercial validation.
- Full M5 expansion can be large, so CI uses small synthetic samples and `--max-series`.

## Interview Framing

Use this case study to show the path from model output to decision intelligence: forecasting estimates demand, optimisation proposes actions, constraints and reason codes create an analyst-reviewable plan, DuckDB models facts/dimensions/marts, and SQL plus Streamlit expose executive and operating views.

The strongest framing is honest: this is a credible local analytics and ML decision-support system, not a fake enterprise platform.
