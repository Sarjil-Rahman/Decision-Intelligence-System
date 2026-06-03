-- DuckDB/Postgres-friendly schema for local retail analytics.
-- Finance values are proxy KPIs for portfolio analytics, not real company financials.

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TIMESTAMP,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS dim_run (
  run_id TEXT PRIMARY KEY,
  created_at TIMESTAMP,
  source TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
  d TEXT PRIMARY KEY,
  date DATE,
  wm_yr_wk BIGINT,
  weekday TEXT,
  wday BIGINT,
  month BIGINT,
  year BIGINT,
  event_name_1 TEXT,
  event_type_1 TEXT,
  event_name_2 TEXT,
  event_type_2 TEXT,
  snap_CA BIGINT,
  snap_TX BIGINT,
  snap_WI BIGINT
);

CREATE TABLE IF NOT EXISTS dim_item (
  item_id TEXT PRIMARY KEY,
  dept_id TEXT,
  cat_id TEXT
);

CREATE TABLE IF NOT EXISTS dim_store (
  store_id TEXT PRIMARY KEY,
  state_id TEXT
);

CREATE TABLE IF NOT EXISTS dim_product_store (
  id TEXT PRIMARY KEY,
  item_id TEXT,
  dept_id TEXT,
  cat_id TEXT,
  store_id TEXT,
  state_id TEXT
);

CREATE TABLE IF NOT EXISTS fact_daily_sales (
  run_id TEXT,
  id TEXT,
  item_id TEXT,
  store_id TEXT,
  d TEXT,
  units_sold DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_sell_price (
  run_id TEXT,
  store_id TEXT,
  item_id TEXT,
  wm_yr_wk BIGINT,
  sell_price DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_retail_daily_kpis (
  run_id TEXT,
  d TEXT,
  date DATE,
  store_id TEXT,
  cat_id TEXT,
  units_sold DOUBLE,
  revenue_gbp DOUBLE,
  gross_margin_proxy_gbp DOUBLE,
  avg_selling_price DOUBLE,
  active_items BIGINT,
  zero_sales_items BIGINT
);

CREATE TABLE IF NOT EXISTS fact_price_actions (
  run_id TEXT,
  store_id TEXT,
  item_id TEXT,
  base_price DOUBLE,
  new_price DOUBLE,
  base_demand DOUBLE,
  elasticity DOUBLE,
  base_profit DOUBLE,
  best_profit DOUBLE,
  uplift_profit DOUBLE
);

CREATE TABLE IF NOT EXISTS agg_uplift_by_store (
  run_id TEXT,
  store_id TEXT,
  base_profit DOUBLE,
  optimised_profit DOUBLE,
  uplift_pct DOUBLE
);

CREATE TABLE IF NOT EXISTS agg_uplift_by_category (
  run_id TEXT,
  cat_id TEXT,
  base_profit DOUBLE,
  optimised_profit DOUBLE,
  uplift_pct DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_kpis (
  run_id TEXT,
  base_profit DOUBLE,
  optimised_profit DOUBLE,
  constrained_profit DOUBLE,
  uplift_opt_pct DOUBLE,
  uplift_con_pct DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_scenario_comparison (
  run_id TEXT,
  scenario TEXT,
  scenario_label TEXT,
  profit_gbp DOUBLE,
  uplift_gbp DOUBLE,
  uplift_pct DOUBLE,
  candidate_actions BIGINT,
  selected_actions BIGINT,
  selected_price_changes BIGINT,
  avg_price_change_pct DOUBLE,
  avg_profit_uplift_pct DOUBLE,
  budget_used_gbp DOUBLE,
  forecast_winner TEXT,
  latest_model_wmape DOUBLE,
  latest_best_baseline_wmape DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_action_recommendations (
  run_id TEXT,
  id TEXT,
  store_id TEXT,
  item_id TEXT,
  cat_id TEXT,
  price DOUBLE,
  final_price_recommendation DOUBLE,
  final_profit_projection DOUBLE,
  final_profit_uplift_gbp DOUBLE,
  final_profit_uplift_pct DOUBLE,
  elasticity DOUBLE,
  elasticity_source TEXT,
  reason_code TEXT,
  reason_group TEXT,
  priority TEXT,
  selected BIGINT,
  eligible BIGINT,
  applied_is_change BIGINT
);

CREATE TABLE IF NOT EXISTS agg_reason_code_mix (
  run_id TEXT,
  reason_code TEXT,
  reason_group TEXT,
  rows BIGINT,
  uplift_gbp DOUBLE
);

CREATE TABLE IF NOT EXISTS agg_store_action_summary (
  run_id TEXT,
  store_id TEXT,
  total_rows BIGINT,
  approved_rows BIGINT,
  review_rows BIGINT,
  expected_profit_uplift_gbp DOUBLE
);

CREATE TABLE IF NOT EXISTS agg_category_action_summary (
  run_id TEXT,
  cat_id TEXT,
  total_rows BIGINT,
  approved_rows BIGINT,
  review_rows BIGINT,
  expected_profit_uplift_gbp DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_uplift_backtest (
  run_id TEXT,
  cutoff_d TEXT,
  n BIGINT,
  base_profit_gbp DOUBLE,
  optimised_profit_gbp DOUBLE,
  uplift_gbp DOUBLE,
  uplift_pct DOUBLE
);

CREATE TABLE IF NOT EXISTS dim_reason_codes (
  reason_code TEXT,
  reason_group TEXT,
  priority TEXT,
  meaning TEXT
);

CREATE TABLE IF NOT EXISTS dim_kpi_dictionary (
  kpi_name TEXT,
  kpi_group TEXT,
  definition TEXT,
  interpretation TEXT
);

CREATE TABLE IF NOT EXISTS fact_kpi_anomalies (
  run_id TEXT,
  grain TEXT,
  entity_id TEXT,
  d TEXT,
  date DATE,
  metric_name TEXT,
  metric_value DOUBLE,
  expected_value DOUBLE,
  anomaly_score DOUBLE,
  severity TEXT,
  reason TEXT
);

CREATE OR REPLACE VIEW mart_executive_finance_kpis AS
WITH sales AS (
  SELECT
    run_id,
    SUM(units_sold) AS units_sold,
    SUM(revenue_gbp) AS revenue_gbp,
    SUM(gross_margin_proxy_gbp) AS gross_margin_proxy_gbp,
    SUM(zero_sales_items) AS zero_sales_items,
    SUM(active_items) AS active_items
  FROM fact_retail_daily_kpis
  GROUP BY run_id
),
scenario AS (
  SELECT
    run_id,
    MAX(CASE WHEN scenario = 'baseline_current_price' THEN profit_gbp END) AS baseline_profit_gbp,
    MAX(CASE WHEN scenario = 'unconstrained_price_optimizer' THEN profit_gbp END) AS optimised_profit_gbp,
    MAX(CASE WHEN scenario = 'constrained_execution_plan' THEN profit_gbp END) AS constrained_profit_gbp,
    MAX(CASE WHEN scenario = 'constrained_execution_plan' THEN selected_actions END) AS selected_actions,
    MAX(CASE WHEN scenario = 'constrained_execution_plan' THEN avg_price_change_pct END) AS average_price_change_pct
  FROM fact_scenario_comparison
  GROUP BY run_id
),
actions AS (
  SELECT
    run_id,
    COUNT(*) AS action_rows,
    SUM(CASE WHEN reason_group = 'review' THEN 1 ELSE 0 END) AS review_rows,
    AVG(CASE WHEN elasticity_source = 'global_fallback' THEN 1.0 ELSE 0.0 END) AS global_fallback_elasticity_share,
    AVG(CASE WHEN selected = 1 THEN 1.0 ELSE 0.0 END) AS action_coverage_pct
  FROM fact_action_recommendations
  GROUP BY run_id
)
SELECT
  COALESCE(sales.run_id, scenario.run_id, actions.run_id) AS run_id,
  sales.units_sold,
  sales.revenue_gbp,
  sales.gross_margin_proxy_gbp,
  scenario.baseline_profit_gbp,
  scenario.optimised_profit_gbp,
  scenario.constrained_profit_gbp,
  COALESCE(
    scenario.constrained_profit_gbp - scenario.baseline_profit_gbp,
    scenario.optimised_profit_gbp - scenario.baseline_profit_gbp
  ) AS uplift_gbp,
  CASE
    WHEN scenario.baseline_profit_gbp IS NULL OR ABS(scenario.baseline_profit_gbp) < 1e-9 THEN NULL
    ELSE (
      COALESCE(scenario.constrained_profit_gbp, scenario.optimised_profit_gbp)
      - scenario.baseline_profit_gbp
    ) / ABS(scenario.baseline_profit_gbp) * 100.0
  END AS uplift_pct,
  scenario.selected_actions,
  actions.review_rows,
  actions.global_fallback_elasticity_share,
  scenario.average_price_change_pct,
  actions.action_coverage_pct,
  sales.zero_sales_items,
  sales.active_items
FROM sales
FULL OUTER JOIN scenario USING (run_id)
FULL OUTER JOIN actions ON actions.run_id = COALESCE(sales.run_id, scenario.run_id);

CREATE OR REPLACE VIEW mart_store_finance_kpis AS
SELECT
  run_id,
  store_id,
  SUM(units_sold) AS units_sold,
  SUM(revenue_gbp) AS revenue_gbp,
  SUM(gross_margin_proxy_gbp) AS gross_margin_proxy_gbp,
  SUM(zero_sales_items) AS zero_sales_items,
  SUM(active_items) AS active_items,
  CASE WHEN SUM(active_items) = 0 THEN NULL ELSE SUM(zero_sales_items) * 1.0 / SUM(active_items) END AS zero_sales_item_share
FROM fact_retail_daily_kpis
GROUP BY run_id, store_id;

CREATE OR REPLACE VIEW mart_category_finance_kpis AS
SELECT
  run_id,
  cat_id,
  SUM(units_sold) AS units_sold,
  SUM(revenue_gbp) AS revenue_gbp,
  SUM(gross_margin_proxy_gbp) AS gross_margin_proxy_gbp,
  SUM(zero_sales_items) AS zero_sales_items,
  SUM(active_items) AS active_items,
  CASE WHEN SUM(active_items) = 0 THEN NULL ELSE SUM(zero_sales_items) * 1.0 / SUM(active_items) END AS zero_sales_item_share
FROM fact_retail_daily_kpis
GROUP BY run_id, cat_id;

CREATE OR REPLACE VIEW mart_price_action_profit_bridge AS
SELECT
  run_id,
  SUM(base_profit) AS baseline_profit_gbp,
  SUM(best_profit) AS optimised_profit_gbp,
  SUM(uplift_profit) AS uplift_gbp,
  CASE WHEN ABS(SUM(base_profit)) < 1e-9 THEN NULL ELSE SUM(uplift_profit) / ABS(SUM(base_profit)) * 100.0 END AS uplift_pct,
  AVG(CASE WHEN ABS(base_price) < 1e-9 THEN NULL ELSE (new_price - base_price) / ABS(base_price) * 100.0 END) AS average_price_change_pct
FROM fact_price_actions
GROUP BY run_id;

CREATE OR REPLACE VIEW mart_forecast_vs_baseline_summary AS
SELECT
  run_id,
  forecast_winner,
  latest_model_wmape,
  latest_best_baseline_wmape,
  CASE
    WHEN latest_best_baseline_wmape IS NULL OR latest_best_baseline_wmape = 0 THEN NULL
    ELSE (latest_best_baseline_wmape - latest_model_wmape) / latest_best_baseline_wmape * 100.0
  END AS model_wmape_improvement_pct
FROM fact_scenario_comparison
WHERE scenario = 'baseline_current_price';

CREATE OR REPLACE VIEW mart_execution_readiness AS
SELECT
  run_id,
  COUNT(*) AS recommendation_rows,
  SUM(CASE WHEN selected = 1 THEN 1 ELSE 0 END) AS selected_actions,
  SUM(CASE WHEN reason_group = 'review' THEN 1 ELSE 0 END) AS review_rows,
  AVG(CASE WHEN elasticity_source = 'global_fallback' THEN 1.0 ELSE 0.0 END) AS global_fallback_elasticity_share,
  AVG(CASE WHEN selected = 1 THEN 1.0 ELSE 0.0 END) * 100.0 AS action_coverage_pct
FROM fact_action_recommendations
GROUP BY run_id;
