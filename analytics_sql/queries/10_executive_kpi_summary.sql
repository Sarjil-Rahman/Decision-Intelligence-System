SELECT
  run_id,
  units_sold,
  revenue_gbp,
  gross_margin_proxy_gbp,
  baseline_profit_gbp,
  optimised_profit_gbp,
  constrained_profit_gbp,
  uplift_gbp,
  uplift_pct,
  selected_actions,
  review_rows,
  global_fallback_elasticity_share,
  average_price_change_pct,
  action_coverage_pct
FROM mart_executive_finance_kpis
ORDER BY run_id DESC;
