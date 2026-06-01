SELECT
  run_id,
  baseline_profit_gbp,
  optimised_profit_gbp,
  uplift_gbp,
  uplift_pct,
  average_price_change_pct
FROM mart_price_action_profit_bridge
ORDER BY run_id DESC;
