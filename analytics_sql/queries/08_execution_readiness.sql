SELECT
  s.run_id,
  MAX(CASE WHEN s.scenario = 'baseline_current_price' THEN s.profit_gbp END) AS baseline_profit_gbp,
  MAX(CASE WHEN s.scenario = 'unconstrained_price_optimizer' THEN s.profit_gbp END) AS unconstrained_profit_gbp,
  MAX(CASE WHEN s.scenario = 'constrained_execution_plan' THEN s.profit_gbp END) AS constrained_profit_gbp,
  MAX(CASE WHEN s.scenario = 'constrained_execution_plan' THEN s.selected_price_changes END) AS constrained_price_changes,
  SUM(CASE WHEN a.reason_group = 'review' THEN 1 ELSE 0 END) AS review_rows,
  AVG(CASE WHEN a.elasticity_source = 'global_fallback' THEN 1.0 ELSE 0.0 END) AS global_fallback_share
FROM fact_scenario_comparison s
LEFT JOIN fact_action_recommendations a USING (run_id)
GROUP BY 1
ORDER BY 1 DESC;
