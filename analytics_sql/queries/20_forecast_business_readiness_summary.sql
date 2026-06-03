SELECT
  e.run_id,
  f.forecast_winner,
  f.latest_model_wmape,
  f.latest_best_baseline_wmape,
  f.model_wmape_improvement_pct,
  e.recommendation_rows,
  e.selected_actions,
  e.review_rows,
  e.global_fallback_elasticity_share,
  e.action_coverage_pct
FROM mart_execution_readiness e
LEFT JOIN mart_forecast_vs_baseline_summary f USING (run_id)
ORDER BY e.run_id DESC;
