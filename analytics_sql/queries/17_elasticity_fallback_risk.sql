SELECT
  run_id,
  elasticity_source,
  COUNT(*) AS rows,
  AVG(CASE WHEN reason_group = 'review' THEN 1.0 ELSE 0.0 END) AS review_share,
  SUM(final_profit_uplift_gbp) AS projected_uplift_gbp
FROM fact_action_recommendations
GROUP BY run_id, elasticity_source
ORDER BY run_id DESC, rows DESC;
