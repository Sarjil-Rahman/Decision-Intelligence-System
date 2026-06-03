SELECT
  run_id,
  COUNT(*) AS recommendation_rows,
  SUM(CASE WHEN selected = 1 THEN 1 ELSE 0 END) AS selected_actions,
  SUM(CASE WHEN reason_group = 'review' THEN 1 ELSE 0 END) AS review_rows,
  SUM(CASE WHEN reason_group = 'approved' THEN 1 ELSE 0 END) AS approved_rows
FROM fact_action_recommendations
GROUP BY run_id
ORDER BY run_id DESC;
