SELECT
  run_id,
  grain,
  entity_id,
  date,
  metric_name,
  metric_value,
  expected_value,
  anomaly_score,
  severity,
  reason
FROM fact_kpi_anomalies
WHERE severity IN ('high', 'medium')
ORDER BY date DESC, anomaly_score DESC
LIMIT 50;
