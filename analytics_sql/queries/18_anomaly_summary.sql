SELECT
  run_id,
  grain,
  metric_name,
  severity,
  COUNT(*) AS anomaly_rows,
  MAX(anomaly_score) AS max_anomaly_score
FROM fact_kpi_anomalies
GROUP BY run_id, grain, metric_name, severity
ORDER BY run_id DESC, anomaly_rows DESC;
