SELECT
  approx_quantile(uplift_pct, 0.5) AS median_uplift_pct,
  approx_quantile(uplift_pct, 0.1) AS p10_uplift_pct,
  approx_quantile(uplift_pct, 0.9) AS p90_uplift_pct
FROM agg_uplift_by_store
WHERE run_id = $run_id;
