SELECT
  run_id,
  date,
  SUM(gross_margin_proxy_gbp) AS gross_margin_proxy_gbp,
  CASE
    WHEN SUM(revenue_gbp) = 0 THEN NULL
    ELSE SUM(gross_margin_proxy_gbp) / SUM(revenue_gbp)
  END AS gross_margin_proxy_rate
FROM fact_retail_daily_kpis
GROUP BY run_id, date
ORDER BY run_id DESC, date;
