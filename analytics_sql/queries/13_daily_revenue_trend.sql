SELECT
  run_id,
  date,
  SUM(units_sold) AS units_sold,
  SUM(revenue_gbp) AS revenue_gbp
FROM fact_retail_daily_kpis
GROUP BY run_id, date
ORDER BY run_id DESC, date;
