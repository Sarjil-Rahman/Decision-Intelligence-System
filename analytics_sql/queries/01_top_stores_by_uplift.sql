SELECT
  store_id,
  SUM(uplift_profit) AS profit_uplift
FROM fact_price_actions
WHERE run_id = $run_id
GROUP BY 1
ORDER BY profit_uplift DESC
LIMIT 10;
