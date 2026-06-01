SELECT
  cat_id,
  SUM(uplift_profit) AS profit_uplift
FROM fact_price_actions
WHERE run_id = $run_id
GROUP BY 1
HAVING SUM(uplift_profit) < 0
ORDER BY profit_uplift ASC;
