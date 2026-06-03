SELECT
  store_id, item_id,
  AVG(new_price - base_price) AS avg_price_delta
FROM fact_price_actions
WHERE run_id = $run_id
GROUP BY 1,2
ORDER BY avg_price_delta DESC
LIMIT 20;
