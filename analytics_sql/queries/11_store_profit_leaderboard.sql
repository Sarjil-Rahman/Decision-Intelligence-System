SELECT
  run_id,
  store_id,
  units_sold,
  revenue_gbp,
  gross_margin_proxy_gbp,
  zero_sales_item_share
FROM mart_store_finance_kpis
ORDER BY gross_margin_proxy_gbp DESC NULLS LAST;
