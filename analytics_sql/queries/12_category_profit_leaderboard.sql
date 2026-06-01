SELECT
  run_id,
  cat_id,
  units_sold,
  revenue_gbp,
  gross_margin_proxy_gbp,
  zero_sales_item_share
FROM mart_category_finance_kpis
ORDER BY gross_margin_proxy_gbp DESC NULLS LAST;
