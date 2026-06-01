SELECT run_id, store_id, total_rows, approved_rows, review_rows, expected_profit_uplift_gbp
FROM agg_store_action_summary
ORDER BY run_id DESC, expected_profit_uplift_gbp DESC;
