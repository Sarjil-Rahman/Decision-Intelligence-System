SELECT run_id, reason_group, reason_code, rows, uplift_gbp
FROM agg_reason_code_mix
ORDER BY run_id DESC, rows DESC, uplift_gbp DESC;
