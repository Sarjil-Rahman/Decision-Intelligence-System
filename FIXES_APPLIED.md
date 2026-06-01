# Fixes Applied

- Fixed `/ab-test/simulate` by importing `json` and creating the output folder before writing the report.
- Fixed `/run-agent-pipeline` to call the orchestrator with `data_dir` and `params` correctly.
- Fixed `/promo-selection` default handling so `promo_discount_grid` works even when omitted.
- Fixed Prometheus scraping target from `api:8000` to `app:8000` for Docker Compose.
- Improved Streamlit dashboard defaults to use `./data/reports` and fall back to bundled pipeline outputs.
- Improved DuckDB loader defaults to use `./data/reports` and fall back to `data/price_optimization_results.csv` and `data/kpis.json`.
- Added dashboard/report aliases from the price-optimisation step: `price_actions.csv`, `price_optimization_summary.json`, and `kpis.json` under `data/reports/`.
- Added `tests/conftest.py` so `pytest -q` works from the repo root.

## Verification

- Python modules compile successfully with `python -m compileall`.
- `pytest -q` passes.
- `price_actions(...)` runs successfully on the bundled data.
- `promo_selection(...)` runs successfully with the default promo grid.
- `simulate_ab_test(...)` now writes its JSON report successfully.
- `analytics_sql/load_duckdb.py --db data/analytics.duckdb` loads successfully with the new defaults.
