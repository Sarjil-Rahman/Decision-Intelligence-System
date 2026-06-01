# Business Pack Changes

## Added
- Executive KPI summary generation (`m5_pipeline/business_outputs.py`)
- Reason-coded action recommendations export
- Scenario comparison export
- Dashboard-ready reporting tables under `data/reports/dashboard_ready/`
- KPI dictionary and user guide under `data/docs/`
- Power BI / Tableau field map and starter measures/calculated fields
- Business-pack API endpoint: `POST /business-pack`
- Streamlit dashboard upgraded for executive/business views
- DuckDB schema and loader extended for dashboard-ready tables
- Dashboard mockup PNG for portfolio/demo use

## Improved
- Subset selection now uses a more representative round-robin sampler instead of plain first-N row slicing
- Price optimisation now records `elasticity_source` to expose whether each action used estimated, category-fallback, or global-fallback elasticity
- Zero-demand rows are excluded from price optimisation outputs to keep the action layer commercially relevant

## Validation
- Added tests for representative sampling, reason coding, scenario comparison, and API business-pack endpoint
- Full local test status in this environment: `7 passed`

## Truth-in-advertising
- Current sample outputs are still directional, not live-production proof
- Current forecast sample shows baseline beating the model on latest WMAPE
- Current pricing sample relies on global fallback elasticity for all recommended rows
- Current constrained promo plan selects zero live price changes
