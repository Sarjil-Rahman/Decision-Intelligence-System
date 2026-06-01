# Power BI / Tableau Dashboard Field Map

Use `data/reports/dashboard_ready/` as the import folder.

## Core tables
- `fact_scenario_comparison.csv`: executive scenario cards and waterfall-style comparisons
- `fact_action_recommendations.csv`: detailed action list with reason codes and governance flags
- `agg_reason_code_mix.csv`: reason-code bar charts and review-vs-approved mix
- `agg_store_action_summary.csv`: store leaderboard
- `agg_category_action_summary.csv`: category leaderboard
- `fact_uplift_backtest.csv`: scenario stability over historical cutoffs
- `dim_reason_codes.csv`: friendly descriptions for reason codes
- `dim_kpi_dictionary.csv`: tooltip / glossary source

## Recommended visuals
1. KPI cards
   - Baseline profit
   - Unconstrained profit
   - Constrained profit
   - Approved price changes
   - Review rows
2. Clustered column chart
   - Axis: `scenario_label`
   - Value: `profit_gbp`
3. Horizontal bar chart
   - Axis: `reason_code`
   - Value: `rows`
   - Colour: `reason_group`
4. Table
   - `store_id`, `expected_profit_uplift_gbp`, `approved_rows`, `review_rows`
5. Scatter plot
   - X: `final_profit_uplift_pct`
   - Y: `price_change_pct`
   - Details: `item_id`
   - Legend: `reason_group`
6. Line chart
   - Axis: `cutoff_d`
   - Value: `uplift_pct`

## Join logic
- `fact_action_recommendations.reason_code` -> `dim_reason_codes.reason_code`
- `fact_scenario_comparison` stands alone
- `agg_*` tables are already dashboard-ready and normally do not need joins

## Power BI DAX starter measures
See `dashboards/power_bi_tableau/power_bi_measures.dax`.

## Tableau starter calculated fields
See `dashboards/power_bi_tableau/tableau_calculated_fields.txt`.
