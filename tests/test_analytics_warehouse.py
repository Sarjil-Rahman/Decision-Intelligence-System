from __future__ import annotations

from pathlib import Path
import json

import duckdb
import pandas as pd

from analytics_sql.build_warehouse import build_warehouse


def _write_minimal_m5(data_dir: Path) -> None:
    calendar = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 6)],
            "date": pd.date_range("2011-01-01", periods=5).astype(str),
            "wm_yr_wk": [11101] * 5,
            "weekday": ["Sat", "Sun", "Mon", "Tue", "Wed"],
            "wday": [1, 2, 3, 4, 5],
            "month": [1] * 5,
            "year": [2011] * 5,
            "event_name_1": [None] * 5,
            "event_type_1": [None] * 5,
            "event_name_2": [None] * 5,
            "event_type_2": [None] * 5,
            "snap_CA": [0] * 5,
            "snap_TX": [0] * 5,
            "snap_WI": [0] * 5,
        }
    )
    sales = pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_validation", "HOBBIES_1_001_TX_1_validation"],
            "item_id": ["FOODS_1_001", "HOBBIES_1_001"],
            "dept_id": ["FOODS_1", "HOBBIES_1"],
            "cat_id": ["FOODS", "HOBBIES"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
            "d_1": [1, 0],
            "d_2": [2, 0],
            "d_3": [3, 1],
            "d_4": [4, 1],
            "d_5": [5, 2],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "TX_1"],
            "item_id": ["FOODS_1_001", "HOBBIES_1_001"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [2.0, 5.0],
        }
    )
    calendar.to_csv(data_dir / "calendar.csv", index=False)
    sales.to_csv(data_dir / "sales_train_validation.csv", index=False)
    prices.to_csv(data_dir / "sell_prices.csv", index=False)


def test_build_warehouse_without_business_pack(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    reports_dir = data_dir / "reports"
    data_dir.mkdir()
    reports_dir.mkdir()
    _write_minimal_m5(data_dir)

    db = tmp_path / "analytics.duckdb"
    result = build_warehouse(
        data_dir=data_dir,
        reports_dir=reports_dir,
        db=db,
        run_id="test_run",
        max_series=2,
        start_d="d_1",
        end_d="d_5",
    )

    assert result["fact_daily_sales_rows"] == 10
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["run_id"] == "test_run"
    assert manifest["row_counts"]["fact_daily_sales"] == 10
    assert manifest["warnings"]
    with duckdb.connect(str(db)) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert "dim_product_store" in tables
        assert "fact_daily_sales" in tables
        assert "fact_retail_daily_kpis" in tables
        assert "mart_executive_finance_kpis" in tables
        revenue = con.execute(
            "SELECT SUM(revenue_gbp) FROM fact_retail_daily_kpis WHERE run_id = 'test_run'"
        ).fetchone()[0]
        margin = con.execute(
            "SELECT SUM(gross_margin_proxy_gbp) FROM fact_retail_daily_kpis WHERE run_id = 'test_run'"
        ).fetchone()[0]
    assert revenue == 50.0
    assert margin == 15.0


def test_build_warehouse_loads_business_pack_when_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dashboard_dir = data_dir / "reports" / "dashboard_ready"
    dashboard_dir.mkdir(parents=True)
    _write_minimal_m5(data_dir)
    pd.DataFrame(
        {
            "scenario": ["baseline_current_price", "unconstrained_price_optimizer"],
            "scenario_label": ["Baseline", "Optimised"],
            "profit_gbp": [100.0, 120.0],
            "uplift_gbp": [0.0, 20.0],
            "uplift_pct": [0.0, 20.0],
            "candidate_actions": [0, 2],
            "selected_actions": [0, 2],
            "selected_price_changes": [0, 1],
            "avg_price_change_pct": [0.0, 5.0],
            "avg_profit_uplift_pct": [0.0, 10.0],
            "budget_used_gbp": [0.0, 0.0],
            "forecast_winner": ["lgbm", "lgbm"],
            "latest_model_wmape": [0.2, 0.2],
            "latest_best_baseline_wmape": [0.3, 0.3],
        }
    ).to_csv(dashboard_dir / "fact_scenario_comparison.csv", index=False)
    pd.DataFrame(
        {
            "id": ["x1", "x2"],
            "store_id": ["CA_1", "TX_1"],
            "item_id": ["FOODS_1_001", "HOBBIES_1_001"],
            "price": [2.0, 5.0],
            "best_price": [1.8, 4.5],
            "base_demand_28d": [10.0, 20.0],
            "elasticity": [-1.5, -2.0],
            "base_profit": [10.0, 20.0],
            "best_profit": [11.0, 22.0],
            "profit_gain": [1.0, 2.0],
        }
    ).to_csv(data_dir / "price_optimization_results.csv", index=False)
    pd.DataFrame(
        {
            "id": ["x1", "x1", "x2", "x2"],
            "action_id": ["x1::a", "x1::b", "x2::a", "x2::b"],
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_001", "HOBBIES_1_001", "HOBBIES_1_001"],
            "cat_id": ["FOODS", "FOODS", "HOBBIES", "HOBBIES"],
            "price": [2.0, 2.0, 5.0, 5.0],
            "new_price": [1.8, 1.9, 4.5, 4.75],
            "base_demand_28d": [10.0, 10.0, 20.0, 20.0],
            "new_demand": [11.0, 10.5, 21.0, 20.5],
            "base_revenue": [20.0, 20.0, 100.0, 100.0],
            "new_revenue": [19.8, 19.95, 94.5, 97.375],
            "base_profit": [10.0, 10.0, 20.0, 20.0],
            "new_profit": [11.0, 10.5, 22.0, 21.0],
            "profit_gain": [1.0, 0.5, 2.0, 1.0],
            "demand_gain": [1.0, 0.5, 1.0, 0.5],
            "chosen_delta": [-0.1, -0.05, -0.1, -0.05],
            "promo_spend_proxy": [2.2, 1.05, 10.5, 5.125],
            "eligible": [1, 1, 1, 1],
            "selected": [1, 0, 0, 0],
            "is_change": [1, 1, 1, 1],
        }
    ).to_csv(data_dir / "promo_action_candidates.csv", index=False)
    pd.DataFrame(
        {
            "id": ["x1", "x2"],
            "action_id": ["x1::a", "x2::no_action"],
            "store_id": ["CA_1", "TX_1"],
            "item_id": ["FOODS_1_001", "HOBBIES_1_001"],
            "cat_id": ["FOODS", "HOBBIES"],
            "price": [2.0, 5.0],
            "applied_price": [1.8, 5.0],
            "base_demand_28d": [10.0, 20.0],
            "applied_demand": [11.0, 20.0],
            "base_revenue": [20.0, 100.0],
            "applied_revenue": [19.8, 100.0],
            "base_profit": [10.0, 20.0],
            "applied_profit": [11.0, 20.0],
            "profit_gain": [1.0, 0.0],
            "demand_gain": [1.0, 0.0],
            "chosen_delta": [-0.1, 0.0],
            "promo_spend_proxy": [2.2, 0.0],
            "selected": [1, 0],
            "eligible": [1, 1],
            "applied_is_change": [1, 0],
            "constraint_violation": [0, 0],
        }
    ).to_csv(data_dir / "promo_selection_results.csv", index=False)

    db = tmp_path / "analytics.duckdb"
    result = build_warehouse(
        data_dir=data_dir, reports_dir=data_dir / "reports", db=db, run_id="bp_run"
    )
    with duckdb.connect(str(db)) as con:
        rows = con.execute(
            "SELECT COUNT(*) FROM fact_scenario_comparison WHERE run_id = 'bp_run'"
        ).fetchone()[0]
        candidate_rows = con.execute(
            "SELECT COUNT(*) FROM fact_promo_action_candidates WHERE run_id = 'bp_run'"
        ).fetchone()[0]
        decision_rows = con.execute(
            "SELECT COUNT(*) FROM fact_promo_item_decisions WHERE run_id = 'bp_run'"
        ).fetchone()[0]
        uplift = con.execute(
            "SELECT uplift_gbp FROM mart_executive_finance_kpis WHERE run_id = 'bp_run'"
        ).fetchone()[0]
    assert rows == 2
    assert candidate_rows == 4
    assert decision_rows == 2
    assert result["manifest"]["row_counts"]["fact_promo_action_candidates"] == 4
    assert result["manifest"]["row_counts"]["fact_promo_item_decisions"] == 2
    assert uplift == 20.0


def test_build_warehouse_is_idempotent_for_same_run_id(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    reports_dir = data_dir / "reports"
    data_dir.mkdir()
    reports_dir.mkdir()
    _write_minimal_m5(data_dir)

    db = tmp_path / "analytics.duckdb"
    kwargs = {
        "data_dir": data_dir,
        "reports_dir": reports_dir,
        "db": db,
        "run_id": "demo",
        "max_series": 2,
        "start_d": "d_1",
        "end_d": "d_5",
    }

    build_warehouse(**kwargs)
    result = build_warehouse(**kwargs)

    with duckdb.connect(str(db)) as con:
        counts = dict(con.execute("""
                SELECT 'dim_date' AS table_name, COUNT(*) AS rows FROM dim_date
                UNION ALL
                SELECT 'dim_item', COUNT(*) FROM dim_item
                UNION ALL
                SELECT 'dim_store', COUNT(*) FROM dim_store
                UNION ALL
                SELECT 'dim_product_store', COUNT(*) FROM dim_product_store
                UNION ALL
                SELECT 'fact_daily_sales', COUNT(*) FROM fact_daily_sales WHERE run_id = 'demo'
                UNION ALL
                SELECT 'fact_retail_daily_kpis', COUNT(*) FROM fact_retail_daily_kpis WHERE run_id = 'demo'
                """).fetchall())
        views = {row[0] for row in con.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name IN (
                    'mart_executive_finance_kpis',
                    'mart_store_finance_kpis',
                    'mart_category_finance_kpis',
                    'mart_execution_readiness'
                  )
                """).fetchall()}

    assert counts["dim_date"] == 5
    assert counts["dim_item"] == 2
    assert counts["dim_store"] == 2
    assert counts["dim_product_store"] == 2
    assert counts["fact_daily_sales"] == 10
    assert counts["fact_retail_daily_kpis"] == 10
    assert result["manifest"]["idempotency_policy"].startswith("same run_id")
    assert {
        "mart_executive_finance_kpis",
        "mart_store_finance_kpis",
        "mart_category_finance_kpis",
        "mart_execution_readiness",
    } <= views
