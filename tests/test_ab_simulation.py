from __future__ import annotations

import pandas as pd

from m5_pipeline.ab_testing import ABTestSimConfig, simulate_price_ab_test
from m5_pipeline.m5_price_optimization import _build_price_opt_summary


def test_ab_simulation_zero_uplift_when_price_is_unchanged(tmp_path):
    df = pd.DataFrame(
        {
            "price": [10.0] * 20,
            "best_price": [10.0] * 20,
            "base_demand_28d": [100.0] * 20,
            "cost": [7.0] * 20,
            "elasticity": [-1.5] * 20,
        }
    )
    csv_path = tmp_path / "price_actions.csv"
    out_path = tmp_path / "ab_test_simulation.json"
    df.to_csv(csv_path, index=False)

    report = simulate_price_ab_test(
        ABTestSimConfig(
            price_actions_csv=str(csv_path),
            out_report_json=str(out_path),
            treatment_share=0.30,
            noise_sigma=0.0,
            n_boot=100,
            seed=42,
        )
    )

    assert abs(report["uplift_pct"]["demand"]) < 1e-9
    assert abs(report["uplift_pct"]["revenue"]) < 1e-9
    assert abs(report["uplift_pct"]["profit"]) < 1e-9
    assert "revenue_ci_bootstrap" in report["uplift_pct"]
    assert "uplift_absolute" in report
    assert out_path.exists()


def test_price_opt_summary_has_dashboard_and_kpi_aliases():
    df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "store_id": ["S1", "S1"],
            "item_id": ["I1", "I2"],
            "cat_id": ["C1", "C1"],
            "base_profit": [10.0, 20.0],
            "best_profit": [12.0, 25.0],
            "profit_gain_pct": [20.0, 25.0],
            "demand_gain_pct": [5.0, 10.0],
            "suspicious_uplift": [0, 1],
            "elasticity": [-1.0, -2.0],
        }
    )

    summary = _build_price_opt_summary(df)

    assert summary["uplift_opt_pct"] == summary["uplift_pct"]
    assert summary["n_actions"] == 2
    assert summary["suspicious_uplift_rows"] == 1
    assert summary["guardrail_hits"] == 1
