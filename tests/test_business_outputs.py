from __future__ import annotations

import pandas as pd

from m5_pipeline.business_outputs import build_reason_coded_actions, build_scenario_comparison
from m5_pipeline.utils import select_representative_series_subset


def test_select_representative_series_subset_spreads_groups():
    df = pd.DataFrame(
        {
            "id": [f"A{i}" for i in range(6)],
            "store_id": ["S1", "S1", "S1", "S2", "S2", "S3"],
            "cat_id": ["C1", "C1", "C2", "C1", "C2", "C3"],
        }
    )
    out = select_representative_series_subset(df, max_series=3)
    assert len(out) == 3
    assert out["store_id"].nunique() >= 2


def test_reason_coded_actions_marks_global_fallback_for_review():
    price_df = pd.DataFrame(
        {
            "id": ["x1"],
            "store_id": ["S1"],
            "item_id": ["I1"],
            "cat_id": ["C1"],
            "price": [10.0],
            "best_price": [12.0],
            "base_profit": [100.0],
            "best_profit": [115.0],
            "profit_gain": [15.0],
            "best_delta": [0.2],
            "elasticity": [-1.2],
            "elasticity_source": ["global_fallback"],
            "bad_unit_econ": [0],
            "suspicious_uplift": [0],
        }
    )
    out = build_reason_coded_actions(price_df)
    assert out.loc[0, "reason_code"] == "REVIEW_GLOBAL_ELASTICITY_FALLBACK"
    assert out.loc[0, "reason_group"] == "review"


def test_scenario_comparison_has_three_scenarios():
    price_df = pd.DataFrame(
        {
            "id": ["x1", "x2"],
            "price": [10.0, 20.0],
            "best_price": [11.0, 19.0],
            "base_profit": [100.0, 200.0],
            "best_profit": [110.0, 210.0],
            "best_delta": [0.1, -0.05],
        }
    )
    promo_df = pd.DataFrame(
        {
            "price": [10.0, 20.0],
            "applied_price": [10.0, 19.0],
            "base_profit": [100.0, 200.0],
            "applied_profit": [100.0, 205.0],
            "selected": [0, 1],
            "eligible": [1, 1],
            "applied_is_change": [0, 1],
            "promo_spend_proxy": [0.0, 5.0],
        }
    )
    out = build_scenario_comparison(price_df, promo_df, backtests=[])
    assert set(out["scenario"]) == {
        "baseline_current_price",
        "unconstrained_price_optimizer",
        "constrained_execution_plan",
    }
