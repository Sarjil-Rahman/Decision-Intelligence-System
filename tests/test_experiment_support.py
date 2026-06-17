from __future__ import annotations

import pandas as pd
import pytest

from m5_pipeline.ab_testing import (
    analyse_randomized_price_experiment,
    approximate_sample_size_per_group,
    assign_stratified_treatment,
    treatment_balance,
)


def _experiment_frame(effect: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(40):
        assignment = "treatment" if i % 2 == 0 else "control"
        pre = 10.0 + (i % 4)
        post = pre + (effect if assignment == "treatment" else 0.0)
        rows.append(
            {
                "unit_id": f"u{i}",
                "store_id": f"S{i % 2}",
                "cat_id": f"C{i % 3}",
                "baseline_demand_band": "mid",
                "assignment": assignment,
                "pre_metric": pre,
                "post_metric": post,
                "margin_guardrail": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_stratified_assignment_is_deterministic() -> None:
    df = _experiment_frame()[["unit_id", "store_id", "cat_id", "baseline_demand_band"]]
    a = assign_stratified_treatment(df, seed=7)
    b = assign_stratified_treatment(df, seed=7)
    assert a["assignment"].tolist() == b["assignment"].tolist()
    assert {"control", "treatment"} <= set(a["assignment"])


def test_balance_requires_both_groups() -> None:
    df = _experiment_frame()
    summary = treatment_balance(df, covariate_cols=["pre_metric"])
    assert summary["n_control"] == summary["n_treatment"]
    with pytest.raises(ValueError, match="both control and treatment"):
        treatment_balance(df.assign(assignment="control"), covariate_cols=["pre_metric"])


def test_no_effect_experiment_estimate_is_zero() -> None:
    result = analyse_randomized_price_experiment(
        _experiment_frame(effect=0.0), n_boot=100, n_perm=100
    )
    assert abs(result["estimate"]) < 1e-12
    assert result["causal_validated"] is True


def test_known_synthetic_treatment_effect_is_recovered() -> None:
    result = analyse_randomized_price_experiment(
        _experiment_frame(effect=2.5), n_boot=100, n_perm=100
    )
    assert result["estimate"] == pytest.approx(2.5)


def test_sample_size_returns_assumptions() -> None:
    out = approximate_sample_size_per_group(baseline_std=10.0, minimum_detectable_effect=2.0)
    assert out["sample_size_per_group"] > 0
    assert out["assumptions"]["method"] == "normal_approximation_two_sample_means"


def test_invalid_experiment_assignment_fails() -> None:
    with pytest.raises(ValueError, match="both treatment and control"):
        analyse_randomized_price_experiment(_experiment_frame().assign(assignment="control"))
