from __future__ import annotations

import builtins
import numpy as np
import pandas as pd
import pytest

from m5_pipeline.m5_promo_selection import PromoSelectionConfig, run_promo_selection


def _write_price_opt_input(tmp_path, n: int = 5, elasticity: float = -3.0) -> None:
    base_price = 10.0
    base_demand = 100.0
    cost = 6.0
    df = pd.DataFrame(
        {
            "id": [f"item_{i}_store_1_validation" for i in range(n)],
            "item_id": [f"item_{i}" for i in range(n)],
            "store_id": ["store_1"] * n,
            "cat_id": ["cat_1"] * n,
            "price": [base_price] * n,
            "cost": [cost] * n,
            "elasticity": [elasticity] * n,
            "base_demand_28d": [base_demand] * n,
            "base_profit": [(base_price - cost) * base_demand] * n,
            "best_price": [9.0] * n,
        }
    )
    df.to_csv(tmp_path / "price_optimization_results.csv", index=False)


def _selected_rows(path) -> pd.DataFrame:
    out = pd.read_csv(path)
    return out[out["selected"] == 1].copy()


def test_default_config_requires_real_discount_changes():
    cfg = PromoSelectionConfig(data_dir="data")

    assert cfg.require_price_change is True
    assert 0.0 not in cfg.promo_discount_grid
    assert all(delta < 0.0 for delta in cfg.promo_discount_grid)


def test_milp_selected_rows_have_nonzero_price_changes_and_respect_total(tmp_path):
    pytest.importorskip("pulp")
    _write_price_opt_input(tmp_path, n=5)

    res = run_promo_selection(
        PromoSelectionConfig(
            data_dir=str(tmp_path),
            max_price_changes_total=2,
            max_price_changes_per_store=None,
            max_price_changes_per_cat=None,
            promo_discount_grid=(-0.10, 0.0),
            require_price_change=True,
            write_reports=False,
        )
    )

    selected = _selected_rows(res["promo_path"])
    assert res["method"] == "milp_pulp"
    assert res["solver_status"] == "Optimal"
    assert len(selected) <= 2
    assert selected["id"].is_unique
    assert int(res["constraint_report"]["total_changes"]) <= 2
    assert (np.abs(selected["applied_price"] - selected["price"]) > 1e-9).all()
    assert (np.abs(selected["chosen_delta"]) > 1e-9).all()


def test_greedy_fallback_does_not_select_noops_when_price_change_required(tmp_path, monkeypatch):
    _write_price_opt_input(tmp_path, n=5)
    real_import = builtins.__import__

    def blocked_pulp_import(name, *args, **kwargs):
        if name == "pulp":
            raise ImportError("blocked in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_pulp_import)

    res = run_promo_selection(
        PromoSelectionConfig(
            data_dir=str(tmp_path),
            max_price_changes_total=2,
            max_price_changes_per_store=None,
            max_price_changes_per_cat=None,
            promo_discount_grid=(-0.10, 0.0),
            require_price_change=True,
            allow_greedy_fallback=True,
            write_reports=False,
        )
    )

    selected = _selected_rows(res["promo_path"])
    assert res["method"] == "greedy_fallback"
    assert len(selected) <= 2
    assert int(res["constraint_report"]["total_changes"]) <= 2
    assert (np.abs(selected["applied_price"] - selected["price"]) > 1e-9).all()
    assert (np.abs(selected["chosen_delta"]) > 1e-9).all()


def test_no_change_candidate_is_only_added_when_explicitly_allowed(tmp_path):
    _write_price_opt_input(tmp_path, n=3, elasticity=-0.5)

    res = run_promo_selection(
        PromoSelectionConfig(
            data_dir=str(tmp_path),
            max_price_changes_total=None,
            max_price_changes_per_store=None,
            max_price_changes_per_cat=None,
            promo_discount_grid=(-0.10,),
            require_price_change=False,
            write_reports=False,
        )
    )

    out = pd.read_csv(res["promo_path"])
    candidates = pd.read_csv(res["candidate_path"])
    assert len(out) == 3
    assert out["id"].is_unique
    assert len(candidates) == 6
    assert "selected" in out.columns
    assert (np.abs(out.loc[out["selected"] == 1, "chosen_delta"]) <= 1e-9).all()
    assert int(out["selected"].sum()) == 0
    assert res["summary"]["base_profit_gbp"] == pytest.approx(3 * 400.0)
    assert res["summary"]["constrained_profit_gbp"] == pytest.approx(3 * 400.0)
    assert res["summary"]["base_demand_28d"] == pytest.approx(3 * 100.0)


def test_milp_can_choose_second_best_local_action_for_global_portfolio(tmp_path):
    pytest.importorskip("pulp")
    df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "item_id": ["A", "B"],
            "store_id": ["S1", "S1"],
            "cat_id": ["C1", "C1"],
            "price": [10.0, 10.0],
            "cost": [1.0, 9.5],
            "elasticity": [-5.0, -1.0],
            "base_demand_28d": [100.0, 100.0],
            "base_profit": [900.0, 50.0],
            "best_price": [9.0, 9.0],
        }
    )
    df.to_csv(tmp_path / "price_optimization_results.csv", index=False)
    res = run_promo_selection(
        PromoSelectionConfig(
            data_dir=str(tmp_path),
            budget=120.0,
            max_price_changes_total=2,
            max_price_changes_per_store=None,
            max_price_changes_per_cat=None,
            promo_discount_grid=(-0.20, -0.10),
            objective="demand",
            write_reports=False,
        )
    )
    selected = _selected_rows(res["promo_path"])
    assert selected["id"].is_unique
    assert float((selected["selected"] * selected["promo_spend_proxy"]).sum()) <= 120.0 + 1e-6


def test_promo_outputs_split_candidates_from_one_decision_per_item(tmp_path):
    pytest.importorskip("pulp")
    _write_price_opt_input(tmp_path, n=4)

    res = run_promo_selection(
        PromoSelectionConfig(
            data_dir=str(tmp_path),
            max_price_changes_total=2,
            max_price_changes_per_store=None,
            max_price_changes_per_cat=None,
            promo_discount_grid=(-0.20, -0.10, -0.05),
            write_reports=True,
        )
    )

    decisions = pd.read_csv(res["promo_path"])
    candidates = pd.read_csv(res["candidate_path"])
    assert decisions["id"].is_unique
    assert len(decisions) == 4
    assert len(candidates) == 12
    assert res["candidate_n"] == 12
    assert res["constraint_report"]["candidate_count"] == 12
    assert decisions["base_profit"].sum() == pytest.approx(4 * 400.0)
    assert res["summary"]["base_profit_gbp"] == pytest.approx(4 * 400.0)
    assert res["summary"]["base_profit_gbp"] != pytest.approx(candidates["base_profit"].sum())
