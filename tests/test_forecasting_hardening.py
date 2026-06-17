from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5_pipeline.m5_forecasting import (
    BacktestSplit,
    ForecastConfig,
    PromotionPolicy,
    _add_price_feats,
    _fit_lgbm_with_inner_split,
    _fill_sell_price_future,
    _fill_sell_price_train,
    _recursive_feature_rows,
    apply_conformal_intervals,
    build_conformal_interval_summary,
    evaluate_promotion_policy,
    finite_sample_conformal_quantile,
    make_backtest_splits,
    recursive_evaluate_split,
)


def test_last_window_returns_exactly_one_most_recent_split() -> None:
    splits = make_backtest_splits(
        split_strategy="last_window",
        start_day=1,
        last_train_day=100,
        horizon=7,
        n_backtests=5,
        stride=7,
        max_required_history=28,
    )
    assert len(splits) == 1
    assert splits[0].train_end_d == "d_93"
    assert splits[0].valid_start_d == "d_94"
    assert splits[0].valid_end_d == "d_100"


def test_rolling_origin_returns_recent_first_and_respects_stride() -> None:
    splits = make_backtest_splits(
        split_strategy="rolling_origin",
        start_day=1,
        last_train_day=120,
        horizon=10,
        n_backtests=3,
        stride=14,
        max_required_history=28,
    )
    assert [s.valid_end_d for s in splits] == ["d_120", "d_106", "d_92"]
    assert all(int(s.train_end_d[2:]) < int(s.valid_start_d[2:]) for s in splits)


def test_impossible_split_configuration_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="not enough history"):
        make_backtest_splits(
            split_strategy="last_window",
            start_day=1,
            last_train_day=30,
            horizon=10,
            n_backtests=1,
            stride=7,
            max_required_history=28,
        )


def test_leading_missing_prices_stay_missing_and_later_price_does_not_backfill() -> None:
    df = pd.DataFrame(
        {
            "id": ["A", "A", "A", "B", "B"],
            "date": pd.date_range("2020-01-01", periods=5),
            "sell_price": [np.nan, np.nan, 3.0, np.nan, 9.0],
        }
    )
    out = _fill_sell_price_train(df)
    assert out.loc[out["id"].eq("A"), "sell_price"].iloc[:2].isna().all()
    assert pd.isna(out.loc[out["id"].eq("B"), "sell_price"].iloc[0])
    assert set(out["price_imputation_source"]) >= {"missing_leading", "observed"}


def test_future_price_fill_uses_last_training_price_before_training_stat() -> None:
    train = pd.DataFrame(
        {
            "id": ["A", "A", "B"],
            "date": pd.date_range("2020-01-01", periods=3),
            "sell_price": [2.0, 3.0, np.nan],
        }
    )
    future = pd.DataFrame(
        {
            "id": ["A", "B"],
            "date": pd.date_range("2020-02-01", periods=2),
            "sell_price": [np.nan, np.nan],
        }
    )
    out = _fill_sell_price_future(future, train)
    assert out.loc[out["id"].eq("A"), "sell_price"].iloc[0] == 3.0
    assert out.loc[out["id"].eq("A"), "price_imputation_source"].iloc[0] == "last_training_price"
    assert (
        out.loc[out["id"].eq("B"), "price_imputation_source"].iloc[0] == "training_median_fallback"
    )


def test_missing_prices_do_not_create_false_price_change_events() -> None:
    df = pd.DataFrame(
        {
            "id": ["A", "A", "A"],
            "date": pd.date_range("2020-01-01", periods=3),
            "sell_price": [np.nan, 5.0, 5.0],
        }
    )
    out = _add_price_feats(df)
    assert out["price_changed_today"].tolist() == [0.0, 0.0, 0.0]


class EchoLagModel:
    def predict(self, X):
        return X["lag_1"].fillna(0.0).to_numpy(float)


def _recursive_fixture() -> pd.DataFrame:
    days = pd.date_range("2020-01-01", periods=6)
    return pd.DataFrame(
        {
            "id": ["A"] * 6,
            "d": [f"d_{i}" for i in range(1, 7)],
            "date": days,
            "sales": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            "sell_price": [1.0] * 6,
            "price_was_missing": [0] * 6,
            "snap": [0] * 6,
            "wday": [1, 2, 3, 4, 5, 6],
            "month": [1] * 6,
            "year": [2020] * 6,
            "item_id": ["I"] * 6,
            "dept_id": ["D"] * 6,
            "cat_id": ["C"] * 6,
            "store_id": ["S"] * 6,
            "state_id": ["ST"] * 6,
            "weekday": ["Mon"] * 6,
            "event_name_1": [None] * 6,
            "event_type_1": [None] * 6,
            "event_name_2": [None] * 6,
            "event_type_2": [None] * 6,
        }
    )


def test_recursive_predictions_use_previous_predictions_not_validation_actuals() -> None:
    frame = _recursive_fixture()
    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    for col in cat_cols:
        frame[col] = pd.Categorical(frame[col].astype("object").fillna("none").astype(str))
    cfg = ForecastConfig(
        data_dir="unused", lags=(1,), rolls=(1,), nonzero_rolls=(1,), two_stage=False
    )
    eval_rows = recursive_evaluate_split(
        frame,
        split=BacktestSplit("d_3", "d_4", "d_6"),
        model_obj=EchoLagModel(),
        cfg=cfg,
        feature_cols=["lag_1"],
        cat_cols=cat_cols,
        category_reference=frame,
        origin_date=frame["date"].min(),
    )
    assert eval_rows["pred_lgbm"].tolist() == [3.0, 3.0, 3.0]
    assert eval_rows["actual"].tolist() == [100.0, 200.0, 300.0]


def test_recursive_feature_builder_preserves_original_price_missingness_and_parity() -> None:
    cfg = ForecastConfig(
        data_dir="unused", lags=(1,), rolls=(2,), nonzero_rolls=(2,), two_stage=False
    )
    day_rows = pd.DataFrame(
        {
            "id": ["A"],
            "date": [pd.Timestamp("2020-01-04")],
            "sell_price": [5.0],
            "price_was_missing": [1],
            "price_imputation_source": ["last_training_price"],
            "snap": [0],
            "wday": [4],
            "month": [1],
            "year": [2020],
            "item_id": ["I"],
            "dept_id": ["D"],
            "cat_id": ["C"],
            "store_id": ["S"],
            "state_id": ["ST"],
            "weekday": ["Thu"],
            "event_name_1": ["none"],
            "event_type_1": ["none"],
            "event_name_2": ["none"],
            "event_type_2": ["none"],
        }
    )
    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    hist_sales = {"A": [1.0, 2.0, 3.0]}
    hist_price = {"A": [4.0, 4.0, 5.0]}
    first = _recursive_feature_rows(
        day_rows,
        hist_sales={k: list(v) for k, v in hist_sales.items()},
        hist_price={k: list(v) for k, v in hist_price.items()},
        cfg=cfg,
        origin_date=pd.Timestamp("2020-01-01"),
        cat_cols=cat_cols,
    )
    second = _recursive_feature_rows(
        day_rows.copy(),
        hist_sales={k: list(v) for k, v in hist_sales.items()},
        hist_price={k: list(v) for k, v in hist_price.items()},
        cfg=cfg,
        origin_date=pd.Timestamp("2020-01-01"),
        cat_cols=cat_cols,
    )
    assert first == second
    assert first[0]["price_was_missing"] == 1
    assert first[0]["price_imputation_source"] == "last_training_price"


def test_lightgbm_resource_config_reaches_estimators(monkeypatch) -> None:
    class FakeClassifier:
        instances = []

        def __init__(self, **params):
            self.params = params
            self.best_iteration_ = 7
            FakeClassifier.instances.append(self)

        def fit(self, *args, **kwargs):
            return self

        def get_params(self):
            return dict(self.params)

    class FakeRegressor:
        instances = []

        def __init__(self, **params):
            self.params = params
            self.best_iteration_ = 9
            FakeRegressor.instances.append(self)

        def fit(self, *args, **kwargs):
            return self

        def get_params(self):
            return dict(self.params)

    class FakeLgb:
        LGBMClassifier = FakeClassifier
        LGBMRegressor = FakeRegressor

        @staticmethod
        def early_stopping(rounds, verbose=False):
            return ("early_stopping", rounds, verbose)

    import m5_pipeline.m5_forecasting as forecasting

    monkeypatch.setattr(forecasting, "lgb", FakeLgb)
    train_df = pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 9)],
            "sales": [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 2.0],
            "x": [float(i) for i in range(1, 9)],
        }
    )
    cfg = ForecastConfig(
        data_dir="unused",
        two_stage=True,
        classifier_n_estimators=123,
        regressor_n_estimators=234,
        classifier_early_stopping_rounds=11,
        regressor_early_stopping_rounds=22,
        n_jobs=2,
        lightgbm_verbosity=1,
    )
    model, metadata = _fit_lgbm_with_inner_split(
        train_df,
        cfg=cfg,
        feature_cols=["x"],
        split=BacktestSplit("d_8", "d_9", "d_10", "d_4", "d_5", "d_6"),
    )
    clf, reg = model
    assert FakeClassifier.instances[0].params["n_estimators"] == 123
    assert FakeRegressor.instances[0].params["n_estimators"] == 234
    assert clf.params["n_estimators"] == 7
    assert reg.params["n_estimators"] == 9
    assert clf.params["n_jobs"] == 2
    assert reg.params["verbosity"] == 1
    assert metadata["classifier_early_stopping_status"] == "used"
    assert metadata["regressor_early_stopping_status"] == "used"


def test_promotion_policy_rejects_low_aggregate_improvement() -> None:
    diag = [
        {
            "split": {},
            "row_count": 10,
            "actual_abs_sum": 100.0,
            "lgbm_abs_error_sum": 10.0,
            "baseline_mean_28_abs_error_sum": 10.1,
            "baseline_seasonal_7_abs_error_sum": 12.0,
            "baseline_seasonal_364_abs_error_sum": 13.0,
            "wmape_lgbm": 0.10,
            "wmape_baseline_mean_28": 0.101,
            "wmape_baseline_seasonal_7": 0.12,
            "wmape_baseline_seasonal_364": 0.13,
            "segments": {
                name: {
                    "actual_abs_sum": 0.0,
                    "lgbm_abs_error_sum": 0.0,
                    "baseline_mean_28_abs_error_sum": 0.0,
                    "baseline_seasonal_7_abs_error_sum": 0.0,
                    "baseline_seasonal_364_abs_error_sum": 0.0,
                }
                for name in ("event", "non_event", "price_drop", "non_price_drop")
            },
        }
    ]
    decision = evaluate_promotion_policy(
        diag,
        PromotionPolicy(
            promotion_min_backtests=1,
            promotion_min_aggregate_improvement_pct=2.0,
            allow_single_split_promotion=True,
        ),
    )
    assert decision["winner"] == "baseline"
    assert "insufficient_aggregate_improvement" in decision["promotion_reasons"]


def test_conformal_quantile_index_and_interval_bounds() -> None:
    assert finite_sample_conformal_quantile([1, 2, 3, 4], alpha=0.2) == 4.0
    eval_rows = pd.DataFrame(
        {
            "split_index": [1, 1, 0, 0],
            "horizon_step": [1, 2, 1, 2],
            "actual": [10.0, 20.0, 12.0, 25.0],
            "pred_lgbm": [9.0, 18.0, 10.0, 20.0],
        }
    )
    summary = build_conformal_interval_summary(eval_rows, prediction_col="pred_lgbm")
    intervals = apply_conformal_intervals(
        pd.DataFrame({"id": ["A"], "horizon_step": [1], "pred": [1.0]}), summary
    )
    assert intervals.loc[0, "lower"] >= 0.0
    assert intervals.loc[0, "lower"] <= intervals.loc[0, "pred"] <= intervals.loc[0, "upper"]
    assert summary["heldout_split_index"] == 0
