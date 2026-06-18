from __future__ import annotations

import pytest

from agents.agent_guardrails import assess_forecast_gate
from m5_pipeline.pipeline import _best_baseline_wmape


def test_pipeline_summary_prefers_correct_seasonal_baseline_metric_names() -> None:
    latest = {
        "wmape_baseline_mean_28": 0.40,
        "wmape_baseline_seasonal_7": 0.22,
        "wmape_baseline_seas_7": 0.01,
        "wmape_baseline_seasonal_364": 0.31,
        "wmape_baseline_seas_364": 0.02,
    }

    assert _best_baseline_wmape(latest) == pytest.approx(0.22)


def test_pipeline_summary_supports_legacy_seas_baseline_metric_names() -> None:
    latest = {
        "wmape_baseline_mean_28": 0.40,
        "wmape_baseline_seas_7": 0.21,
        "wmape_baseline_seas_364": 0.32,
    }

    assert _best_baseline_wmape(latest) == pytest.approx(0.21)


def test_guardrails_prefer_correct_seasonal_baseline_metric_names() -> None:
    forecast_res = {
        "winner": "baseline",
        "backtests": [
            {
                "wmape_lgbm": 0.20,
                "wmape_baseline_mean_28": 0.50,
                "wmape_baseline_seasonal_7": 0.12,
                "wmape_baseline_seas_7": 0.01,
                "wmape_baseline_seasonal_364": 0.40,
                "wmape_baseline_seas_364": 0.02,
            }
        ],
    }

    ok, message = assess_forecast_gate(forecast_res)

    assert ok is True
    assert "best baseline=0.1200" in message


def test_guardrails_support_legacy_seas_baseline_metric_names() -> None:
    forecast_res = {
        "winner": "baseline",
        "backtests": [
            {
                "wmape_lgbm": 0.20,
                "wmape_baseline_mean_28": 0.50,
                "wmape_baseline_seas_7": 0.13,
                "wmape_baseline_seas_364": 0.40,
            }
        ],
    }

    ok, message = assess_forecast_gate(forecast_res)

    assert ok is True
    assert "best baseline=0.1300" in message
