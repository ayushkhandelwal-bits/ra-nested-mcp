"""Unit tests for mcp4_forecasting.forecasting.

Run with: PYTHONPATH=src pytest tests/test_forecasting.py -v
"""

import pytest

from mcp4_forecasting import forecasting as f
from mcp4_forecasting.data_access import load_daily_leakage_series


def test_compare_forecast_models_uses_chronological_split():
    result = f.compare_forecast_models(test_days=30)
    train_end = result["train_period"]["end"]
    test_start = result["test_period"]["start"]
    # train must end strictly before test begins (chronological, non-overlapping, non-random)
    assert train_end < test_start
    assert result["test_period"]["n"] == 30


def test_compare_forecast_models_reports_both_models_and_a_winner():
    result = f.compare_forecast_models(test_days=30)
    assert set(result["models"].keys()) == {"seasonal_naive", "holt_winters"}
    for m in result["models"].values():
        assert m["mae"] >= 0 and m["rmse"] >= 0
    assert result["better_model"] in result["models"]


def test_compare_forecast_models_rejects_too_large_holdout():
    series = load_daily_leakage_series()
    with pytest.raises(ValueError):
        f.compare_forecast_models(test_days=len(series))  # leaves no training data


def test_forecast_revenue_leakage_shape_and_non_negativity():
    result = f.forecast_revenue_leakage(steps=30)
    assert result["forecast_horizon_days"] == 30
    assert len(result["forecast"]) == 30
    for row in result["forecast"]:
        assert row["forecast"] >= 0
        # revenue_leakage cannot be negative — intervals should be clipped too
        assert row["lower_95"] >= 0
        assert row["lower_80"] >= row["lower_95"]
        assert row["upper_95"] >= row["upper_80"]


def test_forecast_revenue_leakage_intervals_widen_with_horizon():
    result = f.forecast_revenue_leakage(steps=30)
    early_width = result["forecast"][0]["upper_95"] - result["forecast"][0]["lower_95"]
    late_width = result["forecast"][-1]["upper_95"] - result["forecast"][-1]["lower_95"]
    # uncertainty should generally grow (or at least not shrink dramatically) further out
    assert late_width >= early_width * 0.7


def test_forecast_revenue_leakage_includes_limitations_and_direction():
    result = f.forecast_revenue_leakage(steps=14)
    assert result["expected_direction"] in ("increasing", "decreasing", "stable")
    assert "not a guarantee" in result["limitations"].lower() or "projection" in result["limitations"].lower()


def test_get_forecast_accuracy_daily_errors_match_test_days():
    result = f.get_forecast_accuracy(model="holt_winters", test_days=20)
    assert len(result["daily_errors"]) == 20
    for row in result["daily_errors"]:
        # allow 1-cent tolerance: abs_error is computed from unrounded values server-side,
        # this recomputes from the already-rounded actual/predicted fields
        assert abs(row["abs_error"] - round(abs(row["actual"] - row["predicted"]), 2)) <= 0.02


def test_get_forecast_accuracy_rejects_unknown_model():
    with pytest.raises(ValueError):
        f.get_forecast_accuracy(model="not_a_real_model")


def test_get_forecast_intervals_is_slim_wrapper():
    result = f.get_forecast_intervals(steps=10)
    assert len(result["forecast"]) == 10
    assert "limitations" not in result  # slim variant should not include narrative fields
