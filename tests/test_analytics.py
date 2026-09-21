"""Unit tests for mcp3_analytics.analytics.

Run with: PYTHONPATH=src pytest tests/test_analytics.py -v
Requires data/raw and data/processed to be populated (see generate_data.py
in the data-generation package, or point data_access at your own files).
"""

import pytest

from mcp3_analytics import analytics as a


def test_get_kpi_metrics_full_history_is_internally_consistent():
    kpis = a.get_kpi_metrics()
    assert kpis["total_expected_revenue"] > 0
    assert kpis["total_billed_revenue"] > 0
    assert 0 <= kpis["leakage_rate"] < 1
    # leakage should be a small fraction of expected revenue for a healthy operator
    assert kpis["leakage_rate"] < 0.15
    # billed + leakage should roughly reconcile back to expected revenue
    approx_expected = kpis["total_billed_revenue"] + kpis["total_revenue_leakage"]
    assert abs(approx_expected - kpis["total_expected_revenue"]) / kpis["total_expected_revenue"] < 0.05


def test_get_kpi_metrics_date_filter_narrows_period():
    full = a.get_kpi_metrics()
    narrow = a.get_kpi_metrics(start_date="2025-01-01", end_date="2025-01-31")
    assert narrow["period"]["days"] <= 31
    assert narrow["total_cdr_volume"] < full["total_cdr_volume"]


def test_get_kpi_metrics_invalid_range_raises():
    with pytest.raises(ValueError):
        a.get_kpi_metrics(start_date="2099-01-01", end_date="2099-01-31")


def test_detect_anomalies_zscore_returns_valid_shape():
    result = a.detect_anomalies(metric="revenue_leakage", method="zscore", window=14, threshold=3.0)
    assert result["n_anomalies"] == len(result["anomalies"])
    for anomaly in result["anomalies"]:
        assert {"date", "metric", "observed_value", "baseline_value", "deviation", "anomaly_score", "severity"} <= anomaly.keys()
        assert anomaly["severity"] in ("medium", "high", "critical")
        assert abs(anomaly["anomaly_score"]) >= 3.0


def test_detect_anomalies_higher_threshold_finds_fewer_or_equal():
    loose = a.detect_anomalies(metric="revenue_leakage", method="zscore", threshold=2.0)
    strict = a.detect_anomalies(metric="revenue_leakage", method="zscore", threshold=4.0)
    assert strict["n_anomalies"] <= loose["n_anomalies"]


def test_detect_anomalies_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        a.detect_anomalies(metric="not_a_real_column")


def test_detect_anomalies_rejects_unsupported_method():
    with pytest.raises(ValueError):
        a.detect_anomalies(method="not_a_real_method")


def test_run_hypothesis_test_requires_split_date():
    with pytest.raises(ValueError):
        a.run_hypothesis_test(metric="revenue_leakage", split_date=None)


def test_run_hypothesis_test_returns_required_fields():
    result = a.run_hypothesis_test(metric="revenue_leakage", split_date="2025-06-20",
                                    start_date="2025-06-06", end_date="2025-07-04")
    required = {"test_used", "statistic", "p_value", "null_hypothesis", "alternative_hypothesis",
                "statistically_significant", "business_significant_gt10pct", "conclusion", "caveat"}
    assert required <= result.keys()
    assert 0 <= result["p_value"] <= 1
    assert result["test_used"] in ("Welch's t-test (unequal variance)", "Mann-Whitney U (non-parametric)")


def test_analyze_incident_unknown_event_raises():
    with pytest.raises(ValueError):
        a.analyze_incident("EVT_DOES_NOT_EXIST")


def test_analyze_incident_returns_three_windows():
    events = a.load_events() if hasattr(a, "load_events") else None
    from mcp3_analytics.data_access import load_events
    first_event_id = load_events().iloc[0]["event_id"]
    result = a.analyze_incident(first_event_id)
    assert set(result["window_metrics"].keys()) == {"pre_incident", "during_incident", "post_incident"}
    assert "revenue_leakage" in result["window_metrics"]["during_incident"]


def test_get_correlation_analysis_has_caveat_and_is_bounded():
    result = a.get_correlation_analysis()
    assert "caveat" in result and "causa" in result["caveat"].lower()
    for r in result["pearson_correlation_with_revenue_leakage"].values():
        assert -1 <= r <= 1
    for r in result["spearman_correlation_with_revenue_leakage"].values():
        assert -1 <= r <= 1


@pytest.mark.parametrize("fn,dim", [
    (a.analyze_by_region, "region"),
    (a.analyze_by_service, "service_type"),
    (a.analyze_by_network, "network_type"),
    (a.analyze_by_plan, "plan_type"),
])
def test_segmentation_functions_sum_leakage_close_to_total(fn, dim):
    result = fn()
    assert result["dimension"] == dim
    segment_leakage_sum = sum(s["revenue_leakage"] for s in result["segments"])
    total_leakage = a.get_kpi_metrics()["total_revenue_leakage"]
    # segments should account for (approximately) all leakage — small tolerance for rounding
    assert abs(segment_leakage_sum - total_leakage) / total_leakage < 0.02
