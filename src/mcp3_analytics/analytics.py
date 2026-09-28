"""Revenue Assurance analytics functions.

Every function here is pure (no side effects), takes a date range, and
returns a plain dict/list of dicts (JSON-serializable) so it can be
returned directly as an MCP tool result. All KPI formulas are documented
inline — nothing here is an arbitrary or hard-coded metric.
"""

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import stats

from mcp3_analytics.data_access import load_daily, load_events, load_transactions

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _parse_date(d: str | date | None) -> pd.Timestamp | None:
    if d is None or (isinstance(d, str) and not d.strip()):
        return None
    return pd.Timestamp(d)


def _filter_daily(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    df = load_daily()
    start, end = _parse_date(start_date), _parse_date(end_date)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    if df.empty:
        raise ValueError(f"No data in range {start_date}..{end_date}. Available range is {load_daily()['date'].min().date()}..{load_daily()['date'].max().date()}.")
    return df.reset_index(drop=True)


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


# ----------------------------------------------------------------------
# KPI DEFINITIONS
# ----------------------------------------------------------------------
# leakage_rate            = revenue_leakage / expected_revenue
# recovery_rate           = revenue_recovered / revenue_leakage
# cdr_failure_rate        = failed_cdrs / cdr_volume
# mediation_failure_rate  = mediation_failures / cdr_volume
# rating_failure_rate     = rating_failures / cdr_volume
# billing_mismatch_rate   = billing_mismatches / cdr_volume
# duplicate_cdr_rate      = duplicate_cdrs / cdr_volume
# missing_cdr_rate        = missing_cdrs / cdr_volume
# avg_processing_delay    = mean(processing_delay_avg) over the period
# ra_alert_rate           = ra_alerts / cdr_volume  (alerts per CDR processed)

def get_kpi_metrics(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Compute the standard RA KPI set over a date range (defaults to full history)."""
    df = _filter_daily(start_date, end_date)
    expected = df["expected_revenue"].sum()
    billed = df["billed_revenue"].sum()
    leakage = df["revenue_leakage"].sum()
    recovered = df["revenue_recovered"].sum()
    volume = df["cdr_volume"].sum()

    return {
        "period": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date()), "days": len(df)},
        "total_expected_revenue": round(float(expected), 2),
        "total_billed_revenue": round(float(billed), 2),
        "total_revenue_leakage": round(float(leakage), 2),
        "leakage_rate": round(_safe_div(leakage, expected), 5),
        "total_revenue_recovered": round(float(recovered), 2),
        "recovery_rate": round(_safe_div(recovered, leakage), 5),
        "total_cdr_volume": int(volume),
        "cdr_failure_rate": round(_safe_div(df["failed_cdrs"].sum(), volume), 5),
        "mediation_failure_rate": round(_safe_div(df["mediation_failures"].sum(), volume), 5),
        "rating_failure_rate": round(_safe_div(df["rating_failures"].sum(), volume), 5),
        "billing_mismatch_rate": round(_safe_div(df["billing_mismatches"].sum(), volume), 5),
        "duplicate_cdr_rate": round(_safe_div(df["duplicate_cdrs"].sum(), volume), 5),
        "missing_cdr_rate": round(_safe_div(df["missing_cdrs"].sum(), volume), 5),
        "avg_processing_delay_sec": round(float(df["processing_delay_avg"].mean()), 3),
        "ra_alert_rate_per_1000_cdrs": round(_safe_div(df["ra_alerts"].sum(), volume) * 1000, 4),
        "total_customer_complaints": int(df["customer_complaints"].sum()),
        "definitions": {
            "leakage_rate": "revenue_leakage / expected_revenue",
            "recovery_rate": "revenue_recovered / revenue_leakage",
            "cdr_failure_rate": "failed_cdrs / cdr_volume",
            "mediation_failure_rate": "mediation_failures / cdr_volume",
            "rating_failure_rate": "rating_failures / cdr_volume",
            "billing_mismatch_rate": "billing_mismatches / cdr_volume",
            "duplicate_cdr_rate": "duplicate_cdrs / cdr_volume",
            "missing_cdr_rate": "missing_cdrs / cdr_volume",
        },
    }


def get_ra_summary(start_date: str | None = None, end_date: str | None = None) -> dict:
    """High-level descriptive summary: KPIs + descriptive stats on the core metrics.
    This is the "what happened" entry point."""
    df = _filter_daily(start_date, end_date)
    kpis = get_kpi_metrics(start_date, end_date)

    desc_cols = ["revenue_leakage", "processing_delay_avg", "mediation_failures", "rating_failures", "billing_mismatches", "recovery_rate"]
    descriptive = {}
    for col in desc_cols:
        s = df[col]
        descriptive[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "p95": round(float(s.quantile(0.95)), 4),
        }

    # simple trend signal: compare first-half vs second-half mean leakage
    mid = len(df) // 2
    first_half_mean = df["revenue_leakage"].iloc[:mid].mean() if mid > 0 else float("nan")
    second_half_mean = df["revenue_leakage"].iloc[mid:].mean()
    trend_pct = _safe_div(second_half_mean - first_half_mean, first_half_mean) * 100 if mid > 0 else 0.0
    trend_direction = "increasing" if trend_pct > 5 else ("decreasing" if trend_pct < -5 else "stable")

    return {
        "kpis": kpis,
        "descriptive_statistics": descriptive,
        "trend": {
            "direction": trend_direction,
            "first_half_mean_leakage": round(float(first_half_mean), 2) if mid > 0 else None,
            "second_half_mean_leakage": round(float(second_half_mean), 2),
            "pct_change": round(float(trend_pct), 2) if mid > 0 else None,
        },
    }


def get_revenue_leakage_trend(start_date: str | None = None, end_date: str | None = None, rolling_window: int = 7) -> dict:
    """Daily revenue_leakage series plus a rolling mean, for charting/trend analysis."""
    df = _filter_daily(start_date, end_date)
    roll = df["revenue_leakage"].rolling(rolling_window, min_periods=1).mean()
    series = [
        {"date": str(d.date()), "revenue_leakage": round(float(v), 2), f"rolling_{rolling_window}d_mean": round(float(r), 2)}
        for d, v, r in zip(df["date"], df["revenue_leakage"], roll)
    ]
    return {"rolling_window_days": rolling_window, "n_days": len(series), "series": series}


# ----------------------------------------------------------------------
# ANOMALY DETECTION
# ----------------------------------------------------------------------

_SUPPORTED_METRICS = [
    "revenue_leakage", "failed_cdrs", "mediation_failures", "rating_failures",
    "billing_mismatches", "processing_delay_avg", "duplicate_cdrs",
]


def detect_anomalies(
    metric: str = "revenue_leakage",
    method: str = "zscore",
    window: int = 14,
    threshold: float = 3.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Detect anomalous days on a given metric.

    method='zscore': rolling mean/std over `window` days; flags |z| > threshold.
    method='iqr': flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] computed over the whole range.

    Returns one row per detected anomaly with date, observed vs baseline value,
    deviation, a normalized anomaly_score, and a severity bucket. Not every
    unusual point is flagged — only points crossing the stated threshold.
    """
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {_SUPPORTED_METRICS}")
    df = _filter_daily(start_date, end_date)
    s = df[metric]

    anomalies = []
    if method == "zscore":
        roll_mean = s.rolling(window, min_periods=max(5, window // 2)).mean()
        roll_std = s.rolling(window, min_periods=max(5, window // 2)).std()
        z = (s - roll_mean) / roll_std
        for i in range(len(df)):
            if pd.isna(z.iloc[i]):
                continue
            if abs(z.iloc[i]) >= threshold:
                sev = "critical" if abs(z.iloc[i]) >= 5 else ("high" if abs(z.iloc[i]) >= 4 else "medium")
                anomalies.append({
                    "date": str(df["date"].iloc[i].date()),
                    "metric": metric,
                    "observed_value": round(float(s.iloc[i]), 3),
                    "baseline_value": round(float(roll_mean.iloc[i]), 3),
                    "deviation": round(float(s.iloc[i] - roll_mean.iloc[i]), 3),
                    "anomaly_score": round(float(z.iloc[i]), 2),
                    "severity": sev,
                })
    elif method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        baseline = s.median()
        for i in range(len(df)):
            v = s.iloc[i]
            if v < lower or v > upper:
                bound_dist = (v - upper) if v > upper else (lower - v)
                score = round(float(bound_dist / iqr), 2) if iqr else 0.0
                sev = "critical" if score >= 3 else ("high" if score >= 1.5 else "medium")
                anomalies.append({
                    "date": str(df["date"].iloc[i].date()),
                    "metric": metric,
                    "observed_value": round(float(v), 3),
                    "baseline_value": round(float(baseline), 3),
                    "deviation": round(float(v - baseline), 3),
                    "anomaly_score": score,
                    "severity": sev,
                })
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    return {
        "metric": metric,
        "method": method,
        "params": {"window": window, "threshold": threshold} if method == "zscore" else {"iqr_multiplier": 1.5},
        "n_anomalies": len(anomalies),
        "anomalies": anomalies,
    }


# ----------------------------------------------------------------------
# HYPOTHESIS TESTING
# ----------------------------------------------------------------------

def _choose_and_run_test(group_a: pd.Series, group_b: pd.Series) -> dict:
    """Pick an appropriate two-sample test based on sample size and normality.

    - If either group has n < 20, or a Shapiro-Wilk test rejects normality
      (p < 0.05) for either group, use the non-parametric Mann-Whitney U test.
    - Otherwise use Welch's t-test (does not assume equal variance — safer
      default than Student's t-test for operational data with different
      volatility before/after an incident).
    """
    a, b = group_a.dropna().values, group_b.dropna().values
    normal_enough = True
    if len(a) >= 3 and len(b) >= 3:
        try:
            _, p_a = stats.shapiro(a) if len(a) <= 5000 else (None, 1.0)
            _, p_b = stats.shapiro(b) if len(b) <= 5000 else (None, 1.0)
            normal_enough = (p_a > 0.05) and (p_b > 0.05)
        except Exception:
            normal_enough = False
    else:
        normal_enough = False

    if len(a) >= 20 and len(b) >= 20 and normal_enough:
        stat, p = stats.ttest_ind(b, a, equal_var=False)
        test_name = "Welch's t-test (unequal variance)"
    else:
        stat, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        test_name = "Mann-Whitney U (non-parametric)"

    return {
        "test_used": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "group_a_n": int(len(a)), "group_a_mean": round(float(np.mean(a)), 4) if len(a) else None,
        "group_b_n": int(len(b)), "group_b_mean": round(float(np.mean(b)), 4) if len(b) else None,
    }


def run_hypothesis_test(
    metric: str = "revenue_leakage",
    split_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """Test whether `metric` differs significantly before vs. on/after `split_date`.

    H0: mean(metric) before split_date == mean(metric) on/after split_date
    H1: they differ

    Automatically selects Welch's t-test or Mann-Whitney U based on sample
    size and a Shapiro-Wilk normality check (see _choose_and_run_test).
    Also reports a business-significance check: whether the % change in
    means exceeds a 10% materiality threshold, since statistical
    significance and business significance are not the same thing.
    """
    if split_date is None:
        raise ValueError("split_date is required, e.g. an incident start_date")
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {_SUPPORTED_METRICS}")

    df = _filter_daily(start_date, end_date)
    split = _parse_date(split_date)
    before = df.loc[df["date"] < split, metric]
    after = df.loc[df["date"] >= split, metric]
    if len(before) < 3 or len(after) < 3:
        raise ValueError(f"Not enough data on one side of split_date={split_date} (before={len(before)}, after={len(after)}); need >=3 each.")

    result = _choose_and_run_test(before, after)
    significant = result["p_value"] < alpha
    pct_change = _safe_div(result["group_b_mean"] - result["group_a_mean"], result["group_a_mean"]) * 100
    business_significant = abs(pct_change) >= 10.0

    conclusion = (
        f"Reject H0 (p={result['p_value']:.4f} < {alpha}): {metric} differs significantly before vs. after {split_date}."
        if significant else
        f"Fail to reject H0 (p={result['p_value']:.4f} >= {alpha}): no statistically significant difference in {metric}."
    )

    return {
        "metric": metric, "split_date": split_date, "alpha": alpha,
        "null_hypothesis": f"Mean {metric} before {split_date} equals mean {metric} on/after {split_date}.",
        "alternative_hypothesis": f"Mean {metric} before and on/after {split_date} differ.",
        **result,
        "pct_change": round(pct_change, 2),
        "statistically_significant": bool(significant),
        "business_significant_gt10pct": bool(business_significant),
        "conclusion": conclusion,
        "business_interpretation": (
            f"{'Statistically significant AND' if significant else 'Not statistically significant, though'} "
            f"the mean moved {pct_change:+.1f}% "
            f"({'a change worth investigating operationally' if business_significant else 'a change too small to be operationally material on its own'})."
        ),
        "caveat": "Statistical significance indicates the difference is unlikely to be due to random chance; it does not by itself indicate the size of the difference matters to the business, or that the split_date event caused it.",
    }


# ----------------------------------------------------------------------
# INCIDENT ANALYSIS
# ----------------------------------------------------------------------

def analyze_incident(event_id: str, pre_days: int = 14, post_days: int = 14) -> dict:
    """Pre/during/post analysis of a single operational_events.csv incident against
    daily_ra_metrics.csv. Answers: how did leakage/failures change, and did they
    return to baseline after resolution?"""
    events = load_events()
    row = events.loc[events["event_id"] == event_id]
    if row.empty:
        raise ValueError(f"Unknown event_id '{event_id}'. Available: {sorted(events['event_id'].tolist())}")
    ev = row.iloc[0]
    daily = load_daily()

    pre_start = ev["start_date"] - pd.Timedelta(days=pre_days)
    pre_window = daily[(daily["date"] >= pre_start) & (daily["date"] < ev["start_date"])]
    during_window = daily[(daily["date"] >= ev["start_date"]) & (daily["date"] <= ev["end_date"])]
    post_end = ev["end_date"] + pd.Timedelta(days=post_days)
    post_window = daily[(daily["date"] > ev["end_date"]) & (daily["date"] <= post_end)]

    def _stats(w: pd.DataFrame, col: str) -> float | None:
        return round(float(w[col].mean()), 3) if len(w) else None

    metrics = ["revenue_leakage", "mediation_failures", "rating_failures", "billing_mismatches", "processing_delay_avg", "recovery_rate"]
    windows = {"pre_incident": {m: _stats(pre_window, m) for m in metrics},
               "during_incident": {m: _stats(during_window, m) for m in metrics},
               "post_incident": {m: _stats(post_window, m) for m in metrics}}

    # statistical test on the primary metric (revenue_leakage): pre vs during
    test_result = None
    if len(pre_window) >= 3 and len(during_window) >= 3:
        test_result = _choose_and_run_test(pre_window["revenue_leakage"], during_window["revenue_leakage"])
        test_result["p_significant"] = bool(test_result["p_value"] < 0.05)

    # recovery check: is post-incident leakage back within ~15% of pre-incident baseline?
    recovered_to_baseline = None
    if windows["pre_incident"]["revenue_leakage"] and windows["post_incident"]["revenue_leakage"] is not None:
        pre_v, post_v = windows["pre_incident"]["revenue_leakage"], windows["post_incident"]["revenue_leakage"]
        recovered_to_baseline = abs(post_v - pre_v) / pre_v <= 0.15 if pre_v else None

    extra_leakage = None
    if windows["pre_incident"]["revenue_leakage"] is not None and len(during_window):
        baseline_daily = windows["pre_incident"]["revenue_leakage"]
        extra_leakage = round(float((during_window["revenue_leakage"] - baseline_daily).clip(lower=0).sum()), 2)

    return {
        "event": {
            "event_id": ev["event_id"], "event_type": ev["event_type"], "severity": ev["severity"],
            "affected_system": ev["affected_system"], "affected_region": ev["affected_region"],
            "affected_service": ev["affected_service"], "start_date": str(ev["start_date"].date()),
            "end_date": str(ev["end_date"].date()), "description": ev["description"],
        },
        "window_metrics": windows,
        "hypothesis_test_leakage_pre_vs_during": test_result,
        "estimated_extra_revenue_leakage_during_incident": extra_leakage,
        "recovered_to_baseline_within_15pct": recovered_to_baseline,
    }


# ----------------------------------------------------------------------
# CORRELATION / DRIVER ANALYSIS
# ----------------------------------------------------------------------

def get_correlation_analysis(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Pearson and Spearman correlation of operational metrics against revenue_leakage.
    Correlation, not causation — see the `caveat` field."""
    df = _filter_daily(start_date, end_date)
    drivers = ["mediation_failures", "rating_failures", "billing_mismatches", "duplicate_cdrs", "processing_delay_avg", "cdr_volume", "failed_cdrs"]
    pearson = {c: round(float(df["revenue_leakage"].corr(df[c], method="pearson")), 3) for c in drivers}
    spearman = {c: round(float(df["revenue_leakage"].corr(df[c], method="spearman")), 3) for c in drivers}
    ranked = sorted(pearson.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "pearson_correlation_with_revenue_leakage": pearson,
        "spearman_correlation_with_revenue_leakage": spearman,
        "strongest_linear_driver": {"metric": ranked[0][0], "pearson_r": ranked[0][1]} if ranked else None,
        "caveat": "These are correlations, not causal effects. A high correlation means the two metrics move together; it does not establish that one causes the other. Use analyze_incident() for event-based evidence of an operational cause.",
    }


# ----------------------------------------------------------------------
# SEGMENTATION
# ----------------------------------------------------------------------

def _analyze_by_dimension(dimension: str) -> dict:
    tx = load_transactions()
    if dimension not in tx.columns:
        raise ValueError(f"Unknown dimension '{dimension}'; available columns: {list(tx.columns)}")
    g = tx.groupby(dimension).agg(
        cdr_volume=("cdr_id", "size"),
        expected_revenue=("expected_charge", "sum"),
        billed_revenue=("billed_amount", "sum"),
        revenue_leakage=("revenue_leakage", "sum"),
        duplicate_cdrs=("is_duplicate", "sum"),
        mediation_failures=("mediation_status", lambda s: (s == "Failed").sum()),
        billing_mismatches=("billing_status", lambda s: (s == "Failed").sum()),
    ).reset_index()
    g["leakage_rate_pct"] = (g["revenue_leakage"] / g["expected_revenue"] * 100).round(3)
    g["billing_mismatch_rate_pct"] = (g["billing_mismatches"] / g["cdr_volume"] * 100).round(3)
    g = g.sort_values("revenue_leakage", ascending=False)
    records = g.round(2).to_dict(orient="records")
    top = records[0] if records else None
    return {
        "dimension": dimension,
        "segments": records,
        "highest_leakage_segment": {"value": top[dimension], "revenue_leakage": top["revenue_leakage"]} if top else None,
        "highest_leakage_rate_segment": max(records, key=lambda r: r["leakage_rate_pct"])[dimension] if records else None,
    }


def analyze_by_region() -> dict:
    return _analyze_by_dimension("region")


def analyze_by_service() -> dict:
    return _analyze_by_dimension("service_type")


def analyze_by_network() -> dict:
    return _analyze_by_dimension("network_type")


def analyze_by_plan() -> dict:
    return _analyze_by_dimension("plan_type")
