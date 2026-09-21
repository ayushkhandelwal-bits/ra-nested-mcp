"""Forecasting for daily revenue_leakage.

Baseline: seasonal-naive (value from 7 days prior).
Model: Holt-Winters / triple exponential smoothing (additive trend + weekly
seasonality, damped trend to avoid runaway extrapolation) — chosen over
SARIMA for interpretability, per the "prefer interpretability over
unnecessary complexity" principle. Evaluated with a chronological
train/test split only (never random split, which would leak future
information into training for a time series).
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from mcp4_forecasting.data_access import load_daily_leakage_series

logger = logging.getLogger(__name__)

SEASONAL_PERIODS = 7  # weekly seasonality


# ----------------------------------------------------------------------
# baseline + model fitting
# ----------------------------------------------------------------------

def _seasonal_naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    """Forecast = value from the same weekday `SEASONAL_PERIODS` days ago, tiled forward."""
    last_cycle = train.values[-SEASONAL_PERIODS:]
    reps = int(np.ceil(steps / SEASONAL_PERIODS))
    return np.tile(last_cycle, reps)[:steps]


def _fit_holt_winters(train: pd.Series):
    model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def _accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape_pct": round(mape, 2) if mape is not None else None}


# ----------------------------------------------------------------------
# public tools
# ----------------------------------------------------------------------

def compare_forecast_models(test_days: int = 30) -> dict:
    """Chronological holdout evaluation: train on everything except the last
    `test_days`, forecast that many days ahead, and score seasonal-naive vs.
    Holt-Winters against the actual held-out values. Never uses a random split."""
    series = load_daily_leakage_series()
    if len(series) <= test_days + SEASONAL_PERIODS * 3:
        raise ValueError(f"Not enough history ({len(series)} days) for a {test_days}-day chronological holdout.")

    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    naive_pred = _seasonal_naive_forecast(train, test_days)
    hw_fit = _fit_holt_winters(train)
    hw_pred = hw_fit.forecast(test_days).values
    hw_pred = np.clip(hw_pred, 0, None)  # revenue_leakage cannot be negative

    return {
        "evaluation_method": "chronological_holdout (never random split — this is a time series)",
        "train_period": {"start": str(train.index.min().date()), "end": str(train.index.max().date()), "n": len(train)},
        "test_period": {"start": str(test.index.min().date()), "end": str(test.index.max().date()), "n": len(test)},
        "models": {
            "seasonal_naive": {"description": "Repeats the value from 7 days prior.", **_accuracy(test.values, naive_pred)},
            "holt_winters": {"description": "Additive trend (damped) + weekly seasonality.", **_accuracy(test.values, hw_pred)},
        },
        "better_model": "holt_winters" if _accuracy(test.values, hw_pred)["rmse"] < _accuracy(test.values, naive_pred)["rmse"] else "seasonal_naive",
    }


def get_forecast_accuracy(model: str = "holt_winters", test_days: int = 30) -> dict:
    """Detailed day-by-day accuracy for a single model over the chronological holdout."""
    if model not in ("holt_winters", "seasonal_naive"):
        raise ValueError("model must be 'holt_winters' or 'seasonal_naive'")
    series = load_daily_leakage_series()
    if len(series) <= test_days + SEASONAL_PERIODS * 3:
        raise ValueError(f"Not enough history ({len(series)} days) for a {test_days}-day chronological holdout.")
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    if model == "seasonal_naive":
        pred = _seasonal_naive_forecast(train, test_days)
    else:
        pred = np.clip(_fit_holt_winters(train).forecast(test_days).values, 0, None)

    daily = [
        {"date": str(d.date()), "actual": round(float(a), 2), "predicted": round(float(p), 2), "abs_error": round(float(abs(a - p)), 2)}
        for d, a, p in zip(test.index, test.values, pred)
    ]
    return {"model": model, **_accuracy(test.values, pred), "daily_errors": daily}


def forecast_revenue_leakage(steps: int = 30, intervals: bool = True) -> dict:
    """Forecast revenue_leakage `steps` days into the future using Holt-Winters
    fit on the full available history. If `intervals` is True, approximate
    prediction intervals are derived from Monte Carlo simulation of the fitted
    model (not a closed-form formula — Holt-Winters doesn't have one), which is
    the standard approach for this model class."""
    series = load_daily_leakage_series()
    fit = _fit_holt_winters(series)
    point_forecast = np.clip(fit.forecast(steps).values, 0, None)

    result_intervals = None
    if intervals:
        n_sims = 500
        try:
            sims = fit.simulate(nsimulations=steps, repetitions=n_sims, error="add")
            sims = np.clip(sims.values, 0, None) if hasattr(sims, "values") else np.clip(sims, 0, None)
            lower_80 = np.percentile(sims, 10, axis=1)
            upper_80 = np.percentile(sims, 90, axis=1)
            lower_95 = np.percentile(sims, 2.5, axis=1)
            upper_95 = np.percentile(sims, 97.5, axis=1)
        except Exception as e:
            logger.warning("Simulation-based intervals failed (%s); falling back to residual-based approximation.", e)
            resid_std = float(np.std(fit.resid))
            horizon = np.arange(1, steps + 1)
            spread = resid_std * np.sqrt(horizon)
            lower_80, upper_80 = np.clip(point_forecast - 1.28 * spread, 0, None), point_forecast + 1.28 * spread
            lower_95, upper_95 = np.clip(point_forecast - 1.96 * spread, 0, None), point_forecast + 1.96 * spread
        result_intervals = (lower_80, upper_80, lower_95, upper_95)

    last_date = series.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")

    forecast_rows = []
    for i, (d, v) in enumerate(zip(future_dates, point_forecast)):
        row = {"date": str(d.date()), "forecast": round(float(v), 2)}
        if result_intervals:
            row.update({
                "lower_80": round(float(result_intervals[0][i]), 2), "upper_80": round(float(result_intervals[1][i]), 2),
                "lower_95": round(float(result_intervals[2][i]), 2), "upper_95": round(float(result_intervals[3][i]), 2),
            })
        forecast_rows.append(row)

    recent = series.iloc[-14:]
    holdout_check = compare_forecast_models(test_days=min(30, len(series) // 4))

    return {
        "model": "Holt-Winters (additive, damped trend, 7-day seasonality)",
        "forecast_horizon_days": steps,
        "last_observed_date": str(last_date.date()),
        "last_observed_value": round(float(series.iloc[-1]), 2),
        "recent_14d_mean": round(float(recent.mean()), 2),
        "forecast": forecast_rows,
        "expected_direction": (
            "increasing" if point_forecast[-1] > recent.mean() * 1.1 else
            "decreasing" if point_forecast[-1] < recent.mean() * 0.9 else "stable"
        ),
        "model_holdout_accuracy": {
            "rmse": holdout_check["models"]["holt_winters"]["rmse"],
            "mae": holdout_check["models"]["holt_winters"]["mae"],
            "note": f"Measured on the most recent {holdout_check['test_period']['n']}-day chronological holdout, not on this forecast period itself.",
        },
        "limitations": (
            "This is a statistical projection from historical patterns (trend + weekly seasonality), "
            "not a guarantee. It does not know about future operational incidents, planned changes, or "
            "external events. Wider intervals reflect greater uncertainty further into the horizon. "
            "Treat as a range to plan around, not a committed number."
        ),
    }


def get_forecast_intervals(steps: int = 30) -> dict:
    """Convenience wrapper: forecast with intervals only (no extra narrative fields)."""
    full = forecast_revenue_leakage(steps=steps, intervals=True)
    return {"model": full["model"], "forecast_horizon_days": steps, "forecast": full["forecast"]}
