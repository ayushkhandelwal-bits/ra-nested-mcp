"""Minimal data loader for the Forecasting MCP.

Deliberately independent of mcp3_analytics.data_access — each MCP server in
this architecture is meant to be independently deployable, so mcp4 only
loads what it needs (the daily time series) rather than importing mcp3.
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@lru_cache(maxsize=1)
def load_daily_leakage_series() -> pd.Series:
    """Return revenue_leakage as a daily-frequency pandas Series indexed by date."""
    path = PROCESSED_DIR / "daily_ra_metrics.parquet"
    if not path.exists():
        path = RAW_DIR / "daily_ra_metrics.csv"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    s = df["revenue_leakage"].asfreq("D")
    s = s.interpolate(limit_direction="both")  # guard against any calendar gaps
    logger.info("Loaded revenue_leakage series: %d days (%s -> %s)", len(s), s.index.min().date(), s.index.max().date())
    return s
