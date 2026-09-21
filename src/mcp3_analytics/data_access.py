"""Data access layer for the Analytics MCP.

Loads the three RA datasets once at process startup and exposes them as
module-level, immutable pandas DataFrames. All analytics.py functions read
from here rather than touching disk per call.
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
def load_daily() -> pd.DataFrame:
    """One row per day: cdr_volume, failures, revenue_leakage, etc."""
    path = PROCESSED_DIR / "daily_ra_metrics.parquet"
    if not path.exists():
        path = RAW_DIR / "daily_ra_metrics.csv"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded daily_ra_metrics: %d rows (%s -> %s)", len(df), df["date"].min().date(), df["date"].max().date())
    return df


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    """Operational incident log."""
    path = RAW_DIR / "operational_events.csv"
    df = pd.read_csv(path)
    for col in ("start_date", "end_date", "intervention_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    logger.info("Loaded operational_events: %d rows", len(df))
    return df


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    """Transaction-level CDR data (cleaned). Used for network/plan/region/service breakdowns
    that need finer grain than the daily table provides."""
    path = PROCESSED_DIR / "cdr_transactions_clean.parquet"
    if not path.exists():
        path = RAW_DIR / "cdr_transactions.csv"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded cdr_transactions: %d rows", len(df))
    return df


def date_bounds() -> tuple[str, str]:
    df = load_daily()
    return str(df["date"].min().date()), str(df["date"].max().date())
