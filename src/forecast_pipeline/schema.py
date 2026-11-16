"""Time and column conventions used by the local forecasting pipeline."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

WEATHER_COLUMNS = ("wind_speed_mps", "temperature_c", "solar_radiation_wm2")
TARGET_COLUMNS = ("pv_power_kw", "wind_power_kw", "load_kw")

ALIASES = {
    "wind_speed": "wind_speed_mps",
    "wind_speed_10m": "wind_speed_mps",
    "temperature": "temperature_c",
    "temperature_2m": "temperature_c",
    "temp_c": "temperature_c",
    "solar_radiation": "solar_radiation_wm2",
    "shortwave_radiation": "solar_radiation_wm2",
    "ghi": "solar_radiation_wm2",
    "pv_power": "pv_power_kw",
    "solar_power": "pv_power_kw",
    "wind_power": "wind_power_kw",
    "load": "load_kw",
    "demand_kw": "load_kw",
}


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common source names to the pipeline's canonical names."""
    rename = {c: ALIASES[c] for c in frame.columns if c in ALIASES}
    return frame.rename(columns=rename)


def ensure_utc_index(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timezone: str = "Pacific/Auckland",
) -> pd.DataFrame:
    """Return a copy indexed by UTC timestamps, rejecting ambiguous timestamps."""
    if timestamp_col in frame.columns:
        frame = frame.set_index(timestamp_col)
    result = frame.copy()
    index = pd.to_datetime(result.index, errors="raise")
    if index.tz is None:
        index = index.tz_localize(timezone, ambiguous="raise", nonexistent="shift_forward")
    result.index = index.tz_convert("UTC")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def to_15min(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timezone: str = "Pacific/Auckland",
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Normalize source data to a UTC 15-minute grid.

    Numeric signals are averaged within an interval and interpolated only across
    short gaps. This keeps the operation useful for both station samples and
    NWP grids while avoiding unbounded forward filling.
    """
    result = ensure_utc_index(canonicalize_columns(frame), timestamp_col, timezone)
    if columns is None:
        columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    selected = [c for c in columns if c in result.columns]
    result = result[selected].apply(pd.to_numeric, errors="coerce")
    result = result.resample("15min").mean()
    if selected:
        result[selected] = result[selected].interpolate(method="time", limit=8, limit_area="inside")
    return result


def make_future_index(last_timestamp: pd.Timestamp, horizon: str | int) -> pd.DatetimeIndex:
    """Build a 15-minute UTC horizon from a label or number of steps."""
    steps = {
        "ultrashort": 16,  # four hours
        "day_ahead": 96,
        "week_ahead": 7 * 96,
    }.get(str(horizon), None)
    if steps is None:
        steps = int(horizon)
    return pd.date_range(
        start=pd.Timestamp(last_timestamp).tz_convert("UTC") + pd.Timedelta(minutes=15),
        periods=steps,
        freq="15min",
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")
