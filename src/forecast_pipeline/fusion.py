"""Fuse public NWP values with local station observations."""

from __future__ import annotations

import pandas as pd

from .schema import WEATHER_COLUMNS, to_15min


def fuse_weather(
    nwp: pd.DataFrame,
    local: pd.DataFrame | None = None,
    timezone: str = "Pacific/Auckland",
    bias_window: int = 96,
) -> pd.DataFrame:
    """Apply a recent local-observation bias correction to NWP weather.

    Local values are used directly where available. For future timestamps the
    NWP value is corrected by the rolling mean of recent local-minus-NWP errors.
    """
    nwp_15 = to_15min(nwp, timestamp_col="__unused__", timezone=timezone, columns=WEATHER_COLUMNS)
    if local is None or local.empty:
        out = nwp_15.copy()
        out["weather_source"] = "nwp"
        return out
    local_15 = to_15min(local, timestamp_col="timestamp", timezone=timezone, columns=WEATHER_COLUMNS)
    all_index = nwp_15.index.union(local_15.index).sort_values()
    nwp_15 = nwp_15.reindex(all_index)
    local_15 = local_15.reindex(all_index)
    out = nwp_15.copy()
    for column in WEATHER_COLUMNS:
        if column not in out:
            continue
        overlap = (local_15[column] - nwp_15[column]).dropna()
        bias = overlap.rolling(bias_window, min_periods=4).mean().reindex(all_index).ffill().fillna(0.0)
        out[column] = nwp_15[column] + bias
        out.loc[local_15[column].notna(), column] = local_15.loc[local_15[column].notna(), column]
    out["weather_source"] = "nwp_corrected"
    out.loc[local_15[list(WEATHER_COLUMNS)].notna().any(axis=1), "weather_source"] = "local_or_corrected"
    return out
