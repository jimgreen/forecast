"""Public weather data providers.

Open-Meteo is used as the default keyless provider. The adapter deliberately
returns a small canonical schema so it can be replaced by a local NWP archive or
another provider without changing the forecasting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd

from .schema import to_15min


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - URL is constructed from numeric inputs
        return json.loads(response.read().decode("utf-8"))


def _utc_timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@dataclass
class OpenMeteoProvider:
    latitude: float
    longitude: float
    timezone: str = "UTC"

    def _query(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        return _get_json(f"{endpoint}?{query}")

    @staticmethod
    def _frame(payload: dict[str, Any], section: str) -> pd.DataFrame:
        data = payload[section]
        timestamp = pd.to_datetime(data.pop("time"), utc=True)
        frame = pd.DataFrame(data, index=timestamp)
        return frame.rename(
            columns={
                "wind_speed_10m": "wind_speed_mps",
                "temperature_2m": "temperature_c",
                "shortwave_radiation": "solar_radiation_wm2",
            }
        )

    def forecast(self, start: str | date | None = None, end: str | date | None = None) -> pd.DataFrame:
        """Fetch 15-minute forecasts, falling back to hourly fields if needed."""
        params: dict[str, Any] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "forecast_days": 16,
            "minutely_15": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        }
        payload = self._query("https://api.open-meteo.com/v1/forecast", params)
        hourly = to_15min(self._frame(payload, "hourly"), timestamp_col="__unused__", timezone="UTC")
        if "minutely_15" in payload:
            fine = to_15min(self._frame(payload, "minutely_15"), timestamp_col="__unused__", timezone="UTC")
            frame = fine.combine_first(hourly)
        else:
            frame = hourly
        if start is not None:
            frame = frame.loc[_utc_timestamp(start) :]
        if end is not None:
            frame = frame.loc[: _utc_timestamp(end)]
        return frame

    def historical(self, start: str | date, end: str | date) -> pd.DataFrame:
        """Fetch historical hourly weather and normalize it to 15-minute values."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": str(start),
            "end_date": str(end),
            "timezone": "UTC",
            "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        }
        payload = self._query("https://archive-api.open-meteo.com/v1/archive", params)
        return to_15min(self._frame(payload, "hourly"), timestamp_col="__unused__", timezone="UTC")
