"""Run an offline smoke test for ultrashort, day-ahead and week-ahead forecasts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from forecast_pipeline import IntegratedForecastPipeline  # noqa: E402


def make_demo_data() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=90 * 96, freq="15min", tz="Pacific/Auckland")
    hour = np.asarray(index.hour + index.minute / 60, dtype=float)
    day = np.asarray(index.dayofyear, dtype=float)
    rng = np.random.default_rng(42)
    solar = np.maximum(0, 850 * np.sin(np.pi * (hour - 6) / 12)) * (0.8 + 0.15 * np.sin(day / 9))
    wind = 6 + 1.4 * np.sin(day / 5) + rng.normal(0, 0.6, len(index))
    temp = 18 + 7 * np.sin(2 * np.pi * (hour - 8) / 24) + 2 * np.sin(day / 18)
    load = 65 + 12 * np.sin(2 * np.pi * (hour - 7) / 24) + 5 * (hour < 6) + rng.normal(0, 1.5, len(index))
    return pd.DataFrame({
        "timestamp": index,
        "wind_speed_mps": wind.clip(0),
        "temperature_c": temp,
        "solar_radiation_wm2": solar,
        "pv_power_kw": (solar / 1000 * 80 + rng.normal(0, 1, len(index))).clip(0),
        "wind_power_kw": (wind.clip(0) ** 2 * 0.35 + rng.normal(0, 0.3, len(index))).clip(0),
        "load_kw": load.clip(0),
    })


if __name__ == "__main__":
    observations = make_demo_data()
    pipeline = IntegratedForecastPipeline().fit(observations)
    last_weather = observations.tail(1).set_index("timestamp")[list(("wind_speed_mps", "temperature_c", "solar_radiation_wm2"))]
    future_index = pd.date_range(observations.timestamp.iloc[-1] + pd.Timedelta(minutes=15), periods=7 * 96, freq="15min")
    future_weather = pd.DataFrame(index=future_index)
    for column in last_weather.columns:
        future_weather[column] = float(last_weather.iloc[0][column])
    result = pipeline.predict("week_ahead", future_weather=future_weather)
    output = Path(__file__).parents[1] / "demo_week_ahead_forecast.csv"
    result.to_csv(output)
    print(f"wrote {len(result)} 15-minute rows to {output}")
