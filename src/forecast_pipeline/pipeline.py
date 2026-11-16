"""End-to-end local integration of weather and energy forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .fusion import fuse_weather
from .model import Ridge15MinForecaster
from .providers import OpenMeteoProvider
from .schema import TARGET_COLUMNS, WEATHER_COLUMNS, canonicalize_columns, ensure_utc_index, make_future_index, require_columns, to_15min


@dataclass
class IntegratedForecastPipeline:
    provider: OpenMeteoProvider | None = None
    timezone: str = "Pacific/Auckland"
    model: Ridge15MinForecaster | None = None
    history: pd.DataFrame | None = None

    def fit(self, observations: pd.DataFrame, targets: tuple[str, ...] = TARGET_COLUMNS) -> "IntegratedForecastPipeline":
        data = to_15min(canonicalize_columns(observations), timestamp_col="timestamp", timezone=self.timezone)
        require_columns(data, [*WEATHER_COLUMNS, *targets], "observations")
        self.history = data
        self.model = Ridge15MinForecaster(targets=targets).fit(data)
        return self

    def fit_csv(self, path: str | Path, targets: tuple[str, ...] = TARGET_COLUMNS) -> "IntegratedForecastPipeline":
        return self.fit(pd.read_csv(path), targets=targets)

    def weather_for(self, horizon: str | int, future_weather: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("fit the pipeline before requesting a forecast")
        index = make_future_index(self.history.index[-1], horizon)
        if future_weather is None:
            if self.provider is None:
                raise ValueError("future_weather or a weather provider is required")
            future_weather = self.provider.forecast(start=index[0], end=index[-1])
        local_weather = self.history[list(WEATHER_COLUMNS)]
        weather = fuse_weather(future_weather, local=local_weather, timezone=self.timezone)
        weather = weather[list(WEATHER_COLUMNS)].reindex(index).interpolate(method="time").ffill().bfill()
        return weather

    def predict(self, horizon: str | int, future_weather: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.model is None or self.history is None:
            raise RuntimeError("fit the pipeline before predicting")
        weather = self.weather_for(horizon, future_weather=future_weather)
        predictions = self.model.predict(self.history, weather)
        return predictions.join(weather[list(WEATHER_COLUMNS)])
