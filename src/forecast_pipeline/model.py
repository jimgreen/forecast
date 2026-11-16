"""Small dependency-light multi-target 15-minute forecaster.

The class is intentionally simple and auditable. It provides a strong baseline
for integration and can later be replaced by OpenSTEF, Quartz or a deep model
without changing the input/output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import WEATHER_COLUMNS, TARGET_COLUMNS, require_columns


def _feature_row(timestamp: pd.Timestamp, history: pd.DataFrame, target: str) -> list[float]:
    row = history.loc[timestamp]
    values = [
        float(row.get("wind_speed_mps", 0.0)),
        float(row.get("temperature_c", 0.0)),
        float(row.get("solar_radiation_wm2", 0.0)),
        np.sin(2 * np.pi * timestamp.hour / 24),
        np.cos(2 * np.pi * timestamp.hour / 24),
        np.sin(2 * np.pi * timestamp.dayofweek / 7),
        np.cos(2 * np.pi * timestamp.dayofweek / 7),
    ]
    for lag in (1, 2, 4, 96, 672):
        lag_timestamp = timestamp - pd.Timedelta(minutes=15 * lag)
        values.append(float(history.at[lag_timestamp, target]) if lag_timestamp in history.index else 0.0)
    return values


@dataclass
class Ridge15MinForecaster:
    targets: tuple[str, ...] = TARGET_COLUMNS
    alpha: float = 1.0
    coefficients: dict[str, np.ndarray] = field(default_factory=dict)
    feature_count: int = 0

    def fit(self, frame: pd.DataFrame) -> "Ridge15MinForecaster":
        required = [*WEATHER_COLUMNS, *self.targets]
        require_columns(frame, required, "training data")
        data = frame.copy().sort_index()
        for target in self.targets:
            rows: list[list[float]] = []
            ys: list[float] = []
            for timestamp in data.index:
                if not np.isfinite(data.at[timestamp, target]):
                    continue
                rows.append(_feature_row(timestamp, data, target))
                ys.append(float(data.at[timestamp, target]))
            if len(rows) < 32:
                raise ValueError(f"not enough valid rows to train {target}: {len(rows)}")
            x = np.asarray(rows, dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            y = np.asarray(ys, dtype=float)
            penalty = self.alpha * np.eye(x.shape[1])
            penalty[0, 0] = 0.0
            self.coefficients[target] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            self.feature_count = x.shape[1]
        return self

    def predict(self, history: pd.DataFrame, future_weather: pd.DataFrame) -> pd.DataFrame:
        require_columns(history, [*WEATHER_COLUMNS, *self.targets], "history")
        require_columns(future_weather, WEATHER_COLUMNS, "future weather")
        state = history.copy().sort_index()
        future = future_weather.copy().sort_index()
        predictions = pd.DataFrame(index=future.index, columns=self.targets, dtype=float)
        for timestamp in future.index:
            state.loc[timestamp, WEATHER_COLUMNS] = future.loc[timestamp, WEATHER_COLUMNS]
            for target in self.targets:
                row = _feature_row(timestamp, state, target)
                vector = np.asarray([1.0, *row], dtype=float)
                value = float(vector @ self.coefficients[target])
                value = max(0.0, value) if target.endswith("_power_kw") or target == "load_kw" else value
                state.loc[timestamp, target] = value
                predictions.at[timestamp, target] = value
        return predictions
