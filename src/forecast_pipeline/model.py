"""Small dependency-light multi-target 15-minute forecaster.

The class is intentionally simple and auditable. It provides a strong baseline
for integration and can later be replaced by OpenSTEF, Quartz or a deep model
without changing the input/output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .schema import WEATHER_COLUMNS, TARGET_COLUMNS, require_columns


def _scalar(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _feature_row(
    timestamp: pd.Timestamp,
    history: pd.DataFrame,
    target: str,
    *,
    position: int | None = None,
    target_values: np.ndarray | None = None,
) -> list[float]:
    """Build one feature vector while allowing callers to reuse index arrays.

    Training and recursive prediction call this function many times. Reusing the
    already known row position and target array avoids a repeated ``searchsorted``
    and pandas-to-NumPy conversion for every sample.
    """
    if position is None:
        position = int(history.index.searchsorted(timestamp, side="left"))
    row = history.iloc[position] if position < len(history) and history.index[position] == timestamp else history.iloc[max(0, position - 1)]
    if target_values is None:
        target_values = history[target].to_numpy(dtype=float, na_value=np.nan)
    hour = timestamp.hour + timestamp.minute / 60.0
    quarter = timestamp.hour * 4 + timestamp.minute // 15
    values = [
        _scalar(row.get("wind_speed_mps")),
        _scalar(row.get("temperature_c")),
        max(0.0, _scalar(row.get("solar_radiation_wm2"))),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * quarter / 96),
        np.cos(2 * np.pi * quarter / 96),
        np.sin(2 * np.pi * timestamp.dayofweek / 7),
        np.cos(2 * np.pi * timestamp.dayofweek / 7),
    ]
    for lag in (1, 2, 4, 96, 672):
        lag_position = position - lag
        values.append(_scalar(target_values[lag_position]) if 0 <= lag_position < len(target_values) else 0.0)
    for window in (4, 16, 96):
        prior = target_values[max(0, position - window):position]
        prior = prior[np.isfinite(prior)]
        values.extend([_scalar(np.mean(prior)) if len(prior) else 0.0, _scalar(np.std(prior, ddof=1)) if len(prior) > 1 else 0.0])
    return values


@dataclass
class Ridge15MinForecaster:
    targets: tuple[str, ...] = TARGET_COLUMNS
    alpha: float = 1.0
    capacity_limits: Mapping[str, float] = field(default_factory=dict)
    coefficients: dict[str, np.ndarray] = field(default_factory=dict)
    feature_count: int = 0

    def fit(self, frame: pd.DataFrame) -> "Ridge15MinForecaster":
        required = [*WEATHER_COLUMNS, *self.targets]
        require_columns(frame, required, "training data")
        data = frame.copy().sort_index()
        positions = range(len(data))
        target_arrays = {target: data[target].to_numpy(dtype=float, na_value=np.nan) for target in self.targets}
        for target in self.targets:
            rows: list[list[float]] = []
            ys: list[float] = []
            target_values = target_arrays[target]
            for position, timestamp in zip(positions, data.index):
                value = target_values[position]
                if not np.isfinite(value):
                    continue
                rows.append(_feature_row(timestamp, data, target, position=position, target_values=target_values))
                ys.append(_scalar(value))
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
            position = len(state.index) - 1
            for target in self.targets:
                target_values = state[target].to_numpy(dtype=float, na_value=np.nan)
                row = _feature_row(timestamp, state, target, position=position, target_values=target_values)
                vector = np.asarray([1.0, *row], dtype=float)
                value = float(vector @ self.coefficients[target])
                if target.endswith("_power_kw") or target in {"load_kw", "wind_speed_mps", "solar_radiation_wm2"}:
                    value = max(0.0, value)
                limit = _scalar(self.capacity_limits.get(target), 0.0)
                if limit > 0 and (target.endswith("_power_kw") or target == "load_kw"):
                    value = min(limit, value)
                state.loc[timestamp, target] = value
                predictions.at[timestamp, target] = value
        return predictions
