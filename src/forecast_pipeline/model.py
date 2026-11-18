"""Small dependency-light multi-target 15-minute forecaster.

The class is intentionally simple and auditable. It provides a strong baseline
for integration and can later be replaced by OpenSTEF, Quartz or a deep model
without changing the input/output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from math import fsum, isfinite, sqrt
from typing import Mapping

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

    Kept as a scalar reference for feature and recursive-prediction verification.
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


def _base_features(index: pd.DatetimeIndex, weather: np.ndarray) -> np.ndarray:
    hour = np.asarray(index.hour + index.minute / 60.0)
    quarter = np.asarray(index.hour * 4 + index.minute // 15)
    day = np.asarray(index.dayofweek)
    weather = np.where(np.isfinite(weather), weather, 0.0).copy()
    weather[:, 2] = np.maximum(weather[:, 2], 0.0)
    return np.column_stack((weather, np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                            np.sin(2*np.pi*quarter/96), np.cos(2*np.pi*quarter/96),
                            np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)))


def _feature_matrix(data: pd.DataFrame, target: str, base: np.ndarray) -> np.ndarray:
    values = data[target].replace([np.inf, -np.inf], np.nan)
    parts = [np.ones((len(data), 1)), base]
    parts.extend(values.shift(lag).fillna(0).to_numpy()[:, None] for lag in (1, 2, 4, 96, 672))
    # Shift first: the current target must never enter its rolling features.
    prior = values.shift(1)
    for window in (4, 16, 96):
        rolling = prior.rolling(window, min_periods=1)
        parts.extend((rolling.mean().fillna(0).to_numpy()[:, None],
                      rolling.std(ddof=1).fillna(0).to_numpy()[:, None]))
    return np.column_stack(parts)


class _RollingWindow:
    """Finite-only sliding moments with a local origin to limit cancellation."""

    def __init__(self, size: int, values):
        self.size = size
        self.values = deque(float(value) for value in values)
        self.updates = 0
        self._rebuild()

    def _rebuild(self):
        finite = [value for value in self.values if isfinite(value)]
        self.count = len(finite)
        self.origin = finite[0] if finite else 0.0
        shifted = [value - self.origin for value in finite]
        self.total = fsum(shifted)
        self.squares = fsum(value * value for value in shifted)

    def append(self, value):
        if len(self.values) == self.size:
            old = self.values.popleft()
            if isfinite(old):
                shifted = old - self.origin
                self.count -= 1
                self.total -= shifted
                self.squares -= shifted * shifted
        value = float(value)
        self.values.append(value)
        if isfinite(value):
            shifted = value - self.origin
            self.count += 1
            self.total += shifted
            self.squares += shifted * shifted
        self.updates += 1
        # Bound accumulated rounding error, including all-missing transitions.
        if self.updates % 96 == 0 or self.count <= 1:
            self._rebuild()

    def moments(self):
        if not self.count:
            return 0.0, 0.0
        mean = self.origin + self.total / self.count
        if self.count == 1:
            return mean, 0.0
        centered = self.squares - self.total * self.total / self.count
        if centered <= 1e-10 * max(self.squares, 1.0):
            # Nearly constant windows need a direct, stable variance calculation.
            centered = fsum((value - mean) ** 2 for value in self.values if isfinite(value))
        return mean, sqrt(max(0.0, centered) / (self.count - 1))


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
        base = _base_features(data.index, data[list(WEATHER_COLUMNS)].to_numpy(dtype=float))
        for target in self.targets:
            y = data[target].to_numpy(dtype=float)
            valid = np.isfinite(y)
            if valid.sum() < 32:
                raise ValueError(f"not enough valid rows to train {target}: {valid.sum()}")
            x = _feature_matrix(data, target, base)[valid]
            y = y[valid]
            penalty = self.alpha * np.eye(x.shape[1])
            penalty[0, 0] = 0.0
            self.coefficients[target] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            self.feature_count = x.shape[1]
        return self

    def predict(self, history: pd.DataFrame, future_weather: pd.DataFrame) -> pd.DataFrame:
        require_columns(history, [*WEATHER_COLUMNS, *self.targets], "history")
        require_columns(future_weather, WEATHER_COLUMNS, "future weather")
        future = future_weather.copy().sort_index()
        # Allocate once instead of growing a DataFrame for every forecast step.
        index = history.index.union(future.index).sort_values()
        columns = list(dict.fromkeys((*WEATHER_COLUMNS, *self.targets)))
        state = history.reindex(index)[columns].to_numpy(dtype=float, copy=True)
        offsets = {name: i for i, name in enumerate(columns)}
        base = _base_features(future.index, future[list(WEATHER_COLUMNS)].to_numpy(dtype=float))
        predictions = np.empty((len(future), len(self.targets)))
        positions = index.get_indexer(future.index)
        first_position = int(positions[0]) if len(positions) else 0
        windows = {
            target: [_RollingWindow(size, state[max(0, first_position-size):first_position, offsets[target]])
                     for size in (4, 16, 96)]
            for target in self.targets
        }
        previous_position = first_position
        for step, position in enumerate(positions):
            for prior_row in state[previous_position:position]:
                for target in self.targets:
                    for window in windows[target]:
                        window.append(prior_row[offsets[target]])
            previous_position = position
            state[position, :3] = base[step, :3]
            for target_index, target in enumerate(self.targets):
                values = state[:, offsets[target]]
                row = [*np.where(np.isfinite(state[position, :3]), state[position, :3], 0), *base[step, 3:]]
                row[2] = max(0.0, row[2])
                row.extend(_scalar(values[position-lag]) if position >= lag else 0.0 for lag in (1, 2, 4, 96, 672))
                for window in windows[target]:
                    row.extend(window.moments())
                vector = np.asarray([1.0, *row])
                value = float(vector @ self.coefficients[target])
                if target.endswith("_power_kw") or target in {"load_kw", "wind_speed_mps", "solar_radiation_wm2"}:
                    value = max(0.0, value)
                limit = _scalar(self.capacity_limits.get(target), 0.0)
                if limit > 0 and (target.endswith("_power_kw") or target == "load_kw"):
                    value = min(limit, value)
                state[position, offsets[target]] = value
                predictions[step, target_index] = value
        return pd.DataFrame(predictions, index=future.index, columns=self.targets)
