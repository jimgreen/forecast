import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from forecast_pipeline.model import Ridge15MinForecaster, _feature_row, _base_features, _feature_matrix
from forecast_pipeline.cache import PipelineCache
from forecast_pipeline.schema import WEATHER_COLUMNS, TARGET_COLUMNS
from scripts.demo import make_demo_data


def test_vector_features_match_scalar_with_missing_samples():
    frame = make_demo_data().iloc[:800].set_index('timestamp')
    frame.loc[frame.index[[0, 7, 100]], 'load_kw'] = np.nan
    frame.loc[frame.index[10], 'wind_speed_mps'] = np.inf
    base = _base_features(frame.index, frame[list(WEATHER_COLUMNS)].to_numpy())
    expected = np.array([[1, *_feature_row(t, frame, 'load_kw')] for t in frame.index])
    np.testing.assert_allclose(_feature_matrix(frame, 'load_kw', base), expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('targets', [TARGET_COLUMNS, (*WEATHER_COLUMNS, *TARGET_COLUMNS)])
def test_recursive_prediction_matches_scalar_and_does_not_mutate_history(targets):
    frame = make_demo_data().iloc[:800].set_index('timestamp')
    original = frame.copy(deep=True)
    model = Ridge15MinForecaster(targets=targets, capacity_limits={'pv_power_kw': 20}).fit(frame)
    future = frame.iloc[-96:][list(WEATHER_COLUMNS)].copy()
    future.index = pd.date_range(frame.index[-1] + pd.Timedelta(minutes=15), periods=96, freq='15min')
    state = frame.copy()
    expected = []
    for timestamp in future.index:
        state.loc[timestamp, list(WEATHER_COLUMNS)] = future.loc[timestamp, list(WEATHER_COLUMNS)]
        values = []
        for target in targets:
            value = float(np.array([1, *_feature_row(timestamp, state, target)]) @ model.coefficients[target])
            if target != 'temperature_c':
                value = max(0, value)
            if target == 'pv_power_kw':
                value = min(20, value)
            state.loc[timestamp, target] = value
            values.append(value)
        expected.append(values)
    np.testing.assert_allclose(model.predict(frame, future).to_numpy(), expected, atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(frame, original)


def test_cache_invalidates_early_backfills_and_shares_identical_work():
    frame = make_demo_data().iloc[:128].set_index('timestamp')
    cache = PipelineCache(maxsize=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda _: cache.get(frame, TARGET_COLUMNS, {}, 'point'), range(4)))
    assert all(model is models[0] for model in models)
    frame.iloc[0, frame.columns.get_loc('load_kw')] += 1
    corrected = cache.get(frame, TARGET_COLUMNS, {}, 'point')
    assert corrected is not models[0]
    cache.get(frame, TARGET_COLUMNS, {'pv_power_kw': 1}, 'point')
    assert len(cache.entries) == 2
