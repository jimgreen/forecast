"""Alternate old/new model runs on identical offline data; no live DB writes."""
import json
import subprocess
import sys
import types
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from forecast_pipeline.model import Ridge15MinForecaster
from forecast_pipeline.schema import WEATHER_COLUMNS
from scripts.demo import make_demo_data


def main():
    revision = '2d59c8b'
    source = subprocess.check_output(['git', 'show', f'{revision}:src/forecast_pipeline/model.py'], cwd=ROOT, text=True, encoding='utf-8')
    module = types.ModuleType('forecast_pipeline._benchmark_reference')
    sys.modules[module.__name__] = module
    exec(compile(source, '<baseline>', 'exec'), module.__dict__)
    classes = {'baseline': module.Ridge15MinForecaster, 'optimized': Ridge15MinForecaster}
    history = make_demo_data().iloc[:14*96].set_index('timestamp')
    future = history.iloc[-672:][list(WEATHER_COLUMNS)].copy()
    future.index = pd.date_range(history.index[-1]+pd.Timedelta(minutes=15), periods=672, freq='15min')
    times = {key: {'fit': [], 'predict': []} for key in classes}
    results = {}
    for round_index in range(3):
        for key in (list(classes) if round_index % 2 == 0 else list(reversed(classes))):
            started = perf_counter()
            model = classes[key]().fit(history)
            times[key]['fit'].append(perf_counter()-started)
            started = perf_counter()
            results[key] = model.predict(history, future).to_numpy()
            times[key]['predict'].append(perf_counter()-started)
    np.testing.assert_allclose(results['baseline'], results['optimized'], atol=1e-6, rtol=1e-6)
    print(json.dumps({'baseline_commit': revision, 'training_rows': len(history), 'forecast_rows': len(future),
                      'rounds': 3, 'seconds': times,
                      'median_seconds': {k: {phase: median(values) for phase, values in v.items()} for k,v in times.items()},
                      'max_absolute_difference': float(np.max(np.abs(results['baseline']-results['optimized'])))}, indent=2))


if __name__ == '__main__':
    main()
