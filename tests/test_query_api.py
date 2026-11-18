import importlib.util
import json
import threading
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

import pandas as pd
import pytest
from forecast_pipeline import storage


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'DEFAULT_DB_PATH', tmp_path / 'forecast.db')
    spec = importlib.util.spec_from_file_location('isolated_forecast_web', Path(__file__).parents[1] / 'scripts/serve_web.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DB.create_run('QUERY-OLD', 'P002', 'weather', 'hour_ahead', object_id='OBJ-WIND-P002-U01')
    module.DB.save_forecast_rows('QUERY-OLD', 'P002', 'hour_ahead', [{'timestamp':'2026-01-01T00:00:00Z','wind_speed_mps':3}], kind='weather')
    module.DB.create_run('QUERY-NEW', 'P002', 'energy', 'hour_ahead', object_id='OBJ-WIND-P002')
    module.DB.save_forecast_rows('QUERY-NEW', 'P002', 'hour_ahead', [{'timestamp':'2026-01-01T00:00:00Z','wind_power_kw':99}], kind='energy')
    module.DB.save_energy_measurements([{'point_id':'P002','timestamp_utc':'2026-01-01T00:00:00Z','pv_power_kw':10,'pv_available_power_kw':20,'source':'test'}])
    module.DB.save_energy_measurements([{'point_id':'P002','timestamp_utc':'2026-01-01T00:15:00Z','pv_power_kw':10,'pv_available_power_kw':20,'pv_curtailment_kw':10,'source':'local'}])
    server = module.ThreadingHTTPServer(('127.0.0.1', 0), module.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown(); server.server_close(); worker.join()


def test_exact_query_derives_object_horizon_and_rejects_conflict(api):
    result = json.load(urlopen(api + '/api/forecast?run_id=QUERY-OLD'))
    assert result['point_id'] == 'P002'
    assert result['object_id'] == 'OBJ-WIND-P002-U01'
    assert result['horizon'] == 'hour_ahead'
    assert result['run']['run_id'] == 'QUERY-OLD'
    assert result['run']['forecast_kind'] == 'weather'
    assert result['rows'][0]['wind_speed_mps'] == 3
    with pytest.raises(HTTPError) as error:
        urlopen(api + '/api/forecast?run_id=QUERY-OLD&horizon=week_ahead')
    assert error.value.code == 400


def test_weather_query_returns_weather_run_when_energy_is_newer(api):
    result = json.load(urlopen(api + '/api/forecast?point_id=P002&horizon=hour_ahead&kind=weather'))
    assert result['run_id'] == 'QUERY-OLD'
    assert result['run']['status'] == 'success'
    assert result['object_id'] == 'OBJ-WIND-P002-U01'
    assert result['rows'][0]['wind_speed_mps'] == 3


def test_legacy_point_runs_can_return_to_latest_results(api):
    result = json.load(urlopen(api + '/api/forecast?object_id=P001&horizon=week_ahead&kind=energy'))
    assert result['run_id'] == 'SEED-WEEK-AHEAD'
    assert result['object_id'] == 'P001'
    assert len(result['rows']) == 672
    exact = json.load(urlopen(api + '/api/forecast?run_id=SEED-WEEK-AHEAD&object_id=P001'))
    assert exact['rows'] == result['rows']


def test_test_source_curtailment_is_reported_as_unknown(api):
    result = json.load(urlopen(api + '/api/curtailment-summary?point_id=P002'))
    assert result['known'] is False
    assert result['curtailment_samples'] == 0
    assert result['unknown_samples'] == 2
    assert result['pv_loss_kw'] is None
    assert result['first_confirmed_at'] is None
    assert result['last_confirmed_at'] is None


def test_unconfirmed_available_power_stays_out_of_training_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'DEFAULT_DB_PATH', tmp_path / 'forecast.db')
    spec = importlib.util.spec_from_file_location('unconfirmed_training_web', Path(__file__).parents[1] / 'scripts/serve_web.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DB.save_energy_measurements([{
        'point_id': 'P002',
        'timestamp_utc': '2026-01-01T00:00:00Z',
        'pv_power_kw': 10,
        'pv_available_power_kw': 20,
        'source': 'local',
    }])
    frame = module.load_energy_observation_frame('P002')
    assert frame.iloc[0]['pv_power_kw'] == 10
    prepared, quality = module.prepare_power_training_frame(pd.DataFrame([{
        'timestamp': '2026-01-01T00:00:00Z',
        'source': 'local',
        'pv_power_kw': 10,
        'pv_available_power_kw': 20,
        'curtailment_flag': 0,
    }]))
    assert prepared.iloc[0]['pv_power_kw'] == 10
    assert quality == {'curtailment_samples': 0, 'adjusted_samples': 0}


def test_measurement_api_applies_sql_time_bounds(api):
    result = json.load(urlopen(api + '/api/energy-measurements?point_id=P002&limit=20&start=2026-01-01T00:15:00Z&end=2026-01-01T00:15:00Z'))
    assert result['count'] == 1
    assert result['measurements'][0]['timestamp_utc'] == '2026-01-01T00:15:00Z'


def test_invalid_measurement_time_range_is_rejected(api):
    with pytest.raises(HTTPError) as error:
        urlopen(api + '/api/energy-measurements?point_id=P002&start=2026-01-02T00:00:00Z&end=2026-01-01T00:00:00Z')
    assert error.value.code == 400
