from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from forecast_pipeline.storage import ForecastDB


def test_forecast_database_schema_and_roundtrip(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    assert db.list_points()[0]["id"] == "P001"
    db.create_run("TEST-1", "P001", "combined", "day_ahead")
    rows = [{
        "timestamp": "2026-01-01T00:00:00Z",
        "wind_speed_mps": 4.2,
        "temperature_c": 12.5,
        "solar_radiation_wm2": 0,
        "pv_power_kw": 0,
        "wind_power_kw": 8.0,
        "load_kw": 51.0,
    }]
    assert db.save_forecast_rows("TEST-1", "P001", "day_ahead", rows) == 1
    result = db.get_forecast_rows("P001", "day_ahead", 4)
    assert result[0]["wind_speed_mps"] == 4.2
    assert result[0]["load_kw"] == 51.0
    assert db.health()["tables"]["energy_forecasts"] == 1


def test_measurement_queries_apply_sql_time_bounds(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    db.save_weather_observations([
        {"point_id": "P001", "timestamp_utc": "2026-01-01T00:00:00Z", "wind_speed_mps": 1.0, "source": "local"},
        {"point_id": "P001", "timestamp_utc": "2026-01-01T00:15:00Z", "wind_speed_mps": 2.0, "source": "local"},
    ])
    db.save_energy_measurements([
        {"point_id": "P001", "timestamp_utc": "2026-01-01T00:00:00Z", "pv_power_kw": 1.0, "source": "local"},
        {"point_id": "P001", "timestamp_utc": "2026-01-01T00:15:00Z", "pv_power_kw": 2.0, "source": "local"},
    ])
    weather = db.list_weather_observations("P001", 20, "2026-01-01T00:15:00Z", "2026-01-01T00:15:00Z")
    energy = db.list_energy_measurements("P001", 20, "2026-01-01T00:15:00Z", "2026-01-01T00:15:00Z")
    assert [row["wind_speed_mps"] for row in weather] == [2.0]
    assert [row["pv_power_kw"] for row in energy] == [2.0]


def test_model_schedule_and_evaluation_roundtrip(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    db.register_model({
        "model_id": "MDL-1",
        "model_name": "Ridge",
        "model_type": "ridge15min",
        "target_kind": "combined",
        "horizon": "day_ahead",
        "point_id": "P001",
        "version": "v1",
        "metrics": {"load_kw": {"mae": 1.2}},
    })
    db.create_training_run("TRN-1", "P001", "combined", "day_ahead", "offline", 672)
    db.finish_training_run("TRN-1", "success", "MDL-1", 100, {"load_kw": {"mae": 1.2}})
    db.create_run("TEST-1", "P001", "combined", "day_ahead")
    db.upsert_schedule({"schedule_id": "SCH-1", "name": "日前", "point_ids": ["P001"], "forecast_kind": "combined", "horizon": "day_ahead", "cadence_minutes": 1440, "next_run_at": "2026-01-01T00:00:00Z"})
    db.save_evaluation({"evaluation_id": "EVA-1", "run_id": "TEST-1", "point_id": "P001", "forecast_kind": "combined", "horizon": "day_ahead", "sample_count": 0, "status": "pending"})
    db.save_evaluation({"evaluation_id": "EVA-1", "run_id": "TEST-1", "point_id": "P001", "forecast_kind": "combined", "horizon": "day_ahead", "sample_count": 4, "status": "success", "mae": 1.0})
    assert db.list_models()[0]["model_id"] == "MDL-1"
    assert db.list_training_runs()[0]["status"] == "success"
    assert db.list_schedules()[0]["schedule_id"] == "SCH-1"
    assert db.list_evaluations()[0]["status"] == "success"


def test_hour_weather_and_nwp_cache_roundtrip(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    db.create_run("HOUR-1", "P001", "weather", "hour_ahead")
    rows = [{
        "source_id": "openmeteo-icon",
        "point_id": "P001",
        "issued_at_utc": "2026-01-01T00:00:00Z",
        "target_time_utc": "2026-01-01T01:00:00Z",
        "horizon": "hour_ahead",
        "wind_speed_mps": 4.0,
        "temperature_c": 15.0,
        "solar_radiation_wm2": 120.0,
    }]
    assert db.save_nwp_forecasts(rows) == 1
    assert db.list_nwp_forecasts("P001", "hour_ahead")[0]["wind_speed_mps"] == 4.0
    assert db.save_forecast_rows("HOUR-1", "P001", "hour_ahead", [{
        "timestamp": "2026-01-01T01:00:00Z",
        "wind_speed_mps": 4.0,
        "temperature_c": 15.0,
        "solar_radiation_wm2": 120.0,
    }], kind="weather") == 1


def test_object_scope_and_curtailment_roundtrip(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    objects = db.list_objects()
    assert any(item["object_id"] == "OBJ-GRID" and item["object_type"] == "grid" for item in objects)
    assert db.save_energy_measurements([{
        "point_id": "P001",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "pv_power_kw": 50.0,
        "pv_available_power_kw": 80.0,
        "wind_power_kw": 10.0,
        "wind_available_power_kw": 10.0,
        "load_kw": 20.0,
        "curtailment_flag": 1,
        "curtailment_reason": "调度限发",
    }]) == 1
    measurement = db.list_energy_measurements("P001")[0]
    assert measurement["curtailment_flag"] == 1
    assert measurement["pv_curtailment_kw"] == 30.0
    assert measurement["curtailment_reason"] == "调度限发"
    assert db.save_energy_measurements([{
        "point_id": "P001",
        "timestamp_utc": "2026-01-01T00:15:00Z",
        "pv_power_kw": 50.0,
        "pv_available_power_kw": 80.0,
        "wind_power_kw": 10.0,
        "wind_available_power_kw": 10.0,
        "load_kw": 20.0,
        # A power difference alone is retained as raw telemetry, not a
        # confirmed curtailment event.
        "pv_curtailment_kw": 30.0,
    }]) == 1
    unconfirmed = [item for item in db.list_energy_measurements("P001") if item["timestamp_utc"].endswith("00:15:00Z")][0]
    assert unconfirmed["curtailment_flag"] == 0
    assert unconfirmed["pv_curtailment_kw"] == 30.0
    db.create_run("OBJ-RUN", "P001", "energy", "hour_ahead", object_id="OBJ-GRID")
    db.save_forecast_rows("OBJ-RUN", "P001", "hour_ahead", [{
        "timestamp": "2026-01-01T01:00:00Z",
        "pv_power_kw": 80.0,
        "wind_power_kw": 10.0,
        "load_kw": 20.0,
    }], kind="energy")
    assert db.list_runs(1)[0]["object_id"] == "OBJ-GRID"
    assert db.get_forecast_rows("P001", "hour_ahead", 4, object_id="OBJ-GRID")[0]["pv_power_kw"] == 80.0


def test_model_records_keep_object_scope(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    db.register_model({
        "model_id": "MDL-OBJECT",
        "model_name": "混合特征 Ridge",
        "model_type": "hybrid_ridge15min",
        "target_kind": "energy",
        "horizon": "hour_ahead",
        "point_id": "P002",
        "object_id": "OBJ-WIND-P002-U01",
        "version": "v-object",
        "metrics": {"wind_power_kw": {"mae": 0.5}},
    })
    db.create_training_run("TRN-OBJECT", "P002", "energy", "hour_ahead", "offline", 64, object_id="OBJ-WIND-P002-U01")
    db.save_model_update({"update_id": "UPD-OBJECT", "model_id": "MDL-OBJECT", "point_id": "P002", "object_id": "OBJ-WIND-P002-U01", "horizon": "hour_ahead", "window_size": 64})
    assert db.list_models(object_id="OBJ-WIND-P002-U01")[0]["object_id"] == "OBJ-WIND-P002-U01"
    assert db.list_training_runs()[0]["object_id"] == "OBJ-WIND-P002-U01"
    assert db.list_model_updates()[0]["object_id"] == "OBJ-WIND-P002-U01"


def test_run_filters_keep_horizon_and_object_scope(tmp_path):
    db = ForecastDB(tmp_path / "forecast.db")
    db.initialize()
    db.seed_defaults()
    db.create_run("RUN-HOUR", "P002", "energy", "hour_ahead", object_id="OBJ-WIND-P002-U01")
    db.create_run("RUN-DAY", "P002", "energy", "day_ahead", object_id="OBJ-WIND-P002")
    assert [row["run_id"] for row in db.list_runs(horizon="hour_ahead")] == ["RUN-HOUR"]
    assert [row["run_id"] for row in db.list_runs(object_id="OBJ-WIND-P002")] == ["RUN-DAY"]
def test_exact_run_and_weather_kind_do_not_use_another_object(tmp_path):
    db = ForecastDB(tmp_path / 'forecast.db')
    db.initialize()
    db.seed_defaults()
    for name, kind, object_id, value in [('OLD', 'weather', 'OBJ-GRID', 1), ('NEW', 'energy', 'OBJ-PV-P001-U01', 99)]:
        db.create_run(name, 'P001', kind, 'week_ahead', object_id=object_id)
        db.save_forecast_rows(name, 'P001', 'week_ahead', [{'timestamp': '2026-01-01T00:00:00Z', 'wind_speed_mps': value, 'pv_power_kw': value}], kind=kind)
    assert db.get_forecast_rows('P001', 'week_ahead', 672, kind='weather')[0]['wind_speed_mps'] == 1
    assert db.get_forecast_rows('P001', 'week_ahead', 672, run_id='OLD')[0]['wind_speed_mps'] == 1
    assert db.get_forecast_rows('P001', 'week_ahead', 672, run_id='OLD', object_id='OBJ-PV-P001-U01') == []
