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
    assert db.list_models()[0]["model_id"] == "MDL-1"
    assert db.list_training_runs()[0]["status"] == "success"
    assert db.list_schedules()[0]["schedule_id"] == "SCH-1"
    assert db.list_evaluations()[0]["status"] == "pending"
