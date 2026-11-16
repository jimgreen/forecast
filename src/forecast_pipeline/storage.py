"""SQLite persistence for the forecast UI and calculation backend.

The database is deliberately kept behind a small repository class. Forecast
models can write pandas records through the bulk methods, while the HMI reads
stable JSON-shaped dictionaries without knowing the SQL schema.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(UTC_FORMAT)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_points (
    point_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    point_type TEXT NOT NULL DEFAULT 'custom',
    latitude REAL,
    longitude REAL,
    pv_capacity_kw REAL NOT NULL DEFAULT 0,
    wind_capacity_kw REAL NOT NULL DEFAULT 0,
    objects_json TEXT NOT NULL DEFAULT '["pv", "wind", "load"]',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    model_name TEXT NOT NULL DEFAULT 'Ridge 15分钟基线',
    weather_source TEXT NOT NULL DEFAULT '本地采集 + Open-Meteo',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY,
    point_id TEXT,
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    trigger_type TEXT NOT NULL DEFAULT 'manual' CHECK(trigger_type IN ('manual', 'scheduled', 'api', 'seed')),
    scheduled_at TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('accepted', 'running', 'success', 'failed')),
    rows_written INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS weather_observations (
    point_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    wind_speed_mps REAL,
    temperature_c REAL,
    solar_radiation_wm2 REAL,
    source TEXT NOT NULL,
    quality_code TEXT NOT NULL DEFAULT 'valid',
    is_realtime INTEGER NOT NULL DEFAULT 0 CHECK(is_realtime IN (0, 1)),
    received_at TEXT NOT NULL,
    PRIMARY KEY(point_id, timestamp_utc, source),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS weather_forecasts (
    run_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    issued_at_utc TEXT NOT NULL,
    target_time_utc TEXT NOT NULL,
    horizon TEXT NOT NULL,
    wind_speed_mps REAL,
    temperature_c REAL,
    solar_radiation_wm2 REAL,
    source TEXT NOT NULL,
    model_name TEXT NOT NULL,
    quality_score REAL,
    PRIMARY KEY(run_id, target_time_utc),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS energy_measurements (
    point_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    pv_power_kw REAL,
    wind_power_kw REAL,
    load_kw REAL,
    source TEXT NOT NULL,
    quality_code TEXT NOT NULL DEFAULT 'valid',
    is_realtime INTEGER NOT NULL DEFAULT 0 CHECK(is_realtime IN (0, 1)),
    received_at TEXT NOT NULL,
    PRIMARY KEY(point_id, timestamp_utc, source),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS energy_forecasts (
    run_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    issued_at_utc TEXT NOT NULL,
    target_time_utc TEXT NOT NULL,
    horizon TEXT NOT NULL,
    pv_power_kw REAL,
    wind_power_kw REAL,
    load_kw REAL,
    model_name TEXT NOT NULL,
    quality_score REAL,
    PRIMARY KEY(run_id, target_time_utc),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS nwp_sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    endpoint TEXT,
    refresh_minutes INTEGER NOT NULL DEFAULT 60,
    variables_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'unknown',
    last_sync_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_tasks (
    task_id TEXT PRIMARY KEY,
    point_id TEXT NOT NULL,
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    schedule_time TEXT,
    timezone TEXT NOT NULL DEFAULT 'Pacific/Auckland',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    target_kind TEXT NOT NULL CHECK(target_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    point_id TEXT,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('training', 'active', 'retired', 'failed')),
    artifact_path TEXT,
    hyperparams_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    trained_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS training_runs (
    training_id TEXT PRIMARY KEY,
    model_id TEXT,
    point_id TEXT,
    target_kind TEXT NOT NULL CHECK(target_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    training_mode TEXT NOT NULL CHECK(training_mode IN ('offline', 'rolling')),
    window_size INTEGER,
    sample_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('accepted', 'running', 'success', 'failed')),
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS model_updates (
    update_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    point_id TEXT,
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    update_strategy TEXT NOT NULL CHECK(update_strategy IN ('rolling_window', 'incremental', 'retrain')),
    window_size INTEGER,
    sample_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('accepted', 'success', 'failed')),
    created_at TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS forecast_schedules (
    schedule_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    point_ids_json TEXT NOT NULL DEFAULT '[]',
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    cadence_minutes INTEGER NOT NULL DEFAULT 1440,
    next_run_at TEXT,
    last_run_at TEXT,
    last_run_id TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    point_id TEXT,
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('ultrashort', 'day_ahead', 'week_ahead')),
    window_start TEXT,
    window_end TEXT,
    sample_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('pending', 'success', 'failed')),
    mae REAL,
    rmse REAL,
    bias REAL,
    nmae REAL,
    coverage REAL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE INDEX IF NOT EXISTS idx_weather_obs_point_time ON weather_observations(point_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_weather_fc_point_target ON weather_forecasts(point_id, target_time_utc);
CREATE INDEX IF NOT EXISTS idx_energy_meas_point_time ON energy_measurements(point_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_energy_fc_point_target ON energy_forecasts(point_id, target_time_utc);
CREATE INDEX IF NOT EXISTS idx_runs_point_started ON forecast_runs(point_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_point_horizon ON model_registry(point_id, horizon, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_started ON training_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_next_run ON forecast_schedules(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_eval_run ON forecast_evaluations(run_id, evaluated_at DESC);
"""


class ForecastDB:
    """Repository for the forecast SQLite database."""

    def __init__(self, path: str | Path = "data/forecast.db") -> None:
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(forecast_points)")}
            if "objects_json" not in columns:
                conn.execute("ALTER TABLE forecast_points ADD COLUMN objects_json TEXT NOT NULL DEFAULT '[\"pv\", \"wind\", \"load\"]'")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def seed_defaults(self) -> None:
        now = utc_now()
        points = [
            ("P001", "南岛综合能源站", "风光储站", -36.85, 174.76, 120, 80, 1, "Ridge 15分钟基线", "本地采集 + Open-Meteo"),
            ("P002", "北侧风机群", "风电场", -36.72, 174.85, 0, 420, 1, "hefty NWP + PV model", "本地采集 + NOAA NWP"),
            ("P003", "园区负荷汇总", "负荷点", -36.88, 174.72, 0, 0, 1, "OpenSTEF probabilistic", "本地采集 + Open-Meteo"),
        ]
        with self.connection() as conn:
            for row in points:
                conn.execute(
                    """INSERT INTO forecast_points
                    (point_id,name,point_type,latitude,longitude,pv_capacity_kw,wind_capacity_kw,objects_json,enabled,model_name,weather_source,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(point_id) DO UPDATE SET name=excluded.name, point_type=excluded.point_type,
                    latitude=excluded.latitude, longitude=excluded.longitude, pv_capacity_kw=excluded.pv_capacity_kw,
                    objects_json=excluded.objects_json,
                    wind_capacity_kw=excluded.wind_capacity_kw, updated_at=excluded.updated_at""",
                    (*row[:7], json.dumps(self._objects_for_point(row[2], row[5], row[6]), ensure_ascii=False), *row[7:], now, now),
                )
            nwp = [
                ("openmeteo-icon", "Open-Meteo ICON", "Open-Meteo", "ICON", "https://api.open-meteo.com/v1/forecast", 15, ["wind_speed_10m", "temperature_2m", "shortwave_radiation"]),
                ("noaa-gfs", "NOAA GFS", "NOAA", "GFS", None, 60, ["wind_speed_10m", "temperature_2m", "shortwave_radiation"]),
            ]
            for source_id, name, provider, model, endpoint, refresh, variables in nwp:
                conn.execute(
                    """INSERT INTO nwp_sources(source_id,name,provider,model_name,endpoint,refresh_minutes,variables_json,enabled,status,updated_at)
                    VALUES(?,?,?,?,?,?,?,1,'configured',?)
                    ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, endpoint=excluded.endpoint,
                    refresh_minutes=excluded.refresh_minutes, variables_json=excluded.variables_json, updated_at=excluded.updated_at""",
                    (source_id, name, provider, model, endpoint, refresh, json.dumps(variables, ensure_ascii=False), now),
                )
            conn.execute(
                """INSERT INTO forecast_schedules
                (schedule_id,name,point_ids_json,forecast_kind,horizon,enabled,cadence_minutes,next_run_at,updated_at)
                VALUES(?,?,?,?,?,1,?,?,?)
                ON CONFLICT(schedule_id) DO NOTHING""",
                ("SCH-DEFAULT-DAY-AHEAD", "默认日前预测", json.dumps(["P001", "P002", "P003"]), "combined", "day_ahead", 1440, now, now),
            )

    def upsert_point(self, point: Mapping[str, object]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO forecast_points(point_id,name,point_type,latitude,longitude,pv_capacity_kw,wind_capacity_kw,objects_json,enabled,model_name,weather_source,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(point_id) DO UPDATE SET name=excluded.name, point_type=excluded.point_type,
                latitude=excluded.latitude, longitude=excluded.longitude, pv_capacity_kw=excluded.pv_capacity_kw,
                objects_json=excluded.objects_json,
                wind_capacity_kw=excluded.wind_capacity_kw, enabled=excluded.enabled, model_name=excluded.model_name,
                weather_source=excluded.weather_source, updated_at=excluded.updated_at""",
                (
                    point["id"], point["name"], point.get("type", "custom"), point.get("lat"), point.get("lon"),
                    point.get("pv", 0), point.get("wind", 0), json.dumps(point.get("objects", self._objects_for_point(point.get("type", "custom"), point.get("pv", 0), point.get("wind", 0))), ensure_ascii=False), int(bool(point.get("enabled", True))),
                    point.get("model", "Ridge 15分钟基线"), point.get("source", "本地采集 + Open-Meteo"), now, now,
                ),
            )

    def list_points(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM forecast_points ORDER BY point_id").fetchall()
        return [
            {"id": r["point_id"], "name": r["name"], "type": r["point_type"], "lat": r["latitude"], "lon": r["longitude"], "pv": r["pv_capacity_kw"], "wind": r["wind_capacity_kw"], "objects": json.loads(r["objects_json"] or "[]"), "enabled": bool(r["enabled"]), "model": r["model_name"], "source": r["weather_source"], "quality": "待检查", "lastRun": "未运行"}
            for r in rows
        ]

    @staticmethod
    def _objects_for_point(point_type: object, pv: object, wind: object) -> list[str]:
        point_type = str(point_type or "").lower()
        if "负荷" in point_type or point_type in {"load", "load_point"}:
            return ["load"]
        objects = []
        if float(pv or 0) > 0:
            objects.append("pv")
        if float(wind or 0) > 0:
            objects.append("wind")
        if "综合" in point_type or "储" in point_type:
            objects.append("load")
        return objects or ["load"]

    def create_run(self, run_id: str, point_id: str | None, kind: str, horizon: str, trigger: str = "manual", scheduled_at: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO forecast_runs(run_id,point_id,forecast_kind,horizon,trigger_type,scheduled_at,started_at,status) VALUES(?,?,?,?,?,?,?,'accepted')",
                (run_id, point_id, kind, horizon, trigger, scheduled_at, utc_now()),
            )

    def finish_run(self, run_id: str, status: str, rows_written: int = 0, error_message: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE forecast_runs SET status=?, finished_at=?, rows_written=?, error_message=? WHERE run_id=?", (status, utc_now(), rows_written, error_message, run_id))

    def list_runs(self, limit: int = 50) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM forecast_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def register_model(self, model: Mapping[str, object]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO model_registry
                (model_id,model_name,model_type,target_kind,horizon,point_id,version,status,artifact_path,hyperparams_json,metrics_json,trained_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id) DO UPDATE SET model_name=excluded.model_name,model_type=excluded.model_type,
                target_kind=excluded.target_kind,horizon=excluded.horizon,point_id=excluded.point_id,version=excluded.version,
                status=excluded.status,artifact_path=excluded.artifact_path,hyperparams_json=excluded.hyperparams_json,
                metrics_json=excluded.metrics_json,updated_at=excluded.updated_at""",
                (model["model_id"], model.get("model_name", "Ridge 15分钟基线"), model.get("model_type", "ridge"),
                 model.get("target_kind", "combined"), model.get("horizon", "day_ahead"), model.get("point_id"),
                 model.get("version", "v1"), model.get("status", "active"), model.get("artifact_path"),
                 json.dumps(model.get("hyperparams", {}), ensure_ascii=False), json.dumps(model.get("metrics", {}), ensure_ascii=False),
                 model.get("trained_at", now), now),
            )

    def list_models(self, point_id: str | None = None, horizon: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 500)
        clauses, params = [], []
        if point_id:
            clauses.append("point_id=?"); params.append(point_id)
        if horizon:
            clauses.append("horizon=?"); params.append(horizon)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(f"SELECT * FROM model_registry {where} ORDER BY updated_at DESC LIMIT ?", (*params, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["hyperparams"] = json.loads(item.pop("hyperparams_json") or "{}")
            item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
            result.append(item)
        return result

    def create_training_run(self, training_id: str, point_id: str | None, target_kind: str, horizon: str, mode: str, window_size: int | None = None, model_id: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO training_runs(training_id,model_id,point_id,target_kind,horizon,training_mode,window_size,started_at,status) VALUES(?,?,?,?,?,?,?,?, 'running')",
                (training_id, model_id, point_id, target_kind, horizon, mode, window_size, utc_now()),
            )

    def finish_training_run(self, training_id: str, status: str, model_id: str | None = None, sample_count: int = 0, metrics: Mapping[str, object] | None = None, error_message: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE training_runs SET status=?,model_id=COALESCE(?,model_id),finished_at=?,sample_count=?,metrics_json=?,error_message=? WHERE training_id=?",
                (status, model_id, utc_now(), sample_count, json.dumps(metrics or {}, ensure_ascii=False), error_message, training_id),
            )

    def list_training_runs(self, limit: int = 50) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?", (min(max(int(limit), 1), 200),)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["metrics"] = json.loads(item.pop("metrics_json") or "{}"); result.append(item)
        return result

    def save_model_update(self, update: Mapping[str, object]) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO model_updates(update_id,model_id,point_id,horizon,update_strategy,window_size,sample_count,status,created_at,details_json)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (update["update_id"], update["model_id"], update.get("point_id"), update["horizon"], update.get("update_strategy", "rolling_window"), update.get("window_size"), update.get("sample_count", 0), update.get("status", "success"), update.get("created_at", utc_now()), json.dumps(update.get("details", {}), ensure_ascii=False)),
            )

    def list_model_updates(self, limit: int = 50) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM model_updates ORDER BY created_at DESC LIMIT ?", (min(max(int(limit), 1), 200),)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["details"] = json.loads(item.pop("details_json") or "{}"); result.append(item)
        return result

    def upsert_schedule(self, schedule: Mapping[str, object]) -> None:
        now = utc_now()
        point_ids = schedule.get("point_ids", [])
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO forecast_schedules(schedule_id,name,point_ids_json,forecast_kind,horizon,enabled,cadence_minutes,next_run_at,last_run_at,last_run_id,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(schedule_id) DO UPDATE SET name=excluded.name,point_ids_json=excluded.point_ids_json,
                forecast_kind=excluded.forecast_kind,horizon=excluded.horizon,enabled=excluded.enabled,cadence_minutes=excluded.cadence_minutes,
                next_run_at=excluded.next_run_at,updated_at=excluded.updated_at""",
                (schedule["schedule_id"], schedule.get("name", "预测定时任务"), json.dumps(point_ids, ensure_ascii=False), schedule.get("forecast_kind", "combined"), schedule.get("horizon", "day_ahead"), int(bool(schedule.get("enabled", True))), int(schedule.get("cadence_minutes", 1440)), schedule.get("next_run_at"), schedule.get("last_run_at"), schedule.get("last_run_id"), now),
            )

    def list_schedules(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM forecast_schedules ORDER BY name").fetchall()
        result = []
        for row in rows:
            item = dict(row); item["point_ids"] = json.loads(item.pop("point_ids_json") or "[]"); item["enabled"] = bool(item["enabled"]); result.append(item)
        return result

    def mark_schedule_run(self, schedule_id: str, run_id: str, next_run_at: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE forecast_schedules SET last_run_at=?,last_run_id=?,next_run_at=?,updated_at=? WHERE schedule_id=?", (utc_now(), run_id, next_run_at, utc_now(), schedule_id))

    def save_evaluation(self, evaluation: Mapping[str, object]) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO forecast_evaluations(evaluation_id,run_id,point_id,forecast_kind,horizon,window_start,window_end,sample_count,status,mae,rmse,bias,nmae,coverage,metrics_json,evaluated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (evaluation["evaluation_id"], evaluation["run_id"], evaluation.get("point_id"), evaluation["forecast_kind"], evaluation["horizon"], evaluation.get("window_start"), evaluation.get("window_end"), evaluation.get("sample_count", 0), evaluation.get("status", "pending"), evaluation.get("mae"), evaluation.get("rmse"), evaluation.get("bias"), evaluation.get("nmae"), evaluation.get("coverage"), json.dumps(evaluation.get("metrics", {}), ensure_ascii=False), evaluation.get("evaluated_at", utc_now())),
            )

    def list_evaluations(self, run_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connection() as conn:
            if run_id:
                rows = conn.execute("SELECT * FROM forecast_evaluations WHERE run_id=? ORDER BY evaluated_at DESC LIMIT ?", (run_id, min(max(int(limit), 1), 500))).fetchall()
            else:
                rows = conn.execute("SELECT * FROM forecast_evaluations ORDER BY evaluated_at DESC LIMIT ?", (min(max(int(limit), 1), 500),)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["metrics"] = json.loads(item.pop("metrics_json") or "{}"); result.append(item)
        return result

    def save_weather_observations(self, rows: Iterable[Mapping[str, object]]) -> int:
        count = 0
        with self.connection() as conn:
            for row in rows:
                conn.execute(
                    """INSERT INTO weather_observations(point_id,timestamp_utc,wind_speed_mps,temperature_c,solar_radiation_wm2,source,quality_code,is_realtime,received_at)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(point_id,timestamp_utc,source) DO UPDATE SET wind_speed_mps=excluded.wind_speed_mps,temperature_c=excluded.temperature_c,solar_radiation_wm2=excluded.solar_radiation_wm2,quality_code=excluded.quality_code,is_realtime=excluded.is_realtime,received_at=excluded.received_at""",
                    (row["point_id"], row["timestamp_utc"], row.get("wind_speed_mps"), row.get("temperature_c"), row.get("solar_radiation_wm2"), row.get("source", "local"), row.get("quality_code", "valid"), int(bool(row.get("is_realtime", False))), row.get("received_at", utc_now())),
                )
                count += 1
        return count

    def save_energy_measurements(self, rows: Iterable[Mapping[str, object]]) -> int:
        count = 0
        with self.connection() as conn:
            for row in rows:
                conn.execute(
                    """INSERT INTO energy_measurements(point_id,timestamp_utc,pv_power_kw,wind_power_kw,load_kw,source,quality_code,is_realtime,received_at)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(point_id,timestamp_utc,source) DO UPDATE SET pv_power_kw=excluded.pv_power_kw,wind_power_kw=excluded.wind_power_kw,load_kw=excluded.load_kw,quality_code=excluded.quality_code,is_realtime=excluded.is_realtime,received_at=excluded.received_at""",
                    (row["point_id"], row["timestamp_utc"], row.get("pv_power_kw"), row.get("wind_power_kw"), row.get("load_kw"), row.get("source", "local"), row.get("quality_code", "valid"), int(bool(row.get("is_realtime", False))), row.get("received_at", utc_now())),
                )
                count += 1
        return count

    def list_weather_observations(self, point_id: str | None = None, limit: int | str = 96) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 2000)
        with self.connection() as conn:
            if point_id:
                rows = conn.execute("SELECT * FROM weather_observations WHERE point_id=? ORDER BY timestamp_utc DESC LIMIT ?", (point_id, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM weather_observations ORDER BY timestamp_utc DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def list_energy_measurements(self, point_id: str | None = None, limit: int | str = 96) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 2000)
        with self.connection() as conn:
            if point_id:
                rows = conn.execute("SELECT * FROM energy_measurements WHERE point_id=? ORDER BY timestamp_utc DESC LIMIT ?", (point_id, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM energy_measurements ORDER BY timestamp_utc DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def save_forecast_rows(self, run_id: str, point_id: str, horizon: str, rows: Iterable[Mapping[str, object]], kind: str = "combined", model_name: str = "Ridge 15分钟基线", source: str = "local") -> int:
        rows = list(rows)
        issued = utc_now()
        with self.connection() as conn:
            for row in rows:
                if kind in {"weather", "combined"} and any(key in row for key in ("wind_speed_mps", "temperature_c", "solar_radiation_wm2")):
                    conn.execute(
                        "INSERT OR REPLACE INTO weather_forecasts(run_id,point_id,issued_at_utc,target_time_utc,horizon,wind_speed_mps,temperature_c,solar_radiation_wm2,source,model_name,quality_score) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, point_id, issued, row["timestamp"], horizon, row.get("wind_speed_mps"), row.get("temperature_c"), row.get("solar_radiation_wm2"), source, model_name, row.get("quality_score", 0.986)),
                    )
                if kind in {"energy", "combined"} and any(key in row for key in ("pv_power_kw", "wind_power_kw", "load_kw")):
                    conn.execute(
                        "INSERT OR REPLACE INTO energy_forecasts(run_id,point_id,issued_at_utc,target_time_utc,horizon,pv_power_kw,wind_power_kw,load_kw,model_name,quality_score) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (run_id, point_id, issued, row["timestamp"], horizon, row.get("pv_power_kw"), row.get("wind_power_kw"), row.get("load_kw"), model_name, row.get("quality_score", 0.986)),
                    )
        self.finish_run(run_id, "success", len(rows))
        return len(rows)

    def get_forecast_rows(self, point_id: str, horizon: str, limit: int) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT timestamp,wind_speed_mps,temperature_c,solar_radiation_wm2,
                pv_power_kw,wind_power_kw,load_kw,source FROM (
                    SELECT w.target_time_utc AS timestamp,
                    w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
                    e.pv_power_kw,e.wind_power_kw,e.load_kw,
                    COALESCE(w.source,'calculated') AS source
                    FROM weather_forecasts w LEFT JOIN energy_forecasts e
                    ON w.run_id=e.run_id AND w.target_time_utc=e.target_time_utc
                    WHERE w.point_id=? AND w.horizon=?
                    UNION ALL
                    SELECT e.target_time_utc AS timestamp,
                    w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
                    e.pv_power_kw,e.wind_power_kw,e.load_kw,
                    'calculated' AS source
                    FROM energy_forecasts e LEFT JOIN weather_forecasts w
                    ON w.run_id=e.run_id AND w.target_time_utc=e.target_time_utc
                    WHERE e.point_id=? AND e.horizon=? AND w.run_id IS NULL
                ) ORDER BY timestamp LIMIT ?""",
                (point_id, horizon, point_id, horizon, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def health(self) -> dict[str, object]:
        with self.connection() as conn:
            tables = {}
            for table in ("forecast_points", "forecast_runs", "weather_observations", "weather_forecasts", "energy_measurements", "energy_forecasts", "model_registry", "training_runs", "model_updates", "forecast_schedules", "forecast_evaluations"):
                tables[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"status": "ok", "path": str(self.path), "schema_version": SCHEMA_VERSION, "tables": tables}


def seed_from_demo_csv(db: ForecastDB, csv_path: str | Path, point_id: str = "P001") -> int:
    """Load the bundled demo forecast into SQLite once, idempotently."""
    import csv

    csv_path = Path(csv_path)
    with db.connection() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM energy_forecasts").fetchone()[0]
    if existing or not csv_path.exists():
        return existing
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = row.get("") or row.get("timestamp")
            if not timestamp:
                continue
            rows.append({"timestamp": timestamp, **{key: float(row[key]) for key in ("pv_power_kw", "wind_power_kw", "load_kw", "wind_speed_mps", "temperature_c", "solar_radiation_wm2")}})
    run_id = "SEED-WEEK-AHEAD"
    try:
        db.create_run(run_id, point_id, "combined", "week_ahead", "seed")
    except sqlite3.IntegrityError:
        pass
    return db.save_forecast_rows(run_id, point_id, "week_ahead", rows, "combined", "Ridge 15分钟基线", "seed")
