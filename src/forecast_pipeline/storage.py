"""SQLite persistence for the forecast UI and calculation backend.

The database is deliberately kept behind a small repository class. Forecast
models can write pandas records through the bulk methods, while the HMI reads
stable JSON-shaped dictionaries without knowing the SQL schema.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 6
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DEFAULT_DB_PATH = Path(os.environ.get("FORECAST_DB_PATH", r"D:\codex\sqldb\realtime\forecast\forecast.db"))


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
CREATE TABLE IF NOT EXISTS forecast_objects (
    object_id TEXT PRIMARY KEY,
    point_id TEXT NOT NULL,
    object_name TEXT NOT NULL,
    object_type TEXT NOT NULL CHECK(object_type IN ('grid', 'plant', 'unit', 'point')),
    technology TEXT NOT NULL DEFAULT 'mixed' CHECK(technology IN ('wind', 'pv', 'mixed', 'load', 'unknown')),
    parent_object_id TEXT,
    capacity_kw REAL NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id),
    FOREIGN KEY(parent_object_id) REFERENCES forecast_objects(object_id)
);
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY,
    point_id TEXT,
    object_id TEXT,
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
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
    pv_available_power_kw REAL,
    wind_available_power_kw REAL,
    pv_curtailment_kw REAL,
    wind_curtailment_kw REAL,
    curtailment_flag INTEGER NOT NULL DEFAULT 0 CHECK(curtailment_flag IN (0, 1)),
    curtailment_reason TEXT,
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
    pv_available_power_kw REAL,
    wind_available_power_kw REAL,
    prediction_basis TEXT NOT NULL DEFAULT 'available_power',
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
CREATE TABLE IF NOT EXISTS nwp_forecasts (
    source_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    issued_at_utc TEXT NOT NULL,
    target_time_utc TEXT NOT NULL,
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
    wind_speed_mps REAL,
    temperature_c REAL,
    solar_radiation_wm2 REAL,
    quality_score REAL,
    received_at TEXT NOT NULL,
    PRIMARY KEY(source_id, point_id, issued_at_utc, target_time_utc),
    FOREIGN KEY(source_id) REFERENCES nwp_sources(source_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE TABLE IF NOT EXISTS forecast_tasks (
    task_id TEXT PRIMARY KEY,
    point_id TEXT NOT NULL,
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
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
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
    point_id TEXT,
    object_id TEXT,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('training', 'active', 'retired', 'failed')),
    artifact_path TEXT,
    hyperparams_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    trained_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
    ,FOREIGN KEY(object_id) REFERENCES forecast_objects(object_id)
);
CREATE TABLE IF NOT EXISTS training_runs (
    training_id TEXT PRIMARY KEY,
    model_id TEXT,
    point_id TEXT,
    object_id TEXT,
    target_kind TEXT NOT NULL CHECK(target_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
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
    ,FOREIGN KEY(object_id) REFERENCES forecast_objects(object_id)
);
CREATE TABLE IF NOT EXISTS model_updates (
    update_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    point_id TEXT,
    object_id TEXT,
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
    update_strategy TEXT NOT NULL CHECK(update_strategy IN ('rolling_window', 'incremental', 'retrain')),
    window_size INTEGER,
    sample_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('accepted', 'success', 'failed')),
    created_at TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
    ,FOREIGN KEY(object_id) REFERENCES forecast_objects(object_id)
);
CREATE TABLE IF NOT EXISTS forecast_schedules (
    schedule_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    point_ids_json TEXT NOT NULL DEFAULT '[]',
    object_ids_json TEXT NOT NULL DEFAULT '[]',
    forecast_kind TEXT NOT NULL CHECK(forecast_kind IN ('weather', 'energy', 'combined')),
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
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
    horizon TEXT NOT NULL CHECK(horizon IN ('hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead')),
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
CREATE TABLE IF NOT EXISTS curtailment_events (
    event_id TEXT PRIMARY KEY,
    point_id TEXT NOT NULL,
    object_id TEXT,
    start_time_utc TEXT NOT NULL,
    end_time_utc TEXT NOT NULL,
    technology TEXT NOT NULL CHECK(technology IN ('wind', 'pv', 'mixed', 'unknown')),
    available_power_kw REAL,
    measured_power_kw REAL,
    curtailed_power_kw REAL,
    reason TEXT,
    source TEXT NOT NULL DEFAULT 'local',
    created_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES forecast_points(point_id)
);
CREATE INDEX IF NOT EXISTS idx_weather_obs_point_time ON weather_observations(point_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_weather_fc_point_target ON weather_forecasts(point_id, target_time_utc);
CREATE INDEX IF NOT EXISTS idx_nwp_fc_point_target ON nwp_forecasts(point_id, target_time_utc);
CREATE INDEX IF NOT EXISTS idx_energy_meas_point_time ON energy_measurements(point_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_energy_fc_point_target ON energy_forecasts(point_id, target_time_utc);
CREATE INDEX IF NOT EXISTS idx_runs_point_started ON forecast_runs(point_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_point_horizon ON model_registry(point_id, horizon, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_started ON training_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_next_run ON forecast_schedules(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_eval_run ON forecast_evaluations(run_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_curtailment_point_time ON curtailment_events(point_id, start_time_utc);
"""


class ForecastDB:
    """Repository for the forecast SQLite database."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH

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
            version_row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
            current_version = int(version_row["value"]) if version_row else 0
            if current_version < 3:
                self._migrate_horizon_checks(conn)
                conn.executescript(SCHEMA_SQL)
            if current_version < 5:
                self._migrate_v4(conn)
                conn.executescript(SCHEMA_SQL)
            if current_version < 6:
                self._migrate_v6(conn)
                conn.executescript(SCHEMA_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_object_started ON forecast_runs(object_id, started_at DESC)")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _migrate_horizon_checks(conn: sqlite3.Connection) -> None:
        """Rebuild legacy tables whose SQLite CHECK omitted ``hour_ahead``."""
        tables = (
            "forecast_runs", "weather_forecasts", "energy_forecasts", "forecast_tasks",
            "model_registry", "training_runs", "model_updates", "forecast_schedules",
            "forecast_evaluations",
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            for table in tables:
                row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if not row or not row["sql"] or "hour_ahead" in row["sql"]:
                    continue
                temp = f"{table}__v3"
                create_sql = row["sql"].replace(f"CREATE TABLE {table}", f"CREATE TABLE {temp}", 1)
                create_sql = create_sql.replace("'ultrashort', 'day_ahead', 'week_ahead'", "'hour_ahead', 'ultrashort', 'day_ahead', 'week_ahead'")
                conn.execute(create_sql)
                columns = [info["name"] for info in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                names = ",".join(columns)
                conn.execute(f"INSERT INTO {temp} ({names}) SELECT {names} FROM {table}")
                conn.execute(f"DROP TABLE {table}")
                conn.execute(f"ALTER TABLE {temp} RENAME TO {table}")
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _migrate_v4(conn: sqlite3.Connection) -> None:
        """Add object scope and curtailment provenance without rewriting data."""
        columns = {
            "forecast_runs": {"object_id": "TEXT"},
            "forecast_schedules": {"object_ids_json": "TEXT NOT NULL DEFAULT '[]'"},
            "energy_measurements": {
                "pv_available_power_kw": "REAL",
                "wind_available_power_kw": "REAL",
                "pv_curtailment_kw": "REAL",
                "wind_curtailment_kw": "REAL",
                "curtailment_flag": "INTEGER NOT NULL DEFAULT 0",
                "curtailment_reason": "TEXT",
            },
            "energy_forecasts": {
                "pv_available_power_kw": "REAL",
                "wind_available_power_kw": "REAL",
                "prediction_basis": "TEXT NOT NULL DEFAULT 'available_power'",
            },
        }
        for table, additions in columns.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in additions.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_objects (
                object_id TEXT PRIMARY KEY,
                point_id TEXT NOT NULL,
                object_name TEXT NOT NULL,
                object_type TEXT NOT NULL,
                technology TEXT NOT NULL DEFAULT 'mixed',
                parent_object_id TEXT,
                capacity_kw REAL NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS curtailment_events (
                event_id TEXT PRIMARY KEY,
                point_id TEXT NOT NULL,
                object_id TEXT,
                start_time_utc TEXT NOT NULL,
                end_time_utc TEXT NOT NULL,
                technology TEXT NOT NULL,
                available_power_kw REAL,
                measured_power_kw REAL,
                curtailed_power_kw REAL,
                reason TEXT,
                source TEXT NOT NULL DEFAULT 'local',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute("UPDATE forecast_runs SET object_id=point_id WHERE object_id IS NULL")

    @staticmethod
    def _migrate_v6(conn: sqlite3.Connection) -> None:
        """Persist the selected forecast object on model and training records."""
        additions = {
            "model_registry": {"object_id": "TEXT"},
            "training_runs": {"object_id": "TEXT"},
            "model_updates": {"object_id": "TEXT"},
        }
        for table, columns in additions.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_models_object_horizon ON model_registry(object_id, horizon, updated_at DESC)")

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
                    ON CONFLICT(point_id) DO NOTHING""",
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
                    ON CONFLICT(source_id) DO NOTHING""",
                    (source_id, name, provider, model, endpoint, refresh, json.dumps(variables, ensure_ascii=False), now),
                )
            objects = [
                ("OBJ-GRID", "P001", "全网风光汇总", "grid", "mixed", None, 200.0),
                ("OBJ-WIND-P002", "P002", "北侧风机群", "plant", "wind", "OBJ-GRID", 420.0),
                ("OBJ-WIND-P002-U01", "P002", "北侧风机 WT-001", "unit", "wind", "OBJ-WIND-P002", 420.0),
                ("OBJ-PV-P001-U01", "P001", "南岛光伏 PV-001", "unit", "pv", "OBJ-GRID", 120.0),
                ("OBJ-LOAD-P003", "P003", "园区负荷汇总", "grid", "load", None, 0.0),
            ]
            for object_id, point_id, name, object_type, technology, parent_id, capacity in objects:
                conn.execute(
                    """INSERT INTO forecast_objects
                    (object_id,point_id,object_name,object_type,technology,parent_object_id,capacity_kw,enabled,metadata_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,1,'{}',?,?)
                    ON CONFLICT(object_id) DO UPDATE SET point_id=excluded.point_id,object_name=excluded.object_name,
                    object_type=excluded.object_type,technology=excluded.technology,parent_object_id=excluded.parent_object_id,
                    capacity_kw=excluded.capacity_kw,updated_at=excluded.updated_at""",
                    (object_id, point_id, name, object_type, technology, parent_id, capacity, now, now),
                )
            conn.execute(
                """INSERT INTO forecast_schedules
                (schedule_id,name,point_ids_json,forecast_kind,horizon,enabled,cadence_minutes,next_run_at,updated_at)
                VALUES(?,?,?,?,?,1,?,?,?)
                ON CONFLICT(schedule_id) DO NOTHING""",
                ("SCH-DEFAULT-DAY-AHEAD", "默认日前预测", json.dumps(["P001", "P002", "P003"]), "combined", "day_ahead", 1440, now, now),
            )
            for point_id in ("P001", "P002", "P003"):
                for horizon, schedule_time in (("hour_ahead", "每小时"), ("ultrashort", "每15分钟"), ("day_ahead", "06:00"), ("week_ahead", "06:30")):
                    conn.execute(
                        """INSERT INTO forecast_tasks(task_id,point_id,forecast_kind,horizon,enabled,schedule_time,timezone,updated_at)
                        VALUES(?,?,?,?,1,?,?,?) ON CONFLICT(task_id) DO NOTHING""",
                        (f"TASK-{point_id}-{horizon}", point_id, "combined", horizon, schedule_time, "Pacific/Auckland", now),
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
            rows = conn.execute(
                """SELECT p.*,
                    (SELECT MAX(started_at) FROM forecast_runs r WHERE r.point_id=p.point_id) AS last_run_at,
                    (SELECT AVG(CASE WHEN quality_code='valid' THEN 1.0 ELSE 0.0 END)
                       FROM weather_observations w WHERE w.point_id=p.point_id) AS weather_quality,
                    (SELECT AVG(CASE WHEN quality_code='valid' THEN 1.0 ELSE 0.0 END)
                       FROM energy_measurements e WHERE e.point_id=p.point_id) AS energy_quality
                   FROM forecast_points p ORDER BY p.point_id"""
            ).fetchall()
        result = []
        for row in rows:
            quality_values = [value for value in (row["weather_quality"], row["energy_quality"]) if value is not None]
            quality = f"{sum(quality_values) / len(quality_values) * 100:.1f}%" if quality_values else "待检查"
            result.append(
                {
                    "id": row["point_id"], "name": row["name"], "type": row["point_type"],
                    "lat": row["latitude"], "lon": row["longitude"], "pv": row["pv_capacity_kw"],
                    "wind": row["wind_capacity_kw"], "objects": json.loads(row["objects_json"] or "[]"),
                    "enabled": bool(row["enabled"]), "model": row["model_name"], "source": row["weather_source"],
                    "quality": quality, "lastRun": row["last_run_at"] or "未运行",
                }
            )
        return result

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

    def list_objects(self, object_id: str | None = None, enabled_only: bool = False) -> list[dict[str, object]]:
        clauses = []
        params: list[object] = []
        if object_id:
            clauses.append("o.object_id=?")
            params.append(object_id)
        if enabled_only:
            clauses.append("o.enabled=1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""SELECT o.*,p.name AS point_name,p.point_type,p.latitude,p.longitude
                    FROM forecast_objects o JOIN forecast_points p ON p.point_id=o.point_id{where}
                    ORDER BY o.object_type,o.object_id""",
                params,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["enabled"] = bool(item["enabled"])
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def upsert_object(self, obj: Mapping[str, object]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO forecast_objects
                (object_id,point_id,object_name,object_type,technology,parent_object_id,capacity_kw,enabled,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(object_id) DO UPDATE SET point_id=excluded.point_id,object_name=excluded.object_name,
                object_type=excluded.object_type,technology=excluded.technology,parent_object_id=excluded.parent_object_id,
                capacity_kw=excluded.capacity_kw,enabled=excluded.enabled,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
                (
                    obj["object_id"], obj["point_id"], obj.get("object_name", obj["object_id"]),
                    obj.get("object_type", "point"), obj.get("technology", "mixed"), obj.get("parent_object_id"),
                    float(obj.get("capacity_kw", 0) or 0), int(bool(obj.get("enabled", True))),
                    json.dumps(obj.get("metadata", {}), ensure_ascii=False), obj.get("created_at", now), now,
                ),
            )

    def resolve_object(self, object_id: str | None) -> dict[str, object] | None:
        if not object_id:
            return None
        objects = self.list_objects(object_id)
        return objects[0] if objects else None

    def create_run(self, run_id: str, point_id: str | None, kind: str, horizon: str, trigger: str = "manual", scheduled_at: str | None = None, object_id: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO forecast_runs(run_id,point_id,object_id,forecast_kind,horizon,trigger_type,scheduled_at,started_at,status) VALUES(?,?,?,?,?,?,?,?,'accepted')",
                (run_id, point_id, object_id or point_id, kind, horizon, trigger, scheduled_at, utc_now()),
            )

    def finish_run(self, run_id: str, status: str, rows_written: int = 0, error_message: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE forecast_runs SET status=?, finished_at=?, rows_written=?, error_message=? WHERE run_id=?", (status, utc_now(), rows_written, error_message, run_id))

    def list_runs(
        self,
        limit: int = 50,
        status: str | None = None,
        point_id: str | None = None,
        horizon: str | None = None,
        object_id: str | None = None,
    ) -> list[dict[str, object]]:
        with self.connection() as conn:
            clauses = []
            params: list[object] = []
            if status:
                clauses.append("status=?"); params.append(status)
            if point_id:
                clauses.append("point_id=?"); params.append(point_id)
            if horizon:
                clauses.append("horizon=?"); params.append(horizon)
            if object_id:
                clauses.append("object_id=?"); params.append(object_id)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(f"SELECT * FROM forecast_runs{where} ORDER BY started_at DESC LIMIT ?", (*params, min(max(int(limit), 1), 200))).fetchall()
        return [dict(r) for r in rows]

    def register_model(self, model: Mapping[str, object]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO model_registry
                (model_id,model_name,model_type,target_kind,horizon,point_id,object_id,version,status,artifact_path,hyperparams_json,metrics_json,trained_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id) DO UPDATE SET model_name=excluded.model_name,model_type=excluded.model_type,
                target_kind=excluded.target_kind,horizon=excluded.horizon,point_id=excluded.point_id,object_id=excluded.object_id,version=excluded.version,
                status=excluded.status,artifact_path=excluded.artifact_path,hyperparams_json=excluded.hyperparams_json,
                metrics_json=excluded.metrics_json,updated_at=excluded.updated_at""",
                (model["model_id"], model.get("model_name", "混合特征 Ridge 15分钟"), model.get("model_type", "ridge"),
                 model.get("target_kind", "combined"), model.get("horizon", "day_ahead"), model.get("point_id"), model.get("object_id"),
                 model.get("version", "v1"), model.get("status", "active"), model.get("artifact_path"),
                 json.dumps(model.get("hyperparams", {}), ensure_ascii=False), json.dumps(model.get("metrics", {}), ensure_ascii=False),
                 model.get("trained_at", now), now),
            )

    def list_models(self, point_id: str | None = None, horizon: str | None = None, object_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 500)
        clauses, params = [], []
        if point_id:
            clauses.append("point_id=?"); params.append(point_id)
        if horizon:
            clauses.append("horizon=?"); params.append(horizon)
        if object_id:
            clauses.append("object_id=?"); params.append(object_id)
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

    def create_training_run(self, training_id: str, point_id: str | None, target_kind: str, horizon: str, mode: str, window_size: int | None = None, model_id: str | None = None, object_id: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO training_runs(training_id,model_id,point_id,object_id,target_kind,horizon,training_mode,window_size,started_at,status) VALUES(?,?,?,?,?,?,?,?,?, 'running')",
                (training_id, model_id, point_id, object_id, target_kind, horizon, mode, window_size, utc_now()),
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
                """INSERT INTO model_updates(update_id,model_id,point_id,object_id,horizon,update_strategy,window_size,sample_count,status,created_at,details_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (update["update_id"], update["model_id"], update.get("point_id"), update.get("object_id"), update["horizon"], update.get("update_strategy", "rolling_window"), update.get("window_size"), update.get("sample_count", 0), update.get("status", "success"), update.get("created_at", utc_now()), json.dumps(update.get("details", {}), ensure_ascii=False)),
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
                """INSERT INTO forecast_schedules(schedule_id,name,point_ids_json,object_ids_json,forecast_kind,horizon,enabled,cadence_minutes,next_run_at,last_run_at,last_run_id,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(schedule_id) DO UPDATE SET name=excluded.name,point_ids_json=excluded.point_ids_json,
                object_ids_json=excluded.object_ids_json,
                forecast_kind=excluded.forecast_kind,horizon=excluded.horizon,enabled=excluded.enabled,cadence_minutes=excluded.cadence_minutes,
                next_run_at=excluded.next_run_at,updated_at=excluded.updated_at""",
                (schedule["schedule_id"], schedule.get("name", "预测定时任务"), json.dumps(point_ids, ensure_ascii=False), json.dumps(schedule.get("object_ids", []), ensure_ascii=False), schedule.get("forecast_kind", "combined"), schedule.get("horizon", "day_ahead"), int(bool(schedule.get("enabled", True))), int(schedule.get("cadence_minutes", 1440)), schedule.get("next_run_at"), schedule.get("last_run_at"), schedule.get("last_run_id"), now),
            )

    def list_schedules(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM forecast_schedules ORDER BY name").fetchall()
        result = []
        for row in rows:
            item = dict(row); item["point_ids"] = json.loads(item.pop("point_ids_json") or "[]"); item["object_ids"] = json.loads(item.pop("object_ids_json", "[]") or "[]"); item["enabled"] = bool(item["enabled"]); result.append(item)
        return result

    def list_nwp_sources(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM nwp_sources ORDER BY source_id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["variables"] = json.loads(item.pop("variables_json") or "[]")
            item["enabled"] = bool(item["enabled"])
            result.append(item)
        return result

    def upsert_nwp_source(self, source: Mapping[str, object]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO nwp_sources(source_id,name,provider,model_name,endpoint,refresh_minutes,variables_json,enabled,status,last_sync_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET name=excluded.name,provider=excluded.provider,model_name=excluded.model_name,
                endpoint=excluded.endpoint,refresh_minutes=excluded.refresh_minutes,variables_json=excluded.variables_json,
                enabled=excluded.enabled,status=excluded.status,last_sync_at=excluded.last_sync_at,updated_at=excluded.updated_at""",
                (source["source_id"], source.get("name", source["source_id"]), source.get("provider", "custom"), source.get("model_name", "custom"), source.get("endpoint"), int(source.get("refresh_minutes", 60)), json.dumps(source.get("variables", []), ensure_ascii=False), int(bool(source.get("enabled", True))), source.get("status", "configured"), source.get("last_sync_at"), now),
            )

    def save_nwp_forecasts(self, rows: Iterable[Mapping[str, object]]) -> int:
        count = 0
        with self.connection() as conn:
            for row in rows:
                conn.execute(
                    """INSERT OR REPLACE INTO nwp_forecasts
                    (source_id,point_id,issued_at_utc,target_time_utc,horizon,wind_speed_mps,temperature_c,solar_radiation_wm2,quality_score,received_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row["source_id"], row["point_id"], row["issued_at_utc"], row["target_time_utc"],
                        row.get("horizon", "day_ahead"), row.get("wind_speed_mps"), row.get("temperature_c"),
                        row.get("solar_radiation_wm2"), row.get("quality_score", 0.9), row.get("received_at", utc_now()),
                    ),
                )
                count += 1
        return count

    def list_nwp_forecasts(self, point_id: str | None = None, horizon: str | None = None, limit: int | str = 672) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 5000)
        clauses = []
        params: list[object] = []
        if point_id:
            clauses.append("point_id=?"); params.append(point_id)
        if horizon:
            clauses.append("horizon=?"); params.append(horizon)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(f"SELECT * FROM nwp_forecasts{where} ORDER BY target_time_utc LIMIT ?", (*params, limit)).fetchall()
        return [dict(row) for row in rows]

    def list_tasks(self, point_id: str | None = None) -> list[dict[str, object]]:
        with self.connection() as conn:
            if point_id:
                rows = conn.execute("SELECT * FROM forecast_tasks WHERE point_id=? ORDER BY task_id", (point_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM forecast_tasks ORDER BY task_id").fetchall()
        return [dict(row) | {"enabled": bool(row["enabled"])} for row in rows]

    def get_settings(self) -> dict[str, object]:
        with self.connection() as conn:
            rows = conn.execute("SELECT key,value FROM schema_meta WHERE key LIKE 'setting.%'").fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"][8:]] = json.loads(row["value"])
            except json.JSONDecodeError:
                result[row["key"][8:]] = row["value"]
        return result

    def save_settings(self, settings: Mapping[str, object]) -> None:
        with self.connection() as conn:
            for key, value in settings.items():
                conn.execute("INSERT INTO schema_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (f"setting.{key}", json.dumps(value, ensure_ascii=False)))

    def upsert_task(self, task: Mapping[str, object]) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO forecast_tasks(task_id,point_id,forecast_kind,horizon,enabled,schedule_time,timezone,updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET point_id=excluded.point_id,forecast_kind=excluded.forecast_kind,
                horizon=excluded.horizon,enabled=excluded.enabled,schedule_time=excluded.schedule_time,
                timezone=excluded.timezone,updated_at=excluded.updated_at""",
                (task["task_id"], task["point_id"], task.get("forecast_kind", "combined"), task.get("horizon", "day_ahead"), int(bool(task.get("enabled", True))), task.get("schedule_time"), task.get("timezone", "Pacific/Auckland"), utc_now()),
            )

    def mark_schedule_run(self, schedule_id: str, run_id: str, next_run_at: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE forecast_schedules SET last_run_at=?,last_run_id=?,next_run_at=?,updated_at=? WHERE schedule_id=?", (utc_now(), run_id, next_run_at, utc_now(), schedule_id))

    def save_evaluation(self, evaluation: Mapping[str, object]) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO forecast_evaluations(evaluation_id,run_id,point_id,forecast_kind,horizon,window_start,window_end,sample_count,status,mae,rmse,bias,nmae,coverage,metrics_json,evaluated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (evaluation["evaluation_id"], evaluation["run_id"], evaluation.get("point_id"), evaluation["forecast_kind"], evaluation["horizon"], evaluation.get("window_start"), evaluation.get("window_end"), evaluation.get("sample_count", 0), evaluation.get("status", "pending"), evaluation.get("mae"), evaluation.get("rmse"), evaluation.get("bias"), evaluation.get("nmae"), evaluation.get("coverage"), json.dumps(evaluation.get("metrics", {}), ensure_ascii=False), evaluation.get("evaluated_at", utc_now())),
            )

    def list_evaluations(self, run_id: str | None = None, point_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connection() as conn:
            clauses = []
            params: list[object] = []
            if run_id:
                clauses.append("run_id=?"); params.append(run_id)
            if point_id:
                clauses.append("point_id=?"); params.append(point_id)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"""SELECT * FROM (
                    SELECT e.*, ROW_NUMBER() OVER (PARTITION BY e.run_id ORDER BY e.evaluated_at DESC) AS rn
                    FROM forecast_evaluations e{where}
                ) WHERE rn=1 ORDER BY evaluated_at DESC LIMIT ?""",
                (*params, min(max(int(limit), 1), 500)),
            ).fetchall()
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
                pv_actual = row.get("pv_power_kw")
                wind_actual = row.get("wind_power_kw")
                pv_available = row.get("pv_available_power_kw", row.get("available_pv_power_kw"))
                wind_available = row.get("wind_available_power_kw", row.get("available_wind_power_kw"))
                pv_loss = row.get("pv_curtailment_kw")
                wind_loss = row.get("wind_curtailment_kw")
                if pv_loss is None and pv_available is not None and pv_actual is not None:
                    pv_loss = max(float(pv_available) - float(pv_actual), 0.0)
                if wind_loss is None and wind_available is not None and wind_actual is not None:
                    wind_loss = max(float(wind_available) - float(wind_actual), 0.0)
                curtailment_flag = bool(row.get("curtailment_flag", False) or float(pv_loss or 0) > 1e-6 or float(wind_loss or 0) > 1e-6)
                conn.execute(
                    """INSERT INTO energy_measurements
                    (point_id,timestamp_utc,pv_power_kw,wind_power_kw,load_kw,pv_available_power_kw,wind_available_power_kw,
                     pv_curtailment_kw,wind_curtailment_kw,curtailment_flag,curtailment_reason,source,quality_code,is_realtime,received_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(point_id,timestamp_utc,source) DO UPDATE SET pv_power_kw=excluded.pv_power_kw,
                    wind_power_kw=excluded.wind_power_kw,load_kw=excluded.load_kw,pv_available_power_kw=excluded.pv_available_power_kw,
                    wind_available_power_kw=excluded.wind_available_power_kw,pv_curtailment_kw=excluded.pv_curtailment_kw,
                    wind_curtailment_kw=excluded.wind_curtailment_kw,curtailment_flag=excluded.curtailment_flag,
                    curtailment_reason=excluded.curtailment_reason,quality_code=excluded.quality_code,is_realtime=excluded.is_realtime,received_at=excluded.received_at""",
                    (row["point_id"], row["timestamp_utc"], pv_actual, wind_actual, row.get("load_kw"), pv_available,
                     wind_available, pv_loss, wind_loss, int(curtailment_flag), row.get("curtailment_reason"),
                     row.get("source", "local"), row.get("quality_code", "valid"), int(bool(row.get("is_realtime", False))), row.get("received_at", utc_now())),
                )
                count += 1
        return count

    def save_curtailment_events(self, rows: Iterable[Mapping[str, object]]) -> int:
        count = 0
        with self.connection() as conn:
            for row in rows:
                conn.execute(
                    """INSERT OR REPLACE INTO curtailment_events
                    (event_id,point_id,object_id,start_time_utc,end_time_utc,technology,available_power_kw,measured_power_kw,curtailed_power_kw,reason,source,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["event_id"], row["point_id"], row.get("object_id"), row["start_time_utc"], row["end_time_utc"],
                     row.get("technology", "unknown"), row.get("available_power_kw"), row.get("measured_power_kw"),
                     row.get("curtailed_power_kw"), row.get("reason"), row.get("source", "local"), row.get("created_at", utc_now())),
                )
                count += 1
        return count

    def list_curtailment_events(self, point_id: str | None = None, limit: int | str = 100) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 2000)
        with self.connection() as conn:
            if point_id:
                rows = conn.execute("SELECT * FROM curtailment_events WHERE point_id=? ORDER BY start_time_utc DESC LIMIT ?", (point_id, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM curtailment_events ORDER BY start_time_utc DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

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
                        """INSERT OR REPLACE INTO energy_forecasts
                        (run_id,point_id,issued_at_utc,target_time_utc,horizon,pv_power_kw,wind_power_kw,load_kw,
                         pv_available_power_kw,wind_available_power_kw,prediction_basis,model_name,quality_score)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, point_id, issued, row["timestamp"], horizon, row.get("pv_power_kw"), row.get("wind_power_kw"), row.get("load_kw"),
                         row.get("pv_available_power_kw", row.get("pv_power_kw")), row.get("wind_available_power_kw", row.get("wind_power_kw")),
                         row.get("prediction_basis", "available_power"), model_name, row.get("quality_score", 0.986)),
                    )
        self.finish_run(run_id, "success", len(rows))
        return len(rows)

    def get_forecast_rows(self, point_id: str, horizon: str, limit: int, run_id: str | None = None, object_id: str | None = None) -> list[dict[str, object]]:
        with self.connection() as conn:
            latest_filter = "l.run_id=?" if run_id else "l.rn=1"
            latest_params: tuple[object, ...] = (run_id,) if run_id else ()
            rows = conn.execute(
                f"""WITH latest AS (
                    SELECT run_id, object_id, ROW_NUMBER() OVER (ORDER BY COALESCE(finished_at, started_at) DESC, started_at DESC) AS rn
                    FROM forecast_runs WHERE point_id=? AND horizon=? AND status='success'
                    AND (? IS NULL OR object_id=?)
                )
                SELECT timestamp,wind_speed_mps,temperature_c,solar_radiation_wm2,
                pv_power_kw,wind_power_kw,load_kw,source FROM (
                    SELECT w.target_time_utc AS timestamp,
                    w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
                    e.pv_power_kw,e.wind_power_kw,e.load_kw,
                    COALESCE(w.source,'calculated') AS source
                    FROM weather_forecasts w LEFT JOIN energy_forecasts e
                    ON w.run_id=e.run_id AND w.target_time_utc=e.target_time_utc
                    JOIN latest l ON l.run_id=w.run_id
                    WHERE w.point_id=? AND w.horizon=? AND {latest_filter}
                    UNION ALL
                    SELECT e.target_time_utc AS timestamp,
                    w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
                    e.pv_power_kw,e.wind_power_kw,e.load_kw,
                    'calculated' AS source
                    FROM energy_forecasts e LEFT JOIN weather_forecasts w
                    ON w.run_id=e.run_id AND w.target_time_utc=e.target_time_utc
                    JOIN latest l ON l.run_id=e.run_id
                    WHERE e.point_id=? AND e.horizon=? AND w.run_id IS NULL AND {latest_filter}
                ) ORDER BY timestamp LIMIT ?""",
                (point_id, horizon, object_id, object_id, point_id, horizon, *latest_params, point_id, horizon, *latest_params, min(max(int(limit), 1), 2000)),
            ).fetchall()
        return [dict(r) for r in rows]

    def health(self) -> dict[str, object]:
        with self.connection() as conn:
            tables = {}
            for table in ("forecast_points", "forecast_objects", "forecast_runs", "weather_observations", "weather_forecasts", "energy_measurements", "energy_forecasts", "nwp_sources", "nwp_forecasts", "forecast_tasks", "model_registry", "training_runs", "model_updates", "forecast_schedules", "forecast_evaluations", "curtailment_events"):
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
