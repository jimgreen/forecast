"""Serve the Forecast HMI and the local forecast orchestration API.

The service keeps the browser, SQLite database and calculation workers on one
small local boundary. It supports manual runs, scheduled runs, offline model
training, rolling model updates and post-run evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DATA = ROOT / "demo_week_ahead_forecast.csv"
MODEL_DIR = ROOT / "data" / "models"
sys.path.insert(0, str(ROOT / "src"))

from forecast_pipeline.cache import PipelineCache  # noqa: E402
from forecast_pipeline.pipeline import IntegratedForecastPipeline  # noqa: E402
from forecast_pipeline.fusion import fuse_weather  # noqa: E402
from forecast_pipeline.providers import OpenMeteoProvider  # noqa: E402
from forecast_pipeline.schema import WEATHER_COLUMNS, make_future_index  # noqa: E402
from forecast_pipeline.storage import ForecastDB, seed_from_demo_csv  # noqa: E402

DB = ForecastDB()
DB.initialize()
DB.seed_defaults()
seed_from_demo_csv(DB, DATA)

HORIZONS = {"hour_ahead": 4, "ultrashort": 16, "day_ahead": 96, "week_ahead": 672}
KINDS = {"weather", "energy", "combined"}
TRIGGERS = {"manual", "scheduled", "api", "seed"}
WEATHER_TARGETS = ("wind_speed_mps", "temperature_c", "solar_radiation_wm2")
ENERGY_TARGETS = ("pv_power_kw", "wind_power_kw", "load_kw")


def horizon_count(name: str) -> int:
    return HORIZONS.get(name, 96)


def now_dt() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_dt().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{now_dt().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6].upper()}"


def parse_dt(value: object) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def load_demo_frame() -> pd.DataFrame:
    return _load_demo_frame_cached().copy()


@lru_cache(maxsize=1)
def _load_demo_frame_cached() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_observation_frame(point_id: str | None) -> pd.DataFrame:
    if not point_id:
        return pd.DataFrame()
    with DB.connection() as conn:
        rows = conn.execute(
            """SELECT COALESCE(w.timestamp_utc,e.timestamp_utc) AS timestamp,
            w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
            COALESCE(e.pv_available_power_kw,e.pv_power_kw) AS pv_power_kw,
            COALESCE(e.wind_available_power_kw,e.wind_power_kw) AS wind_power_kw,
            e.load_kw,e.pv_power_kw AS measured_pv_power_kw,e.wind_power_kw AS measured_wind_power_kw,
            e.pv_available_power_kw,e.wind_available_power_kw,e.pv_curtailment_kw,e.wind_curtailment_kw,
            e.curtailment_flag,e.curtailment_reason
            FROM weather_observations w LEFT JOIN energy_measurements e
            ON e.point_id=w.point_id AND e.timestamp_utc=w.timestamp_utc
            WHERE w.point_id=?
            UNION ALL
            SELECT e.timestamp_utc AS timestamp,
            w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
            COALESCE(e.pv_available_power_kw,e.pv_power_kw) AS pv_power_kw,
            COALESCE(e.wind_available_power_kw,e.wind_power_kw) AS wind_power_kw,
            e.load_kw,e.pv_power_kw AS measured_pv_power_kw,e.wind_power_kw AS measured_wind_power_kw,
            e.pv_available_power_kw,e.wind_available_power_kw,e.pv_curtailment_kw,e.wind_curtailment_kw,
            e.curtailment_flag,e.curtailment_reason
            FROM energy_measurements e LEFT JOIN weather_observations w
            ON w.point_id=e.point_id AND w.timestamp_utc=e.timestamp_utc
            WHERE e.point_id=? AND w.point_id IS NULL
            ORDER BY timestamp""",
            (point_id, point_id),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in (*WEATHER_TARGETS, *ENERGY_TARGETS, "measured_pv_power_kw", "measured_wind_power_kw", "pv_available_power_kw", "wind_available_power_kw", "pv_curtailment_kw", "wind_curtailment_kw"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").dropna(subset=[*WEATHER_TARGETS, *ENERGY_TARGETS]).reset_index(drop=True)


def load_weather_observation_frame(point_id: str | None) -> pd.DataFrame:
    """Read the local micro-weather history without requiring energy fields."""
    if not point_id:
        return pd.DataFrame()
    with DB.connection() as conn:
        rows = conn.execute(
            """SELECT timestamp_utc AS timestamp,wind_speed_mps,temperature_c,solar_radiation_wm2,source
               FROM weather_observations WHERE point_id=? ORDER BY timestamp_utc""",
            (point_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in WEATHER_TARGETS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").dropna(subset=list(WEATHER_TARGETS)).reset_index(drop=True)


def load_energy_observation_frame(point_id: str | None) -> pd.DataFrame:
    """Read power history without requiring a weather station row at each time."""
    if not point_id:
        return pd.DataFrame()
    with DB.connection() as conn:
        rows = conn.execute(
            """SELECT timestamp_utc AS timestamp,
               COALESCE(pv_available_power_kw,pv_power_kw) AS pv_power_kw,
               COALESCE(wind_available_power_kw,wind_power_kw) AS wind_power_kw,
               load_kw,pv_power_kw AS measured_pv_power_kw,wind_power_kw AS measured_wind_power_kw,
               pv_available_power_kw,wind_available_power_kw,pv_curtailment_kw,wind_curtailment_kw,
               curtailment_flag,curtailment_reason,source
               FROM energy_measurements WHERE point_id=? ORDER BY timestamp_utc""",
            (point_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in (*ENERGY_TARGETS, "measured_pv_power_kw", "measured_wind_power_kw", "pv_available_power_kw", "wind_available_power_kw", "pv_curtailment_kw", "wind_curtailment_kw"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").dropna(how="all", subset=list(ENERGY_TARGETS)).reset_index(drop=True)


def load_power_training_frame(point_id: str | None) -> pd.DataFrame:
    """Join real power history to weather history, with a deterministic weather fallback."""
    power = load_energy_observation_frame(point_id)
    if power.empty:
        return power
    weather = load_weather_observation_frame(point_id)
    if not weather.empty:
        result = power.merge(weather[["timestamp", *WEATHER_TARGETS]], on="timestamp", how="left")
    else:
        demo = load_demo_frame()[["timestamp", *WEATHER_TARGETS]].copy()
        demo["timestamp"] = pd.to_datetime(demo["timestamp"], utc=True)
        result = power.merge(demo, on="timestamp", how="left")
    for column in WEATHER_TARGETS:
        if result[column].isna().any():
            result[column] = result[column].interpolate(limit_direction="both")
            if result[column].isna().any():
                result[column] = result[column].fillna(float(load_demo_frame()[column].mean()))
    return result.dropna(subset=list(WEATHER_TARGETS)).reset_index(drop=True)


def point_record(point_id: str) -> dict[str, object] | None:
    return next((point for point in DB.list_points() if point["id"] == point_id), None)


def object_record(object_id: str | None, fallback_point_id: str | None = None) -> dict[str, object] | None:
    if object_id:
        return DB.resolve_object(object_id)
    if fallback_point_id:
        return next((item for item in DB.list_objects() if item["point_id"] == fallback_point_id), None)
    return None


def capacity_limits(point_id: str | None, object_id: str | None = None) -> dict[str, float]:
    """Resolve physical output limits for the selected grid, plant or unit."""
    point = point_record(point_id) if point_id else None
    obj = object_record(object_id, point_id)
    if not point:
        return {}
    pv = float(point.get("pv", 0) or 0)
    wind = float(point.get("wind", 0) or 0)
    if obj:
        technology = str(obj.get("technology") or "mixed")
        capacity = float(obj.get("capacity_kw", 0) or 0)
        if technology == "pv":
            pv, wind = capacity, 0.0
        elif technology == "wind":
            pv, wind = 0.0, capacity
        elif technology == "load":
            pv, wind = 0.0, 0.0
        elif capacity > 0 and technology == "mixed":
            total = max(pv + wind, 1.0)
            scale = capacity / total
            pv, wind = pv * scale, wind * scale
    limits = {}
    if pv > 0:
        limits["pv_power_kw"] = pv
    if wind > 0:
        limits["wind_power_kw"] = wind
    return limits


def prepare_power_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use available power when curtailment metadata proves measured output is clipped."""
    if frame.empty:
        return frame, {"curtailment_samples": 0, "adjusted_samples": 0}
    result = frame.copy()
    flag = result.get("curtailment_flag", pd.Series(False, index=result.index)).fillna(0).astype(bool)
    adjusted = 0
    adjusted_mask = pd.Series(False, index=result.index)
    for target, available in (("pv_power_kw", "pv_available_power_kw"), ("wind_power_kw", "wind_available_power_kw")):
        if available in result:
            available_values = pd.to_numeric(result[available], errors="coerce")
            use = available_values.notna() & (flag | (available_values > pd.to_numeric(result[target], errors="coerce")))
            result.loc[use, target] = available_values.loc[use]
            adjusted += int(use.sum())
            adjusted_mask = adjusted_mask | use
    result["curtailment_adjusted"] = flag | adjusted_mask
    return result, {"curtailment_samples": int(flag.sum()), "adjusted_samples": adjusted}


def energy_targets_for(frame: pd.DataFrame, minimum_samples: int = 32) -> tuple[str, ...]:
    available = tuple(target for target in ENERGY_TARGETS if target in frame and int(frame[target].notna().sum()) >= minimum_samples)
    return available or ENERGY_TARGETS


def curtailment_summary(point_id: str | None = None) -> dict[str, object]:
    with DB.connection() as conn:
        where = "WHERE point_id=?" if point_id else ""
        params = (point_id,) if point_id else ()
        row = conn.execute(
            f"""SELECT COUNT(*) AS samples,
                SUM(curtailment_flag) AS flagged_samples,
                SUM(COALESCE(pv_curtailment_kw,0)) AS pv_loss_kw,
                SUM(COALESCE(wind_curtailment_kw,0)) AS wind_loss_kw
                FROM energy_measurements {where}""", params,
        ).fetchone()
    samples = int(row["samples"] or 0)
    flagged = int(row["flagged_samples"] or 0)
    return {"samples": samples, "curtailment_samples": flagged, "curtailment_rate": round(flagged / samples * 100, 2) if samples else 0.0, "pv_loss_kw": float(row["pv_loss_kw"] or 0), "wind_loss_kw": float(row["wind_loss_kw"] or 0)}


def forecast_anchor(frame: pd.DataFrame) -> pd.Timestamp:
    """Return the latest usable observation, or the current 15-minute bucket."""
    if not frame.empty and "timestamp" in frame.columns:
        timestamp = pd.Timestamp(frame["timestamp"].max())
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
    return pd.Timestamp(now_dt()).floor("15min")


def nwp_weather_frame(point_id: str, horizon: str, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load cached NWP, then try Open-Meteo, and retain a deterministic fallback."""
    cached = DB.list_nwp_forecasts(point_id, horizon, max(horizon_count(horizon) * 4, 96))
    nwp = pd.DataFrame(cached)
    if not nwp.empty:
        nwp["timestamp"] = pd.to_datetime(nwp["target_time_utc"], utc=True)
        nwp = nwp.sort_values(["timestamp", "issued_at_utc"]).drop_duplicates("timestamp", keep="last")
        nwp = nwp.set_index("timestamp").reindex(future_index)
        nwp = nwp[list(WEATHER_TARGETS)].apply(pd.to_numeric, errors="coerce")
        if nwp.notna().all(axis=1).sum() >= max(1, len(future_index) // 2):
            return nwp.interpolate(method="time").ffill().bfill()

    point = point_record(point_id)
    provider_frame = pd.DataFrame()
    if point and point.get("lat") is not None and point.get("lon") is not None:
        try:
            provider_frame = OpenMeteoProvider(float(point["lat"]), float(point["lon"]), timeout=5).forecast(future_index[0], future_index[-1])
        except Exception as exc:  # network availability must not stop local forecasting
            print(f"[nwp] Open-Meteo unavailable for {point_id}: {exc}")
    if provider_frame.empty:
        template = pd.DataFrame(load_template_rows(horizon))
        provider_frame = template.set_index(pd.to_datetime(template["timestamp"], utc=True))[list(WEATHER_TARGETS)]
        provider_frame.index = future_index[: len(provider_frame)]
    provider_frame = provider_frame.reindex(future_index).interpolate(method="time").ffill().bfill()
    issued_at = now_iso()
    rows = [
        {
            "source_id": "openmeteo-icon", "point_id": point_id, "issued_at_utc": issued_at,
            "target_time_utc": index.isoformat(), "horizon": horizon,
            **{column: float(row[column]) for column in WEATHER_TARGETS}, "quality_score": 0.9,
        }
        for index, row in provider_frame.iterrows()
    ]
    try:
        DB.save_nwp_forecasts(rows)
    except Exception as exc:
        print(f"[nwp] cache write failed for {point_id}: {exc}")
    return provider_frame


def fused_future_weather(point_id: str, horizon: str, future_index: pd.DatetimeIndex, local: pd.DataFrame | None = None) -> pd.DataFrame:
    nwp = nwp_weather_frame(point_id, horizon, future_index)
    bias_window = int(DB.get_settings().get("biasWindow", 96) or 96)
    fused = fuse_weather(nwp, local=local, bias_window=bias_window)
    return fused.reindex(future_index)[list(WEATHER_TARGETS)].interpolate(method="time").ffill().bfill()


_PIPELINE_CACHE = PipelineCache(maxsize=16)


def cached_pipeline(history, targets, capacity_limits, cache_scope):
    return _PIPELINE_CACHE.get(history, targets, capacity_limits, cache_scope)


@lru_cache(maxsize=4)
def load_template_rows(horizon: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = row.get("") or row.get("timestamp")
            if not timestamp:
                continue
            item: dict[str, object] = {"timestamp": timestamp, "source": "NWP + 本地校正"}
            for key in (*ENERGY_TARGETS, *WEATHER_TARGETS):
                try:
                    item[key] = float(row.get(key, 0) or 0)
                except (TypeError, ValueError):
                    item[key] = 0.0
            rows.append(item)
            if len(rows) >= horizon_count(horizon):
                break
    return rows


def point_payload(payload: dict[str, object]) -> dict[str, object]:
    if "id" not in payload or not payload.get("name"):
        raise ValueError("point id and name are required")
    point = dict(payload)
    for key in ("lat", "lon", "pv", "wind"):
        if key in point and point[key] not in (None, ""):
            point[key] = float(point[key])
    if "objects" in point and not isinstance(point["objects"], list):
        raise ValueError("objects must be an array")
    return point


def payload_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload.get("rows", [payload])
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("rows must be an array of objects")
    return rows


def metric_values(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int | None]:
    values = pd.concat([actual, predicted], axis=1).dropna()
    if values.empty:
        return {"mae": None, "rmse": None, "bias": None, "nmae": None, "sample_count": 0}
    error = values.iloc[:, 1] - values.iloc[:, 0]
    mae = float(np.abs(error).mean())
    return {"mae": mae, "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()), "nmae": float(mae / max(float(np.abs(values.iloc[:, 0]).mean()), 1e-9) * 100), "sample_count": int(len(values))}


def train_model(point_id: str | None, target_kind: str, horizon: str, mode: str, window_size: int | None = None, object_id: str | None = None, strategy: str = "hybrid_ridge") -> dict[str, object]:
    if target_kind not in KINDS or horizon not in HORIZONS or mode not in {"offline", "rolling"}:
        raise ValueError("invalid training parameters")
    training_id = new_id("TRN")
    model_id = new_id("MDL")
    if object_id:
        obj = DB.resolve_object(object_id)
        if not obj:
            raise ValueError(f"unknown object_id: {object_id}")
        point_id = str(obj["point_id"])
    if strategy not in {"hybrid_ridge", "ridge_baseline"}:
        raise ValueError("invalid model strategy")
    limits = capacity_limits(point_id, object_id)
    DB.create_training_run(training_id, point_id, target_kind, horizon, mode, window_size, None, object_id)
    try:
        frame = load_weather_observation_frame(point_id) if target_kind == "weather" else load_power_training_frame(point_id)
        quality = {"curtailment_samples": 0, "adjusted_samples": 0}
        if target_kind in {"energy", "combined"}:
            frame, quality = prepare_power_training_frame(frame)
        if len(frame) < 64:
            frame = load_demo_frame()
        energy_targets = energy_targets_for(frame)
        training_targets = WEATHER_TARGETS if target_kind == "weather" else energy_targets if target_kind == "energy" else (*WEATHER_TARGETS, *energy_targets)
        if mode == "rolling":
            frame = frame.tail(min(int(window_size or 672), len(frame))).reset_index(drop=True)
        holdout = min(horizon_count(horizon), max(16, len(frame) // 5))
        train_frame = frame.iloc[:-holdout].copy()
        test_frame = frame.iloc[-holdout:].copy()
        if len(train_frame) < 40:
            raise ValueError("not enough historical rows for training")
        train_frame = train_frame.set_index("timestamp")
        test_frame = test_frame.set_index("timestamp")
        pipeline = IntegratedForecastPipeline().fit(train_frame, targets=training_targets, capacity_limits=limits)
        prediction = pipeline.predict(len(test_frame), future_weather=test_frame[list(WEATHER_TARGETS)])
        metrics: dict[str, object] = {target: metric_values(test_frame[target], prediction[target]) for target in training_targets}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {"model_id": model_id, "model_type": "hybrid_ridge15min" if strategy == "hybrid_ridge" else "ridge15min", "target_kind": target_kind, "horizon": horizon, "object_id": object_id, "capacity_limits": limits, "feature_count": pipeline.model.feature_count if pipeline.model else 0, "curtailment_policy": "use_available_power_when_flagged", "data_quality": quality, "coefficients": {key: value.tolist() for key, value in (pipeline.model.coefficients.items() if pipeline.model else [])}}
        artifact_path = MODEL_DIR / f"{model_id}.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        model = {"model_id": model_id, "model_name": "混合特征 Ridge 15分钟", "model_type": "hybrid_ridge15min" if strategy == "hybrid_ridge" else "ridge15min", "target_kind": target_kind, "horizon": horizon, "point_id": point_id, "object_id": object_id, "version": now_dt().strftime("v%Y%m%d%H%M%S"), "status": "active", "artifact_path": str(artifact_path), "hyperparams": {"alpha": 1.0, "mode": mode, "window_size": window_size, "strategy": strategy, "capacity_limits": limits, "curtailment_policy": "use_available_power_when_flagged", "data_quality": quality}, "metrics": {**metrics, "data_quality": quality}}
        DB.register_model(model)
        DB.finish_training_run(training_id, "success", model_id, len(train_frame), metrics)
        if mode == "rolling":
            DB.save_model_update({"update_id": new_id("UPD"), "model_id": model_id, "point_id": point_id, "object_id": object_id, "horizon": horizon, "update_strategy": "rolling_window", "window_size": len(frame), "sample_count": len(train_frame), "status": "success", "details": {"training_id": training_id, "metrics": metrics, "strategy": strategy}})
        return {"training_id": training_id, "model_id": model_id, "object_id": object_id, "status": "success", "sample_count": len(train_frame), "metrics": {**metrics, "data_quality": quality}, "artifact_path": str(artifact_path), "capacity_limits": limits, "curtailment_policy": "use_available_power_when_flagged"}
    except Exception as exc:
        DB.finish_training_run(training_id, "failed", model_id, 0, {}, str(exc))
        raise


def evaluate_run(run_id: str) -> dict[str, object]:
    with DB.connection() as conn:
        run = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
        previous = conn.execute("SELECT evaluation_id FROM forecast_evaluations WHERE run_id=? ORDER BY evaluated_at DESC LIMIT 1", (run_id,)).fetchone()
        if run is None:
            raise ValueError("run not found")
        forecast_kind = run["forecast_kind"]
        targets = WEATHER_TARGETS if forecast_kind == "weather" else ENERGY_TARGETS if forecast_kind == "energy" else (*WEATHER_TARGETS, *ENERGY_TARGETS)
        details: dict[str, object] = {}
        paired = 0
        for target in targets:
            table = "weather_forecasts" if target in WEATHER_TARGETS else "energy_forecasts"
            actual_table = "weather_observations" if target in WEATHER_TARGETS else "energy_measurements"
            forecast_rows = conn.execute(f"SELECT target_time_utc,{target} AS value FROM {table} WHERE run_id=?", (run_id,)).fetchall()
            if target == "pv_power_kw":
                actual_sql = "COALESCE(pv_available_power_kw,pv_power_kw)"
            elif target == "wind_power_kw":
                actual_sql = "COALESCE(wind_available_power_kw,wind_power_kw)"
            else:
                actual_sql = target
            flag_sql = "curtailment_flag" if target in ENERGY_TARGETS else "0"
            actual_rows = conn.execute(f"SELECT timestamp_utc,{actual_sql} AS value,{flag_sql} AS curtailment_flag FROM {actual_table} WHERE point_id=?", (run["point_id"],)).fetchall()
            actual = pd.Series({parse_dt(row["timestamp_utc"]): row["value"] for row in actual_rows}, dtype=float)
            predicted = pd.Series({parse_dt(row["target_time_utc"]): row["value"] for row in forecast_rows}, dtype=float)
            result = metric_values(actual, predicted)
            paired_times = set(actual.index).intersection(predicted.index)
            result["curtailment_samples"] = int(sum(bool(row["curtailment_flag"]) and parse_dt(row["timestamp_utc"]) in paired_times for row in actual_rows)) if target in {"pv_power_kw", "wind_power_kw"} else 0
            details[target] = result
            paired += int(result["sample_count"] or 0)
        available = [value for value in details.values() if value.get("sample_count", 0)]
        status = "success" if available else "pending"
        aggregate = {key: float(np.mean([float(value[key]) for value in available if value.get(key) is not None])) for key in ("mae", "rmse", "bias", "nmae")} if available else {key: None for key in ("mae", "rmse", "bias", "nmae")}
        evaluation = {"evaluation_id": previous["evaluation_id"] if previous else new_id("EVA"), "run_id": run_id, "point_id": run["point_id"], "forecast_kind": forecast_kind, "horizon": run["horizon"], "window_start": None, "window_end": None, "sample_count": paired, "status": status, **aggregate, "coverage": None, "metrics": details, "evaluation_basis": "available_power_for_curtailment"}
    DB.save_evaluation(evaluation)
    return evaluation


def execute_run(run_id: str, point_id: str, kind: str, horizon: str) -> dict[str, object]:
    try:
        with DB.connection() as conn:
            run_row = conn.execute("SELECT object_id FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
        object_id = str(run_row["object_id"]) if run_row and run_row["object_id"] else None
        limits = capacity_limits(point_id, object_id)
        local_weather = load_weather_observation_frame(point_id)
        if kind == "weather":
            history = local_weather if len(local_weather) >= 64 else load_demo_frame()[["timestamp", *WEATHER_TARGETS]]
            history_indexed = history.set_index("timestamp")
            future_index = make_future_index(forecast_anchor(local_weather), horizon)
            future_weather = fused_future_weather(point_id, horizon, future_index, local=local_weather)
            predictions = cached_pipeline(history_indexed, WEATHER_TARGETS, {}, f"weather:{point_id}").predict(horizon, future_weather=future_weather)
            rows = [{"timestamp": index.isoformat(), **{column: float(value) for column, value in row.items()}, "source": "NWP + 本地微气象校正", "quality_score": 0.92} for index, row in predictions.iterrows()]
        else:
            history = load_power_training_frame(point_id)
            source_history = history if len(history) >= 64 else load_demo_frame()
            source_history, quality = prepare_power_training_frame(source_history)
            history_indexed = source_history.set_index("timestamp")
            anchor_source = history if not history.empty else local_weather
            future_index = make_future_index(forecast_anchor(anchor_source), horizon)
            future_weather = fused_future_weather(point_id, horizon, future_index, local=local_weather)
            energy_targets = energy_targets_for(source_history)
            targets = energy_targets if kind == "energy" else (*WEATHER_TARGETS, *energy_targets)
            predictions = cached_pipeline(history_indexed, targets, limits, f"energy:{point_id}:{object_id}:{kind}").predict(horizon, future_weather=future_weather)
            rows = [{"timestamp": index.isoformat(), **{column: float(value) for column, value in row.items()}, "pv_available_power_kw": float(row.get("pv_power_kw", 0.0)), "wind_available_power_kw": float(row.get("wind_power_kw", 0.0)), "prediction_basis": "available_power_corrected", "source": "NWP + 本地微气象校正", "quality_score": 0.90 if quality["adjusted_samples"] else 0.86} for index, row in predictions.iterrows()]
        written = DB.save_forecast_rows(run_id, point_id, horizon, rows, kind, "Ridge 15分钟基线", "NWP + 本地校正")
        return {"run_id": run_id, "status": "success", "rows_written": written, "evaluation": evaluate_run(run_id)}
    except Exception as exc:
        DB.finish_run(run_id, "failed", 0, str(exc))
        raise


def next_schedule_at(schedule: dict[str, object]) -> str:
    return (now_dt() + timedelta(minutes=max(1, int(schedule.get("cadence_minutes") or 1440)))).isoformat()


def run_schedule(schedule: dict[str, object]) -> list[str]:
    run_ids = []
    targets = []
    for object_id in schedule.get("object_ids", []):
        obj = DB.resolve_object(str(object_id))
        if obj:
            targets.append((str(obj["point_id"]), str(obj["object_id"])))
    if not targets:
        targets = [(str(point_id), str(point_id)) for point_id in schedule.get("point_ids", [])]
    for point_id, object_id in targets:
        run_id = new_id("FC")
        DB.create_run(run_id, point_id, schedule["forecast_kind"], schedule["horizon"], "scheduled", object_id=object_id)
        execute_run(run_id, point_id, schedule["forecast_kind"], schedule["horizon"])
        run_ids.append(run_id)
    if run_ids:
        DB.mark_schedule_run(schedule["schedule_id"], run_ids[-1], next_schedule_at(schedule))
    return run_ids


def retry_run(run_id: str) -> dict[str, object]:
    """Re-run a failed forecast with the same object, kind and horizon."""
    with DB.connection() as conn:
        source = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
    if source is None:
        raise ValueError("run not found")
    if source["status"] != "failed":
        raise ValueError("only failed runs can be retried")
    object_id = str(source["object_id"] or source["point_id"])
    new_run_id = new_id("FC")
    DB.create_run(new_run_id, str(source["point_id"]), str(source["forecast_kind"]), str(source["horizon"]), "manual", object_id=object_id)
    result = execute_run(new_run_id, str(source["point_id"]), str(source["forecast_kind"]), str(source["horizon"]))
    result.update({"retry_of": run_id, "object_id": object_id})
    return result


def evaluate_pending(limit: int = 200) -> dict[str, object]:
    """Refresh pending evaluations after new observations have arrived."""
    pending = [item for item in DB.list_evaluations(limit=min(max(int(limit), 1), 500)) if item["status"] == "pending"]
    results: list[dict[str, object]] = []
    for item in pending:
        try:
            results.append(evaluate_run(str(item["run_id"])))
        except Exception as exc:
            results.append({"run_id": item["run_id"], "status": "failed", "error": str(exc)})
    return {
        "requested": len(pending),
        "completed": sum(1 for item in results if item.get("status") == "success"),
        "pending": sum(1 for item in results if item.get("status") == "pending"),
        "failed": sum(1 for item in results if item.get("status") == "failed"),
        "results": results,
    }


def observation_summary(point_id: str | None = None) -> list[dict[str, object]]:
    with DB.connection() as conn:
        if point_id:
            rows = conn.execute("SELECT point_id,source,COUNT(*) AS sample_count,MAX(timestamp_utc) AS last_timestamp,AVG(CASE WHEN quality_code='valid' THEN 1.0 ELSE 0.0 END) AS quality FROM weather_observations WHERE point_id=? GROUP BY point_id,source ORDER BY point_id,source", (point_id,)).fetchall()
        else:
            rows = conn.execute("SELECT point_id,source,COUNT(*) AS sample_count,MAX(timestamp_utc) AS last_timestamp,AVG(CASE WHEN quality_code='valid' THEN 1.0 ELSE 0.0 END) AS quality FROM weather_observations GROUP BY point_id,source ORDER BY point_id,source").fetchall()
        points = {row["point_id"]: row["name"] for row in conn.execute("SELECT point_id,name FROM forecast_points")}
    return [{"station_id": row["point_id"], "name": f"{points.get(row['point_id'], row['point_id'])} · {row['source']}", "point_id": row["point_id"], "source": row["source"], "variables": "风速 / 温度 / 辐射", "sample_count": row["sample_count"], "last_timestamp": row["last_timestamp"], "quality": round(float(row["quality"] or 0) * 100, 1), "status": "在线" if row["last_timestamp"] else "无数据"} for row in rows]


class SchedulerThread(threading.Thread):
    daemon = True

    def __init__(self) -> None:
        super().__init__(name="forecast-scheduler")
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(15):
            try:
                current = now_dt()
                with DB.connection() as conn:
                    due_runs = conn.execute("SELECT run_id,point_id,forecast_kind,horizon FROM forecast_runs WHERE status='accepted' AND scheduled_at IS NOT NULL AND scheduled_at<=?", (current.isoformat(),)).fetchall()
                for run in due_runs:
                    execute_run(run["run_id"], run["point_id"], run["forecast_kind"], run["horizon"])
                for schedule in DB.list_schedules():
                    if schedule["enabled"] and schedule.get("next_run_at") and parse_dt(schedule["next_run_at"]) <= current:
                        run_schedule(schedule)
            except Exception as exc:
                print(f"[scheduler] {exc}")


class Handler(BaseHTTPRequestHandler):
    server_version = "ForecastHMI/0.3"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self) -> dict[str, object] | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            value = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.json_response({"error": "invalid JSON"}, 400); return None
        if not isinstance(value, dict):
            self.json_response({"error": "JSON object required"}, 400); return None
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif parsed.path == "/api/health":
                self.json_response({**DB.health(), "time": now_iso(), "scheduler": "enabled"})
            elif parsed.path == "/api/points":
                self.json_response({"points": DB.list_points()})
            elif parsed.path == "/api/objects":
                self.json_response({"objects": DB.list_objects(enabled_only=False)})
            elif parsed.path in {"/api/forecast", "/api/curves"}:
                point_id = query.get("point_id", ["P001"])[0]; horizon = query.get("horizon", ["day_ahead"])[0]
                if horizon not in HORIZONS: raise ValueError("unsupported horizon")
                object_id = query.get("object_id", [None])[0]
                run_id = query.get("run_id", [None])[0]
                if run_id:
                    with DB.connection() as conn:
                        run = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
                    if run is None:
                        raise ValueError("run not found")
                    for field in ("point_id", "object_id", "horizon"):
                        if field in query and query[field][0] != run[field]:
                            raise ValueError(f"{field} does not match run")
                    point_id, object_id, horizon = run["point_id"], run["object_id"], run["horizon"]
                elif object_id:
                    obj = DB.resolve_object(object_id)
                    if not obj:
                        # Legacy point-scoped runs remain queryable after the
                        # object catalogue has been introduced or changed.
                        with DB.connection() as conn:
                            obj = conn.execute(
                                "SELECT point_id FROM forecast_runs WHERE object_id=? ORDER BY started_at DESC LIMIT 1",
                                (object_id,),
                            ).fetchone()
                    if not obj:
                        raise ValueError("unknown object_id")
                    point_id = str(obj["point_id"])
                kind = query.get("kind", [None])[0]
                if kind is not None and kind not in KINDS:
                    raise ValueError("unsupported kind")
                if not run_id:
                    with DB.connection() as conn:
                        latest = conn.execute(
                            """SELECT run_id,object_id FROM forecast_runs WHERE point_id=? AND horizon=? AND status='success'
                            AND (? IS NULL OR object_id=?) AND (? IS NULL OR forecast_kind=? OR forecast_kind='combined')
                            ORDER BY COALESCE(finished_at,started_at) DESC,started_at DESC LIMIT 1""",
                            (point_id, horizon, object_id, object_id, kind, kind),
                        ).fetchone()
                    run_id = latest["run_id"] if latest else None
                    if latest:
                        object_id = latest["object_id"]
                requested_limit = min(max(int(query.get("limit", [horizon_count(horizon)])[0]), 1), 2000)
                self.json_response({"point_id": point_id, "object_id": object_id, "horizon": horizon, "run_id": run_id, "rows": DB.get_forecast_rows(point_id, horizon, requested_limit, run_id, object_id, kind)})
            elif parsed.path == "/api/runs":
                self.json_response({"runs": DB.list_runs(
                    min(max(int(query.get("limit", [50])[0]), 1), 200),
                    query.get("status", [None])[0],
                    query.get("point_id", [None])[0],
                    query.get("horizon", [None])[0],
                    query.get("object_id", [None])[0],
                )})
            elif parsed.path == "/api/models":
                self.json_response({"models": DB.list_models(query.get("point_id", [None])[0], query.get("horizon", [None])[0], query.get("object_id", [None])[0])})
            elif parsed.path in {"/api/training-runs", "/api/trainings"}:
                self.json_response({"training_runs": DB.list_training_runs(int(query.get("limit", [50])[0]))})
            elif parsed.path == "/api/model-updates":
                self.json_response({"updates": DB.list_model_updates(int(query.get("limit", [50])[0]))})
            elif parsed.path == "/api/schedules":
                self.json_response({"schedules": DB.list_schedules()})
            elif parsed.path == "/api/nwp-sources":
                self.json_response({"sources": DB.list_nwp_sources()})
            elif parsed.path == "/api/tasks":
                self.json_response({"tasks": DB.list_tasks(query.get("point_id", [None])[0])})
            elif parsed.path == "/api/observation-summary":
                self.json_response({"stations": observation_summary(query.get("point_id", [None])[0])})
            elif parsed.path == "/api/curtailment-summary":
                self.json_response(curtailment_summary(query.get("point_id", [None])[0]))
            elif parsed.path == "/api/status":
                health = DB.health()
                runs = DB.list_runs(1)
                schedules = [s for s in DB.list_schedules() if s["enabled"]]
                evaluations = DB.list_evaluations(limit=500)
                self.json_response({"health": health, "latest_run": runs[0] if runs else None, "active_models": len([m for m in DB.list_models() if m["status"] == "active"]), "schedules": len(schedules), "pending_evaluations": sum(1 for item in evaluations if item["status"] == "pending"), "curtailment": curtailment_summary(), "next_schedule_at": min((s["next_run_at"] for s in schedules if s.get("next_run_at")), default=None)})
            elif parsed.path == "/api/settings":
                self.json_response({"settings": DB.get_settings()})
            elif parsed.path == "/api/evaluations":
                self.json_response({"evaluations": DB.list_evaluations(query.get("run_id", [None])[0], query.get("point_id", [None])[0], int(query.get("limit", [100])[0]))})
            elif parsed.path == "/api/weather-observations":
                self.json_response({"observations": DB.list_weather_observations(query.get("point_id", [None])[0], query.get("limit", [96])[0])})
            elif parsed.path == "/api/nwp-forecasts":
                self.json_response({"forecasts": DB.list_nwp_forecasts(query.get("point_id", [None])[0], query.get("horizon", [None])[0], query.get("limit", [672])[0])})
            elif parsed.path == "/api/energy-measurements":
                self.json_response({"measurements": DB.list_energy_measurements(query.get("point_id", [None])[0], query.get("limit", [96])[0])})
            elif parsed.path == "/api/curtailment-events":
                self.json_response({"events": DB.list_curtailment_events(query.get("point_id", [None])[0], query.get("limit", [100])[0])})
            else:
                self.serve_static(parsed.path)
        except (ValueError, KeyError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except Exception as exc:
            self.json_response({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path); payload = self.read_json()
        if payload is None: return
        try:
            if parsed.path == "/api/run/retry":
                self.json_response(retry_run(str(payload["run_id"])))
            elif parsed.path == "/api/evaluate-pending":
                self.json_response(evaluate_pending(int(payload.get("limit", 200))))
            elif parsed.path == "/api/run":
                horizon = str(payload.get("horizon", "day_ahead")); kind = str(payload.get("kind", "combined")); trigger = str(payload.get("trigger", "manual"))
                if horizon not in HORIZONS or kind not in KINDS or trigger not in TRIGGERS: raise ValueError("invalid run parameters")
                object_ids = payload.get("object_ids") or ([payload.get("object_id")] if payload.get("object_id") else [])
                resolved_objects = []
                if object_ids:
                    for object_id in object_ids:
                        obj = DB.resolve_object(str(object_id))
                        if not obj: raise ValueError(f"unknown object_id: {object_id}")
                        if not obj["enabled"]: raise ValueError(f"disabled object_id: {object_id}")
                        resolved_objects.append(obj)
                    point_ids = [str(obj["point_id"]) for obj in resolved_objects]
                else:
                    point_ids = payload.get("point_ids") or [payload.get("point_id", "P001")]
                if not isinstance(point_ids, list) or not point_ids: raise ValueError("point_ids must contain at least one point")
                known_points = {point["id"]: point for point in DB.list_points()}
                unknown = [str(point_id) for point_id in point_ids if str(point_id) not in known_points]
                disabled = [str(point_id) for point_id in point_ids if str(point_id) in known_points and not known_points[str(point_id)]["enabled"]]
                if unknown: raise ValueError(f"unknown point_ids: {', '.join(unknown)}")
                if disabled: raise ValueError(f"disabled point_ids: {', '.join(disabled)}")
                scheduled_at = payload.get("scheduled_at")
                if scheduled_at is not None and str(scheduled_at).strip():
                    scheduled_at = parse_dt(scheduled_at).isoformat()
                else:
                    scheduled_at = None
                run_specs = [
                    (str(point_id), resolved_objects[index]["object_id"] if resolved_objects else str(point_id))
                    for index, point_id in enumerate(point_ids)
                ]

                def run_one(spec: tuple[str, str]) -> dict[str, object]:
                    point_id, object_id = spec
                    run_id = new_id("FC")
                    DB.create_run(run_id, point_id, kind, horizon, trigger, scheduled_at, object_id)
                    if scheduled_at and parse_dt(scheduled_at) > now_dt():
                        return {"run_id": run_id, "object_id": object_id, "status": "accepted"}
                    result = execute_run(run_id, point_id, kind, horizon)
                    result["object_id"] = object_id
                    return result

                if scheduled_at and parse_dt(scheduled_at) > now_dt():
                    run_results = [run_one(spec) for spec in run_specs]
                elif len(run_specs) > 1:
                    # Independent objects use separate SQLite connections, so a
                    # bounded pool reduces waiting time for multi-object runs.
                    with ThreadPoolExecutor(max_workers=min(4, len(run_specs)), thread_name_prefix="forecast-run") as pool:
                        run_results = list(pool.map(run_one, run_specs))
                else:
                    run_results = [run_one(run_specs[0])]
                self.json_response({"run_id": run_results[0]["run_id"], "run_ids": [item["run_id"] for item in run_results], "object_ids": [item.get("object_id") for item in run_results], "status": run_results[-1]["status"], "results": run_results, "received": payload, "created_at": now_iso()})
            elif parsed.path in {"/api/train", "/api/training"}:
                object_ids = payload.get("object_ids") or ([payload.get("object_id")] if payload.get("object_id") else [])
                if object_ids:
                    resolved = [DB.resolve_object(str(object_id)) for object_id in object_ids]
                    if any(item is None for item in resolved): raise ValueError("unknown object_id")
                    results = [train_model(str(item["point_id"]), str(payload.get("target_kind", "combined")), str(payload.get("horizon", "day_ahead")), str(payload.get("mode", "offline")), int(payload["window_size"]) if payload.get("window_size") else None, str(item["object_id"]), str(payload.get("strategy", "hybrid_ridge"))) for item in resolved if item]
                else:
                    point_ids = payload.get("point_ids") or [payload.get("point_id")]
                    results = [train_model(str(point_id) if point_id else None, str(payload.get("target_kind", "combined")), str(payload.get("horizon", "day_ahead")), str(payload.get("mode", "offline")), int(payload["window_size"]) if payload.get("window_size") else None, None, str(payload.get("strategy", "hybrid_ridge"))) for point_id in point_ids]
                self.json_response({"status": "success", "results": results})
            elif parsed.path == "/api/model-update":
                point_id = payload.get("point_id")
                object_id = str(payload["object_id"]) if payload.get("object_id") else None
                if payload.get("object_id"):
                    obj = DB.resolve_object(str(payload["object_id"]))
                    if not obj: raise ValueError("unknown object_id")
                    point_id = obj["point_id"]
                result = train_model(point_id, str(payload.get("target_kind", "combined")), str(payload.get("horizon", "day_ahead")), "rolling", int(payload.get("window_size", 672)), object_id, str(payload.get("strategy", "hybrid_ridge")))
                self.json_response(result)
            elif parsed.path == "/api/evaluate":
                self.json_response(evaluate_run(str(payload["run_id"])))
            elif parsed.path == "/api/schedules":
                schedule = dict(payload); schedule.setdefault("schedule_id", new_id("SCH")); schedule.setdefault("point_ids", ["P001"]); schedule.setdefault("forecast_kind", "combined"); schedule.setdefault("horizon", "day_ahead"); schedule.setdefault("cadence_minutes", 1440); schedule.setdefault("enabled", True); schedule.setdefault("next_run_at", now_iso()); DB.upsert_schedule(schedule); self.json_response({"status": "saved", "schedule": next(item for item in DB.list_schedules() if item["schedule_id"] == schedule["schedule_id"])})
            elif parsed.path == "/api/nwp-sources":
                source = dict(payload); source.setdefault("source_id", new_id("NWP")); DB.upsert_nwp_source(source); self.json_response({"status": "saved", "sources": DB.list_nwp_sources()})
            elif parsed.path == "/api/tasks":
                task = dict(payload); task.setdefault("task_id", new_id("TASK")); DB.upsert_task(task); self.json_response({"status": "saved", "task": next(item for item in DB.list_tasks() if item["task_id"] == task["task_id"])})
            elif parsed.path == "/api/settings":
                DB.save_settings(payload); self.json_response({"status": "saved", "settings": DB.get_settings()})
            elif parsed.path == "/api/points":
                point = point_payload(payload); DB.upsert_point(point); self.json_response({"status": "saved", "point": next(item for item in DB.list_points() if item["id"] == point["id"])})
            elif parsed.path == "/api/objects":
                obj = dict(payload)
                if not obj.get("object_id") or not obj.get("point_id"): raise ValueError("object_id and point_id are required")
                DB.upsert_object(obj)
                self.json_response({"status": "saved", "object": DB.resolve_object(str(obj["object_id"]))})
            elif parsed.path == "/api/weather-observations":
                self.json_response({"status": "saved", "rows_written": DB.save_weather_observations(payload_rows(payload))})
            elif parsed.path == "/api/nwp-forecasts":
                self.json_response({"status": "saved", "rows_written": DB.save_nwp_forecasts(payload_rows(payload))})
            elif parsed.path == "/api/energy-measurements":
                self.json_response({"status": "saved", "rows_written": DB.save_energy_measurements(payload_rows(payload))})
            elif parsed.path == "/api/curtailment-events":
                self.json_response({"status": "saved", "rows_written": DB.save_curtailment_events(payload_rows(payload))})
            else:
                self.json_response({"error": "not found"}, 404)
        except (ValueError, KeyError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except Exception as exc:
            self.json_response({"error": str(exc)}, 500)

    def serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            candidate = (WEB / "index.html").resolve()
        elif path.startswith("/web/"):
            candidate = (WEB / path.removeprefix("/web/")).resolve()
        else:
            candidate = (ROOT / path.lstrip("/")).resolve()
        if not str(candidate).startswith(str(ROOT.resolve())) or not candidate.is_file():
            self.send_error(404); return
        content = candidate.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Forecast HMI")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args(); server = ThreadingHTTPServer((args.host, args.port), Handler); scheduler = SchedulerThread(); scheduler.start()
    print(f"Forecast HMI: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Forecast HMI")
    finally:
        scheduler.stop_event.set(); server.server_close()


if __name__ == "__main__":
    main()
