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
from datetime import datetime, timedelta, timezone
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

from forecast_pipeline.pipeline import IntegratedForecastPipeline  # noqa: E402
from forecast_pipeline.schema import make_future_index  # noqa: E402
from forecast_pipeline.storage import ForecastDB, seed_from_demo_csv  # noqa: E402

DB = ForecastDB(ROOT / "data" / "forecast.db")
DB.initialize()
DB.seed_defaults()
seed_from_demo_csv(DB, DATA)

HORIZONS = {"ultrashort": 16, "day_ahead": 96, "week_ahead": 672}
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
            e.pv_power_kw,e.wind_power_kw,e.load_kw
            FROM weather_observations w LEFT JOIN energy_measurements e
            ON e.point_id=w.point_id AND e.timestamp_utc=w.timestamp_utc
            WHERE w.point_id=?
            UNION ALL
            SELECT e.timestamp_utc AS timestamp,
            w.wind_speed_mps,w.temperature_c,w.solar_radiation_wm2,
            e.pv_power_kw,e.wind_power_kw,e.load_kw
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
    for column in (*WEATHER_TARGETS, *ENERGY_TARGETS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").dropna(subset=[*WEATHER_TARGETS, *ENERGY_TARGETS]).reset_index(drop=True)


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


def metric_values(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int | None]:
    values = pd.concat([actual, predicted], axis=1).dropna()
    if values.empty:
        return {"mae": None, "rmse": None, "bias": None, "nmae": None, "sample_count": 0}
    error = values.iloc[:, 1] - values.iloc[:, 0]
    mae = float(np.abs(error).mean())
    return {"mae": mae, "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()), "nmae": float(mae / max(float(np.abs(values.iloc[:, 0]).mean()), 1e-9) * 100), "sample_count": int(len(values))}


def train_model(point_id: str | None, target_kind: str, horizon: str, mode: str, window_size: int | None = None) -> dict[str, object]:
    if target_kind not in KINDS or horizon not in HORIZONS or mode not in {"offline", "rolling"}:
        raise ValueError("invalid training parameters")
    training_id = new_id("TRN")
    model_id = new_id("MDL")
    DB.create_training_run(training_id, point_id, target_kind, horizon, mode, window_size, None)
    try:
        frame = load_observation_frame(point_id)
        if len(frame) < 64:
            frame = load_demo_frame()
        if mode == "rolling":
            frame = frame.tail(min(int(window_size or 672), len(frame))).reset_index(drop=True)
        holdout = min(horizon_count(horizon), max(16, len(frame) // 5))
        train_frame = frame.iloc[:-holdout].copy()
        test_frame = frame.iloc[-holdout:].copy()
        if len(train_frame) < 40:
            raise ValueError("not enough historical rows for training")
        train_frame = train_frame.set_index("timestamp")
        test_frame = test_frame.set_index("timestamp")
        pipeline = IntegratedForecastPipeline().fit(train_frame)
        prediction = pipeline.predict(len(test_frame), future_weather=test_frame[list(WEATHER_TARGETS)])
        metrics: dict[str, object] = {target: metric_values(test_frame[target], prediction[target]) for target in (*WEATHER_TARGETS, *ENERGY_TARGETS)}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {"model_id": model_id, "model_type": "ridge15min", "target_kind": target_kind, "horizon": horizon, "feature_count": pipeline.model.feature_count if pipeline.model else 0, "coefficients": {key: value.tolist() for key, value in (pipeline.model.coefficients.items() if pipeline.model else [])}}
        artifact_path = MODEL_DIR / f"{model_id}.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        model = {"model_id": model_id, "model_name": "Ridge 15分钟基线", "model_type": "ridge15min", "target_kind": target_kind, "horizon": horizon, "point_id": point_id, "version": now_dt().strftime("v%Y%m%d%H%M%S"), "status": "active", "artifact_path": str(artifact_path), "hyperparams": {"alpha": 1.0, "mode": mode, "window_size": window_size}, "metrics": metrics}
        DB.register_model(model)
        DB.finish_training_run(training_id, "success", model_id, len(train_frame), metrics)
        if mode == "rolling":
            DB.save_model_update({"update_id": new_id("UPD"), "model_id": model_id, "point_id": point_id, "horizon": horizon, "update_strategy": "rolling_window", "window_size": len(frame), "sample_count": len(train_frame), "status": "success", "details": {"training_id": training_id, "metrics": metrics}})
        return {"training_id": training_id, "model_id": model_id, "status": "success", "sample_count": len(train_frame), "metrics": metrics, "artifact_path": str(artifact_path)}
    except Exception as exc:
        DB.finish_training_run(training_id, "failed", model_id, 0, {}, str(exc))
        raise


def evaluate_run(run_id: str) -> dict[str, object]:
    with DB.connection() as conn:
        run = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
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
            actual_rows = conn.execute(f"SELECT timestamp_utc,{target} AS value FROM {actual_table} WHERE point_id=?", (run["point_id"],)).fetchall()
            actual = pd.Series({parse_dt(row["timestamp_utc"]): row["value"] for row in actual_rows}, dtype=float)
            predicted = pd.Series({parse_dt(row["target_time_utc"]): row["value"] for row in forecast_rows}, dtype=float)
            result = metric_values(actual, predicted)
            details[target] = result
            paired += int(result["sample_count"] or 0)
        available = [value for value in details.values() if value.get("sample_count", 0)]
        status = "success" if available else "pending"
        aggregate = {key: float(np.mean([float(value[key]) for value in available if value.get(key) is not None])) for key in ("mae", "rmse", "bias", "nmae")} if available else {key: None for key in ("mae", "rmse", "bias", "nmae")}
        evaluation = {"evaluation_id": new_id("EVA"), "run_id": run_id, "point_id": run["point_id"], "forecast_kind": forecast_kind, "horizon": run["horizon"], "window_start": None, "window_end": None, "sample_count": paired, "status": status, **aggregate, "coverage": None, "metrics": details}
    DB.save_evaluation(evaluation)
    return evaluation


def execute_run(run_id: str, point_id: str, kind: str, horizon: str) -> dict[str, object]:
    try:
        history = load_observation_frame(point_id)
        if len(history) >= 64:
            history_indexed = history.set_index("timestamp")
            future_index = make_future_index(history_indexed.index[-1], horizon)
            template = pd.DataFrame(load_template_rows(horizon))
            future_weather = template[list(WEATHER_TARGETS)].set_axis(future_index)
            predictions = IntegratedForecastPipeline().fit(history_indexed).predict(horizon, future_weather=future_weather)
            rows = [{"timestamp": index.isoformat(), **{column: float(value) for column, value in row.items()}} for index, row in predictions.iterrows()]
        else:
            rows = load_template_rows(horizon)
        written = DB.save_forecast_rows(run_id, point_id, horizon, rows, kind, "Ridge 15分钟基线", "NWP + 本地校正")
        return {"run_id": run_id, "status": "success", "rows_written": written, "evaluation": evaluate_run(run_id)}
    except Exception as exc:
        DB.finish_run(run_id, "failed", 0, str(exc))
        raise


def next_schedule_at(schedule: dict[str, object]) -> str:
    return (now_dt() + timedelta(minutes=max(1, int(schedule.get("cadence_minutes") or 1440)))).isoformat()


def run_schedule(schedule: dict[str, object]) -> list[str]:
    run_ids = []
    for point_id in schedule.get("point_ids", []):
        run_id = new_id("FC")
        DB.create_run(run_id, str(point_id), schedule["forecast_kind"], schedule["horizon"], "scheduled")
        execute_run(run_id, str(point_id), schedule["forecast_kind"], schedule["horizon"])
        run_ids.append(run_id)
    if run_ids:
        DB.mark_schedule_run(schedule["schedule_id"], run_ids[-1], next_schedule_at(schedule))
    return run_ids


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
            if parsed.path == "/api/health":
                self.json_response({**DB.health(), "time": now_iso(), "scheduler": "enabled"})
            elif parsed.path == "/api/points":
                self.json_response({"points": DB.list_points()})
            elif parsed.path in {"/api/forecast", "/api/curves"}:
                point_id = query.get("point_id", ["P001"])[0]; horizon = query.get("horizon", ["day_ahead"])[0]
                if horizon not in HORIZONS: raise ValueError("unsupported horizon")
                self.json_response({"point_id": point_id, "horizon": horizon, "rows": DB.get_forecast_rows(point_id, horizon, horizon_count(horizon))})
            elif parsed.path == "/api/runs":
                self.json_response({"runs": DB.list_runs(min(max(int(query.get("limit", [50])[0]), 1), 200))})
            elif parsed.path == "/api/models":
                self.json_response({"models": DB.list_models(query.get("point_id", [None])[0], query.get("horizon", [None])[0])})
            elif parsed.path in {"/api/training-runs", "/api/trainings"}:
                self.json_response({"training_runs": DB.list_training_runs(int(query.get("limit", [50])[0]))})
            elif parsed.path == "/api/model-updates":
                self.json_response({"updates": DB.list_model_updates(int(query.get("limit", [50])[0]))})
            elif parsed.path == "/api/schedules":
                self.json_response({"schedules": DB.list_schedules()})
            elif parsed.path == "/api/evaluations":
                self.json_response({"evaluations": DB.list_evaluations(query.get("run_id", [None])[0], int(query.get("limit", [100])[0]))})
            elif parsed.path == "/api/weather-observations":
                self.json_response({"observations": DB.list_weather_observations(query.get("point_id", [None])[0], query.get("limit", [96])[0])})
            elif parsed.path == "/api/energy-measurements":
                self.json_response({"measurements": DB.list_energy_measurements(query.get("point_id", [None])[0], query.get("limit", [96])[0])})
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
            if parsed.path == "/api/run":
                horizon = str(payload.get("horizon", "day_ahead")); kind = str(payload.get("kind", "combined")); trigger = str(payload.get("trigger", "manual"))
                if horizon not in HORIZONS or kind not in KINDS or trigger not in TRIGGERS: raise ValueError("invalid run parameters")
                point_ids = payload.get("point_ids") or [payload.get("point_id", "P001")]
                if not isinstance(point_ids, list) or not point_ids: raise ValueError("point_ids must contain at least one point")
                run_results = []
                for point_id in point_ids:
                    run_id = new_id("FC"); DB.create_run(run_id, str(point_id), kind, horizon, trigger, payload.get("scheduled_at"))
                    if payload.get("scheduled_at") and parse_dt(payload["scheduled_at"]) > now_dt():
                        run_results.append({"run_id": run_id, "status": "accepted"})
                    else:
                        run_results.append(execute_run(run_id, str(point_id), kind, horizon))
                self.json_response({"run_id": run_results[0]["run_id"], "run_ids": [item["run_id"] for item in run_results], "status": run_results[-1]["status"], "results": run_results, "received": payload, "created_at": now_iso()})
            elif parsed.path in {"/api/train", "/api/training"}:
                point_ids = payload.get("point_ids") or [payload.get("point_id")]
                results = [train_model(str(point_id) if point_id else None, str(payload.get("target_kind", "combined")), str(payload.get("horizon", "day_ahead")), str(payload.get("mode", "offline")), int(payload["window_size"]) if payload.get("window_size") else None) for point_id in point_ids]
                self.json_response({"status": "success", "results": results})
            elif parsed.path == "/api/model-update":
                result = train_model(payload.get("point_id"), str(payload.get("target_kind", "combined")), str(payload.get("horizon", "day_ahead")), "rolling", int(payload.get("window_size", 672)))
                self.json_response(result)
            elif parsed.path == "/api/evaluate":
                self.json_response(evaluate_run(str(payload["run_id"])))
            elif parsed.path == "/api/schedules":
                schedule = dict(payload); schedule.setdefault("schedule_id", new_id("SCH")); schedule.setdefault("point_ids", ["P001"]); schedule.setdefault("forecast_kind", "combined"); schedule.setdefault("horizon", "day_ahead"); schedule.setdefault("cadence_minutes", 1440); schedule.setdefault("enabled", True); schedule.setdefault("next_run_at", now_iso()); DB.upsert_schedule(schedule); self.json_response({"status": "saved", "schedule": next(item for item in DB.list_schedules() if item["schedule_id"] == schedule["schedule_id"])})
            elif parsed.path == "/api/points":
                point = point_payload(payload); DB.upsert_point(point); self.json_response({"status": "saved", "point": next(item for item in DB.list_points() if item["id"] == point["id"])})
            elif parsed.path == "/api/weather-observations":
                rows = payload.get("rows", [payload]); self.json_response({"status": "saved", "rows_written": DB.save_weather_observations(rows)})
            elif parsed.path == "/api/energy-measurements":
                rows = payload.get("rows", [payload]); self.json_response({"status": "saved", "rows_written": DB.save_energy_measurements(rows)})
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
