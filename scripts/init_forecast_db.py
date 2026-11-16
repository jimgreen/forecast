"""Create and seed the local forecast SQLite database.

Usage:
    python scripts/init_forecast_db.py
    python scripts/init_forecast_db.py --db data/forecast.db --reset
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_pipeline.storage import ForecastDB, seed_from_demo_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the forecast SQLite database")
    parser.add_argument("--db", default=str(ROOT / "data" / "forecast.db"))
    parser.add_argument("--reset", action="store_true", help="remove the named database before initialization")
    args = parser.parse_args()
    db_path = Path(args.db)
    if args.reset and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
    db = ForecastDB(db_path)
    db.initialize()
    db.seed_defaults()
    rows = seed_from_demo_csv(db, ROOT / "demo_week_ahead_forecast.csv")
    print(f"initialized {db_path}")
    print(f"seeded forecast rows: {rows}")
    print(db.health())


if __name__ == "__main__":
    main()
