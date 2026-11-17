"""Run an offline or rolling model update without starting the HMI.

Examples:
    python scripts/train_model.py --point P001 --kind combined --horizon day_ahead
    python scripts/train_model.py --point P001 --kind energy --mode rolling --window-size 672
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from serve_web import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or roll the local forecast model")
    parser.add_argument("--point", default=None)
    parser.add_argument("--kind", choices=("weather", "energy", "combined"), default="combined")
    parser.add_argument("--horizon", choices=("hour_ahead", "ultrashort", "day_ahead", "week_ahead"), default="day_ahead")
    parser.add_argument("--mode", choices=("offline", "rolling"), default="offline")
    parser.add_argument("--window-size", type=int, default=None)
    args = parser.parse_args()
    result = train_model(args.point, args.kind, args.horizon, args.mode, args.window_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
