from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from forecast_pipeline import IntegratedForecastPipeline  # noqa: E402
from scripts.demo import make_demo_data  # noqa: E402


def test_all_horizons_have_15_minute_grid():
    data = make_demo_data()
    pipeline = IntegratedForecastPipeline().fit(data)
    future = data.tail(1).set_index("timestamp")[
        ["wind_speed_mps", "temperature_c", "solar_radiation_wm2"]
    ]
    index = pd.date_range(data.timestamp.iloc[-1] + pd.Timedelta(minutes=15), periods=672, freq="15min")
    future = future.reindex(index).ffill()
    assert len(pipeline.predict("ultrashort", future).index) == 16
    assert len(pipeline.predict("day_ahead", future).index) == 96
    result = pipeline.predict("week_ahead", future)
    assert len(result) == 672
    assert (result.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert result[["pv_power_kw", "wind_power_kw", "load_kw"]].ge(0).all().all()
