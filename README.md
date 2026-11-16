# 15 分钟天气与能源预测整合工程

本目录把不同来源的开源项目作为可替换组件，在本地统一成一条 15 分钟预测链路：

```
公开 NWP（Open-Meteo / hefty） ─┐
本地气象采集（CSV/接口）       ├─ 天气融合 ─┐
历史与实时新能源出力、负荷     ─┘           ├─ 统一特征 ── 预测
                                             ├─ 光伏出力
                                             ├─ 风电出力
                                             └─ 负荷
```

预测尺度统一为：超短期 4 小时（16 点）、日前 24 小时（96 点）、周前 7 天（672 点），间隔均为 15 分钟。当前本地实现提供可审计的 Ridge 基线；后续可把模型替换为 `vendor/` 中的 OpenSTEF、Quartz、hefty 或 `loadforecasting` 模型，数据契约不变。

## 已下载的开源上游

| 目录 | 用途 | 许可证/说明 |
|---|---|---|
| `vendor/hefty` | NOAA GFS/GEFS/HRRR、ECMWF IFS 的风电和光伏 NWP 预测 | BSD-3-Clause |
| `vendor/openstef` | 短期能源预测、特征工程和概率预测流水线 | MPL-2.0 |
| `vendor/quartz-solar-forecast` | 站点级光伏 0–48 小时预测 | 以仓库 LICENSE 为准 |
| `vendor/loadforecasting` | Transformer、LSTM、XGBoost 等短期负荷模型 | MIT |
| `vendor/loaf` | 网格天气预报与本地传感器融合的超本地天气预测 | MIT，项目仍在开发 |

## 数据接口

本地观测 CSV 至少包含：`timestamp, wind_speed_mps, temperature_c, solar_radiation_wm2, pv_power_kw, wind_power_kw, load_kw`。也接受常见别名，如 `wind_speed_10m`、`temperature_2m`、`ghi`、`load`。时间戳可带时区；无时区时按 `Pacific/Auckland` 解释并在内部转换为 UTC。

公开天气默认通过 Open-Meteo 的 forecast/archive API 获取，预测数据可按 15 分钟返回；历史数据按小时获取后重采样到 15 分钟。预测时，`fuse_weather()` 使用训练数据中最近 96 个 15 分钟点的本地观测误差做 NWP 偏差修正，并在已观测时段优先使用本地值，因此历史采集值和最新实时值会进入天气预测链路。

## 离线运行示例

```powershell
cd D:\codex\forecast
python -m pip install -e .
python scripts\demo.py
```

示例会生成 `demo_week_ahead_forecast.csv`，并验证 672 个 15 分钟周前预测点。接入真实数据时：

```python
import pandas as pd
from forecast_pipeline import IntegratedForecastPipeline, OpenMeteoProvider

observations = pd.read_csv("data/observations.csv")
pipeline = IntegratedForecastPipeline(
    provider=OpenMeteoProvider(latitude=-36.85, longitude=174.76)
).fit(observations)

ultrashort = pipeline.predict("ultrashort")
day_ahead = pipeline.predict("day_ahead")
week_ahead = pipeline.predict("week_ahead")
```

真实部署前应按站点做滚动回测，分别报告天气 MAE/RMSE、光伏/风电/负荷 MAE、RMSE、nMAE 和预测区间覆盖率，并对 NWP 版本、观测缺失、夏令时和功率限幅进行记录。

## Web 人机界面

本地 HMI 位于 `web/`，按功能划分为多级目录和独立页面：

- 运行中心：主画面、运行状态
- 预测点管理：预测点列表、点参数配置
- 数据查询：天气输入曲线、能源输入曲线、预测结果
- 气象预测：气象预测总览、气象预测结果、气象误差评估
- 气象预测配置：本地观测管理、NWP 资料配置、气象预测运行
- 预测任务：任务配置、人工启动、历史运行
- 系统设置：数据源与模型

启动本地服务：

```powershell
cd D:\codex\forecast
python scripts\serve_web.py --port 18765
```

打开 `http://127.0.0.1:18765/`。主画面提供运行总览、预测任务快捷启动、预测点健康度和曲线摘要；详细曲线、参数编辑和运行日志分别位于独立页面。页面主体固定在视口内，长内容在当前功能页内部滚动且隐藏滚动条，横向溢出不会撑开浏览器窗口。服务同时提供 `/api/health`、`/api/points`、`/api/forecast` 和 `/api/run` 路由，便于后续接入真实数据库、采集接口和调度服务。

## SQLite 数据库

服务使用 `D:\codex\sqldb\realtime\forecast\forecast.db` 作为界面展示和后台计算的共享数据存储。首次使用时运行：

```powershell
cd D:\codex\forecast
python scripts\init_forecast_db.py
```

初始化会创建 schema、默认预测点、Open-Meteo/NOAA NWP 数据源配置，并把示例周前预测写入数据库。开发环境如需重建，可使用 `python scripts\init_forecast_db.py --reset`。数据库包含预测点、预测任务运行记录、气象历史/实时观测、NWP/融合天气预测、新能源与负荷测量、光伏/风电/负荷预测结果、NWP 数据源、任务配置、模型注册、训练运行、滚动更新、自动调度和结果评估表。服务 API `/api/health`、`/api/points`、`/api/forecast`、`/api/runs`、`/api/weather-observations` 和 `/api/energy-measurements` 均读写这套 SQLite 数据。生产环境应同时备份 `forecast.db`、`forecast.db-wal` 和 `forecast.db-shm` 文件。可通过环境变量 `FORECAST_DB_PATH` 覆盖默认路径。

模型和任务闭环还包括：

- `POST /api/train`：离线训练 Ridge 15 分钟基线，保存模型版本、系数文件、训练样本数和验证指标。
- `POST /api/model-update`：按最近窗口执行滚动更新，登记 `model_updates` 更新记录。
- `POST /api/run`：人工启动或指定未来启动时刻；立即任务会生成预测、写入 `forecast_runs` 并创建后评估记录。
- `GET/POST /api/schedules`：配置自动定时预测；服务内置调度线程按 `next_run_at` 和 `cadence_minutes` 执行任务。
- `GET /api/evaluations`、`POST /api/evaluate`：将预测与同点位实测值按 15 分钟时间戳对齐，计算 MAE、RMSE、BIAS 和 nMAE；实际值尚未到达时记录为 `pending`，补齐后可再次评估。

对应 Web 页面位于“预测任务”和“模型管理”目录：自动定时预测、模型离线训练、模型滚动更新、历史运行和预测结果后评估。模型工件默认写入 `data/models/`。

也可以不启动 Web 服务直接训练：

```powershell
python scripts\train_model.py --point P001 --kind combined --horizon day_ahead
python scripts\train_model.py --point P001 --kind energy --mode rolling --window-size 672
```

当本地观测表已有不少于 64 个完整的 15 分钟历史样本时，训练和运行任务优先使用 SQLite 中的真实观测；样本不足时使用 `demo_week_ahead_forecast.csv` 作为可审计的离线兜底。接入生产采集后，可通过 `/api/weather-observations` 和 `/api/energy-measurements` 持续写入历史/实时数据，训练、滚动更新和后评估会自动切换到真实数据。

气象预测功能单独覆盖风速、天气温度和太阳辐射：气象预测总览显示未来 15 分钟关键值、未来 24 小时曲线、NWP 与本地站点状态；NWP 资料配置维护主备模型、刷新周期和变量映射；本地观测管理维护站点、采集变量和质量；气象预测运行支持单独执行融合、偏差修正和质量检查；误差评估提供 MAE、RMSE、BIAS 和滚动趋势。曲线、明细、质量和误差趋势使用同一页面内的分页标签，每个标签只展示一个功能面板。

## 来源链接

- https://github.com/williamhobbs/hefty
- https://github.com/OpenSTEF/OpenSTEF
- https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast
- https://github.com/erc-fhv/loadforecasting
- https://github.com/PandoroML/LOAF
