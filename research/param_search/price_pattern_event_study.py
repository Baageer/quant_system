# # Price Pattern Event Study
#
# This notebook studies post-event price behavior around indicator conditions such as `bollinger_squeeze`.
#
# Workflow:
# 1. Load daily OHLCV data from the existing `DataAPI`.
# 2. Build a boolean condition series for the selected pattern.
# 3. Collapse consecutive true values into distinct events.
# 4. Expand each event into `[-30, +30]` trading-day windows.
# 5. Align and normalize price paths, then summarize forward returns and dispersion.
# 6. Explore multiple parameter combinations and rank them by forward performance.
#
# The notebook is intentionally stricter than a naive slice-and-plot flow:
# - it deduplicates continuous signals,
# - drops incomplete event windows when requested,
# - keeps per-event metadata for drill-down,
# - compares mean and median paths to reduce outlier bias,
# - separates the single-run event study logic from the parameter exploration loop.

from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import json
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

try:
    from IPython.display import display
except ImportError:
    def display(value):
        if hasattr(value, "to_string"):
            print(value.to_string())
        else:
            print(value)


def find_project_root(start=None):
    start = Path(__file__).resolve().parent if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "run_backtest.py").exists() and (candidate / "config" / "settings.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from the current working directory.")


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_api import DataAPI
from research.param_search.conditions.registry import (
    BASE_CONDITION_PARAMS_MAP,
    CONDITION_PARAM_GRID_MAP,
    build_condition_frame,
)

CONDITION_NAME = "rsrs_breakout"  # bollinger_squeeze | rsrs_breakout
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

SYMBOLS = None      # ["000001", "000002"]
STOCK_FILE = None   # "./data/HS300.txt"
MAX_SYMBOLS = 300
ADJUST_MODE = "qfq"  # "qfq" | "hfq"

WINDOW_BEFORE = 30
WINDOW_AFTER = 30
MIN_GAP_BETWEEN_EVENTS = 20
EVENT_SELECTION = "first_in_run"  # first_in_run | all_true_days
REQUIRE_FULL_WINDOW = True
MIN_DATA_LENGTH = 120

FORWARD_HORIZONS = [1, 5, 10, 20, 30]
SHOW_SAMPLE_PLOTS = 6
EXPORT_RESULTS = False
SHOW_RESULTS = False

RANK_BY_HORIZON = 20
TOP_N_EXPERIMENTS = 15
MIN_EVENTS_PER_EXPERIMENT = 8
MAX_EXPERIMENTS = None
USE_MULTIPROCESSING = True
MAX_WORKERS = 12
PROCESS_POOL_CHUNK_SIZE = 1

if CONDITION_NAME not in BASE_CONDITION_PARAMS_MAP:
    supported = ", ".join(sorted(BASE_CONDITION_PARAMS_MAP.keys()))
    raise ValueError(f"Unsupported CONDITION_NAME: {CONDITION_NAME}. Available: {supported}")

BASE_CONDITION_PARAMS = dict(BASE_CONDITION_PARAMS_MAP[CONDITION_NAME])
CONDITION_PARAM_GRID = dict(CONDITION_PARAM_GRID_MAP.get(CONDITION_NAME, {}))


def count_parameter_combinations(param_grid):
    if not param_grid:
        return 1
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total

STANDARD_PRICE_COLUMNS = [
    "date",
    "code",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]

WORKER_PRICE_CACHE = None
WORKER_CONDITION_NAME = None
WORKER_RANK_BY_HORIZON = None
WORKER_MIN_EVENTS_PER_EXPERIMENT = None


def is_missing(value):
    return value is None or value != value


def standardize_price_frame(raw_df):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    if len(df.columns) == 13:
        df.columns = ["index", *STANDARD_PRICE_COLUMNS]
    elif len(df.columns) == 12:
        df.columns = STANDARD_PRICE_COLUMNS

    rename_map = {}
    for column in df.columns:
        name = str(column).strip().lower()
        if name in {"date", "datetime", "日期"}:
            rename_map[column] = "date"
        elif name in {"open", "开盘"}:
            rename_map[column] = "open"
        elif name in {"close", "收盘"}:
            rename_map[column] = "close"
        elif name in {"high", "最高"}:
            rename_map[column] = "high"
        elif name in {"low", "最低"}:
            rename_map[column] = "low"
        elif name in {"volume", "vol", "成交量"}:
            rename_map[column] = "volume"
        elif name in {"code", "symbol", "股票代码"}:
            rename_map[column] = "code"

    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def extract_event_dates(condition, min_gap_between_events=0, event_selection="first_in_run"):
    condition = condition.fillna(False).astype(bool)

    if event_selection == "all_true_days":
        candidate_dates = list(condition[condition].index)
    else:
        previous_condition = condition.shift(1, fill_value=False)
        run_starts = condition & (~previous_condition)
        candidate_dates = list(run_starts[run_starts].index)

    if min_gap_between_events <= 0:
        return candidate_dates

    index_lookup = {timestamp: i for i, timestamp in enumerate(condition.index)}
    selected_dates = []
    last_position = None

    for event_date in candidate_dates:
        current_position = index_lookup[event_date]
        if last_position is None or current_position - last_position >= min_gap_between_events:
            selected_dates.append(event_date)
            last_position = current_position

    return selected_dates


def build_event_window(df, symbol, event_date, window_before, window_after, require_full_window=True):
    event_position = df.index.get_loc(event_date)
    start_position = event_position - window_before
    end_position = event_position + window_after

    if require_full_window and (start_position < 0 or end_position >= len(df)):
        return None

    start_position = max(start_position, 0)
    end_position = min(end_position, len(df) - 1)
    window = df.iloc[start_position : end_position + 1].copy()

    if window.empty:
        return None

    event_close = float(df.loc[event_date, "close"])
    if is_missing(event_close) or event_close <= 0:
        return None

    relative_days = np.arange(start_position - event_position, end_position - event_position + 1)
    window["relative_day"] = relative_days
    window["symbol"] = symbol
    window["event_date"] = pd.Timestamp(event_date)
    window["event_close"] = event_close
    window["close_to_event"] = window["close"] / event_close - 1.0
    return window.reset_index().rename(columns={window.index.name or "index": "trade_date"})


def build_price_cache(symbols, data_api, start_date, end_date, min_data_length):
    price_cache = {}
    cache_skip_rows = []

    for symbol in tqdm(symbols, desc="Loading price cache", unit="symbol"):
        try:
            raw_df = data_api.get_price_history_data(symbol, start_date, end_date)
            price_df = standardize_price_frame(raw_df)

            required_columns = {"close", "high", "low"}
            if price_df.empty:
                cache_skip_rows.append({"symbol": symbol, "reason": "empty_price_data"})
                continue
            if len(price_df) < min_data_length:
                cache_skip_rows.append({"symbol": symbol, "reason": "insufficient_history"})
                continue
            if not required_columns.issubset(price_df.columns):
                cache_skip_rows.append({"symbol": symbol, "reason": "missing_required_columns"})
                continue

            price_cache[symbol] = price_df
        except Exception as exc:
            cache_skip_rows.append({"symbol": symbol, "reason": type(exc).__name__, "detail": str(exc)})

    return price_cache, pd.DataFrame(cache_skip_rows)


def iter_param_sets(base_params, param_grid, max_experiments=None):
    if not param_grid:
        yield 1, "baseline", dict(base_params)
        return

    keys = list(param_grid.keys())
    value_lists = [param_grid[key] for key in keys]

    for experiment_id, values in enumerate(product(*value_lists), start=1):
        params = dict(base_params)
        overrides = dict(zip(keys, values))
        params.update(overrides)

        if params.get("squeeze_threshold") is not None:
            params["squeeze_quantile"] = base_params.get("squeeze_quantile")
            params["squeeze_lookback"] = base_params.get("squeeze_lookback")

        label = " | ".join(f"{key}={value}" for key, value in overrides.items())
        yield experiment_id, label, params

        if max_experiments is not None and experiment_id >= max_experiments:
            break


EVENT_META_NUMERIC_COLUMNS = [
    "bandwidth",
    "condition_threshold",
    "rsrs_beta",
    "rsrs_r2",
    "rsrs_zscore",
    "rsrs_score",
]
EVENT_META_BOOLEAN_COLUMNS = [
    "breakout_valid",
    "volume_confirmation",
    "trend_long_confirmation",
    "trend_short_confirmation",
    "supertrend_long_confirmation",
    "supertrend_short_confirmation",
    "band_expansion_confirmation",
    "return_up_confirmation",
    "return_down_confirmation",
]
EVENT_META_TEXT_COLUMNS = ["event_direction"]


def _get_event_value(analysis_df, event_date, column):
    if column not in analysis_df.columns:
        return None
    value = analysis_df.loc[event_date, column]
    if isinstance(value, pd.Series):
        return value.iloc[-1]
    return value


def build_event_meta_row(analysis_df, symbol, event_date):
    row = {
        "symbol": symbol,
        "event_date": pd.Timestamp(event_date),
        "event_close": float(analysis_df.loc[event_date, "close"]),
    }

    for column in EVENT_META_NUMERIC_COLUMNS:
        value = _get_event_value(analysis_df, event_date, column)
        if value is None or is_missing(value):
            continue
        row[column] = float(value)

    for column in EVENT_META_BOOLEAN_COLUMNS:
        value = _get_event_value(analysis_df, event_date, column)
        if value is None or is_missing(value):
            continue
        row[column] = bool(value)

    for column in EVENT_META_TEXT_COLUMNS:
        value = _get_event_value(analysis_df, event_date, column)
        if value is None or is_missing(value):
            continue
        row[column] = str(value)

    return row


def run_single_event_study(price_cache, condition_name, condition_params):
    event_windows = []
    event_rows = []
    skip_rows = []

    for symbol, price_df in price_cache.items():
        try:
            condition_frame = build_condition_frame(price_df, condition_name, condition_params)
            analysis_df = price_df.join(condition_frame)
            analysis_df = analysis_df.dropna(subset=["close"])

            event_dates = extract_event_dates(
                analysis_df["condition"],
                min_gap_between_events=MIN_GAP_BETWEEN_EVENTS,
                event_selection=EVENT_SELECTION,
            )

            if not event_dates:
                skip_rows.append({"symbol": symbol, "reason": "no_events"})
                continue

            for event_date in event_dates:
                window_df = build_event_window(
                    analysis_df,
                    symbol=symbol,
                    event_date=event_date,
                    window_before=WINDOW_BEFORE,
                    window_after=WINDOW_AFTER,
                    require_full_window=REQUIRE_FULL_WINDOW,
                )

                if window_df is None:
                    continue

                event_windows.append(window_df)
                event_rows.append(build_event_meta_row(analysis_df, symbol, event_date))
        except Exception as exc:
            skip_rows.append({"symbol": symbol, "reason": type(exc).__name__, "detail": str(exc)})

    event_meta = pd.DataFrame(event_rows).sort_values(["event_date", "symbol"]).reset_index(drop=True) if event_rows else pd.DataFrame()
    event_window_df = pd.concat(event_windows, ignore_index=True) if event_windows else pd.DataFrame()
    skip_df = pd.DataFrame(skip_rows)

    path_matrix = pd.DataFrame()
    forward_summary = pd.DataFrame()
    event_count_by_symbol = pd.Series(dtype=int)

    if not event_meta.empty:
        path_matrix = (
            event_window_df
            .pivot_table(index=["symbol", "event_date"], columns="relative_day", values="close_to_event")
            .sort_index(axis=1)
        )

        forward_rows = []
        for horizon in FORWARD_HORIZONS:
            if horizon not in path_matrix.columns:
                continue

            values = path_matrix[horizon].dropna()
            if values.empty:
                continue

            forward_rows.append(
                {
                    "horizon": horizon,
                    "sample_count": int(values.shape[0]),
                    "mean_return": float(values.mean()),
                    "median_return": float(values.median()),
                    "win_rate": float((values > 0).mean()),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                }
            )

        forward_summary = pd.DataFrame(forward_rows)
        if not event_meta.empty:
            event_count_by_symbol = event_meta.groupby("symbol").size().sort_values(ascending=False).rename("event_count")

    return {
        "event_meta": event_meta,
        "event_window_df": event_window_df,
        "skip_df": skip_df,
        "path_matrix": path_matrix,
        "forward_summary": forward_summary,
        "event_count_by_symbol": event_count_by_symbol,
    }


def summarize_experiment(experiment_id, param_label, params, result, rank_by_horizon):
    event_meta = result["event_meta"]
    forward_summary = result["forward_summary"]
    event_count_by_symbol = result["event_count_by_symbol"]

    summary = {
        "experiment_id": experiment_id,
        "param_label": param_label,
        "event_count": int(len(event_meta)),
        "symbol_count": int(event_meta["symbol"].nunique()) if not event_meta.empty else 0,
        "skip_symbol_count": int(len(result["skip_df"])),
        "avg_events_per_symbol": float(event_count_by_symbol.mean()) if not event_count_by_symbol.empty else 0.0,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }

    for horizon in FORWARD_HORIZONS:
        horizon_row = forward_summary.loc[forward_summary["horizon"] == horizon]
        if horizon_row.empty:
            summary[f"mean_return_{horizon}d"] = np.nan
            summary[f"median_return_{horizon}d"] = np.nan
            summary[f"win_rate_{horizon}d"] = np.nan
            summary[f"sample_count_{horizon}d"] = 0
            continue

        row = horizon_row.iloc[0]
        summary[f"mean_return_{horizon}d"] = float(row["mean_return"])
        summary[f"median_return_{horizon}d"] = float(row["median_return"])
        summary[f"win_rate_{horizon}d"] = float(row["win_rate"])
        summary[f"sample_count_{horizon}d"] = int(row["sample_count"])

    rank_key = f"mean_return_{rank_by_horizon}d"
    summary["rank_metric"] = float(summary.get(rank_key, np.nan)) if not is_missing(summary.get(rank_key, np.nan)) else np.nan
    return summary

def load_runtime_context():
    with open(PROJECT_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    data_config = settings["data"]
    stock_file = STOCK_FILE or data_config.get("stock_file", "./data/HS300.txt")
    adjust_mode = data_config.get("adjust_mode", "qfq") if ADJUST_MODE is None else ADJUST_MODE

    data_api = DataAPI(
        source=data_config.get("source", "akshare"),
        stock_file=stock_file,
        cache_dir=data_config.get("cache_dir", "./data/raw"),
        processed_dir=data_config.get("processed_dir", "./data/processed"),
        adjust_mode=adjust_mode,
    )

    symbols = list(SYMBOLS) if SYMBOLS is not None else data_api.get_stock_list()
    if MAX_SYMBOLS is not None and MAX_SYMBOLS > 0:
        symbols = symbols[:MAX_SYMBOLS]

    return data_api, stock_file, symbols


def build_runtime_summary(data_api, stock_file, symbols):
    return pd.Series(
        {
            "condition_name": CONDITION_NAME,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "stock_file": stock_file,
            "symbol_count": len(symbols),
            "adjust_mode": data_api.adjust_mode_label,
            "window_before": WINDOW_BEFORE,
            "window_after": WINDOW_AFTER,
            "min_gap_between_events": MIN_GAP_BETWEEN_EVENTS,
            "event_selection": EVENT_SELECTION,
            "require_full_window": REQUIRE_FULL_WINDOW,
            "rank_by_horizon": RANK_BY_HORIZON,
            "top_n_experiments": TOP_N_EXPERIMENTS,
            "min_events_per_experiment": MIN_EVENTS_PER_EXPERIMENT,
            "parameter_combinations": count_parameter_combinations(CONDITION_PARAM_GRID),
            "max_experiments": MAX_EXPERIMENTS,
            "use_multiprocessing": USE_MULTIPROCESSING,
            "max_workers": MAX_WORKERS,
            "process_pool_chunk_size": PROCESS_POOL_CHUNK_SIZE,
        }
    )


def resolve_worker_count(param_sets):
    if not param_sets:
        return 1

    requested_workers = MAX_WORKERS if MAX_WORKERS is not None else (os.cpu_count() or 1)
    return max(1, min(requested_workers, len(param_sets)))


def init_experiment_worker(price_cache, condition_name, rank_by_horizon, min_events_per_experiment):
    global WORKER_PRICE_CACHE
    global WORKER_CONDITION_NAME
    global WORKER_RANK_BY_HORIZON
    global WORKER_MIN_EVENTS_PER_EXPERIMENT

    WORKER_PRICE_CACHE = price_cache
    WORKER_CONDITION_NAME = condition_name
    WORKER_RANK_BY_HORIZON = rank_by_horizon
    WORKER_MIN_EVENTS_PER_EXPERIMENT = min_events_per_experiment


def run_experiment_scan_task(task):
    experiment_id, param_label, params = task
    result = run_single_event_study(
        price_cache=WORKER_PRICE_CACHE,
        condition_name=WORKER_CONDITION_NAME,
        condition_params=params,
    )
    summary_row = summarize_experiment(
        experiment_id=experiment_id,
        param_label=param_label,
        params=params,
        result=result,
        rank_by_horizon=WORKER_RANK_BY_HORIZON,
    )

    qualified_result = None
    if summary_row["event_count"] >= WORKER_MIN_EVENTS_PER_EXPERIMENT:
        qualified_result = {
            "param_label": param_label,
            "params": params,
            **result,
        }

    return {
        "experiment_id": experiment_id,
        "param_label": param_label,
        "params": params,
        "summary_row": summary_row,
        "forward_summary": result["forward_summary"],
        "event_meta": result["event_meta"],
        "qualified_result": qualified_result,
    }


def collect_experiment_output(output, experiment_results, grid_summary_rows, forward_summary_frames, event_meta_frames):
    experiment_id = output["experiment_id"]
    param_label = output["param_label"]

    grid_summary_rows.append(output["summary_row"])

    if not output["forward_summary"].empty:
        frame = output["forward_summary"].copy()
        frame["experiment_id"] = experiment_id
        frame["param_label"] = param_label
        forward_summary_frames.append(frame)

    if not output["event_meta"].empty:
        frame = output["event_meta"].copy()
        frame["experiment_id"] = experiment_id
        frame["param_label"] = param_label
        event_meta_frames.append(frame)

    if output["qualified_result"] is not None:
        experiment_results[experiment_id] = output["qualified_result"]


def scan_parameters_serial(param_sets, price_cache):
    for experiment_id, param_label, params in tqdm(param_sets, desc="Exploring params", unit="experiment"):
        result = run_single_event_study(
            price_cache=price_cache,
            condition_name=CONDITION_NAME,
            condition_params=params,
        )
        summary_row = summarize_experiment(
            experiment_id=experiment_id,
            param_label=param_label,
            params=params,
            result=result,
            rank_by_horizon=RANK_BY_HORIZON,
        )

        qualified_result = None
        if summary_row["event_count"] >= MIN_EVENTS_PER_EXPERIMENT:
            qualified_result = {
                "param_label": param_label,
                "params": params,
                **result,
            }

        yield {
            "experiment_id": experiment_id,
            "param_label": param_label,
            "params": params,
            "summary_row": summary_row,
            "forward_summary": result["forward_summary"],
            "event_meta": result["event_meta"],
            "qualified_result": qualified_result,
        }


def scan_parameters_parallel(param_sets, price_cache):
    worker_count = resolve_worker_count(param_sets)
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=init_experiment_worker,
        initargs=(price_cache, CONDITION_NAME, RANK_BY_HORIZON, MIN_EVENTS_PER_EXPERIMENT),
    ) as executor:
        results = executor.map(
            run_experiment_scan_task,
            param_sets,
            chunksize=PROCESS_POOL_CHUNK_SIZE,
        )
        for output in tqdm(results, total=len(param_sets), desc="Exploring params", unit="experiment"):
            yield output


def scan_parameters(param_sets, price_cache):
    should_use_multiprocessing = USE_MULTIPROCESSING and len(param_sets) > 1
    if should_use_multiprocessing:
        yield from scan_parameters_parallel(param_sets, price_cache)
        return
    yield from scan_parameters_serial(param_sets, price_cache)


def select_best_experiment(grid_summary_df, experiment_results):
    if grid_summary_df.empty:
        raise ValueError("No experiments were completed. Adjust the universe, dates, or parameter grid.")

    ranked_candidates = grid_summary_df.loc[grid_summary_df["event_count"] >= MIN_EVENTS_PER_EXPERIMENT].copy()
    if ranked_candidates.empty:
        raise ValueError("No experiments met the minimum event count threshold. Lower MIN_EVENTS_PER_EXPERIMENT or adjust the grid.")

    best_experiment_id = int(ranked_candidates.iloc[0]["experiment_id"])
    return best_experiment_id, experiment_results[best_experiment_id]


def plot_best_result(best_experiment_id, best_result):
    path_matrix = best_result["path_matrix"]
    if path_matrix.empty:
        raise ValueError("Best experiment has no aligned event paths. Check event window construction.")

    x = path_matrix.columns.to_numpy(dtype=int)
    mean_path = path_matrix.mean(axis=0)
    median_path = path_matrix.median(axis=0)
    p25 = path_matrix.quantile(0.25, axis=0)
    p75 = path_matrix.quantile(0.75, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, mean_path.values, label="mean", linewidth=2)
    ax.plot(x, median_path.values, label="median", linewidth=2)
    ax.fill_between(x, p25.values, p75.values, alpha=0.2, label="25%-75% range")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"Aligned price path after {CONDITION_NAME} | experiment {best_experiment_id}")
    ax.set_xlabel("Relative trading day")
    ax.set_ylabel("Return vs. event close")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.show()

    sample_events = best_result["event_meta"].head(SHOW_SAMPLE_PLOTS)
    if sample_events.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for row in sample_events.itertuples(index=False):
        sample_path = path_matrix.loc[(row.symbol, row.event_date)]
        ax.plot(sample_path.index.to_numpy(dtype=int), sample_path.values, alpha=0.8, label=f"{row.symbol} | {row.event_date:%Y-%m-%d}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"Sample event paths | experiment {best_experiment_id}")
    ax.set_xlabel("Relative trading day")
    ax.set_ylabel("Return vs. event close")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    plt.show()


def export_scan_results(grid_summary_df, forward_summary_all_df, event_meta_all_df, best_result):
    output_dir = PROJECT_ROOT / "output" / "pattern_event_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{CONDITION_NAME}_{START_DATE}_{END_DATE}".replace(":", "-")
    grid_summary_df.to_csv(output_dir / f"grid_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    forward_summary_all_df.to_csv(output_dir / f"grid_forward_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    event_meta_all_df.to_csv(output_dir / f"grid_event_meta_{suffix}.csv", index=False, encoding="utf-8-sig")

    best_result["event_meta"].to_csv(output_dir / f"best_event_meta_{suffix}.csv", index=False, encoding="utf-8-sig")
    best_result["event_window_df"].to_csv(output_dir / f"best_event_windows_{suffix}.csv", index=False, encoding="utf-8-sig")
    best_result["forward_summary"].to_csv(output_dir / f"best_forward_summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    print(f"Exported results to: {output_dir}")


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")

    data_api, stock_file, symbols = load_runtime_context()
    runtime_summary = build_runtime_summary(data_api, stock_file, symbols)
    display(runtime_summary.to_frame("value"))

    price_cache, cache_skip_df = build_price_cache(
        symbols=symbols,
        data_api=data_api,
        start_date=START_DATE,
        end_date=END_DATE,
        min_data_length=MIN_DATA_LENGTH,
    )

    cache_summary = pd.Series(
        {
            "requested_symbols": len(symbols),
            "cached_symbols": len(price_cache),
            "skipped_symbols": len(cache_skip_df),
        }
    )
    display(cache_summary.to_frame("value"))
    display(cache_skip_df.head(10))

    experiment_results = {}
    grid_summary_rows = []
    forward_summary_frames = []
    event_meta_frames = []

    param_sets = list(iter_param_sets(BASE_CONDITION_PARAMS, CONDITION_PARAM_GRID, MAX_EXPERIMENTS))
    print(f"Running {len(param_sets)} experiments...")
    if USE_MULTIPROCESSING and len(param_sets) > 1:
        print(f"Using multiprocessing with {resolve_worker_count(param_sets)} workers.")
    else:
        print("Using serial parameter scan.")

    for output in scan_parameters(param_sets, price_cache):
        collect_experiment_output(
            output,
            experiment_results=experiment_results,
            grid_summary_rows=grid_summary_rows,
            forward_summary_frames=forward_summary_frames,
            event_meta_frames=event_meta_frames,
        )

    grid_summary_df = pd.DataFrame(grid_summary_rows)
    if not grid_summary_df.empty:
        grid_summary_df = grid_summary_df.sort_values(
            by=["rank_metric", "event_count"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    forward_summary_all_df = pd.concat(forward_summary_frames, ignore_index=True) if forward_summary_frames else pd.DataFrame()
    event_meta_all_df = pd.concat(event_meta_frames, ignore_index=True) if event_meta_frames else pd.DataFrame()

    print(f"Experiments with >= {MIN_EVENTS_PER_EXPERIMENT} events: {len(experiment_results)}")
    display(grid_summary_df.head(TOP_N_EXPERIMENTS))
    display(forward_summary_all_df.head(20))

    best_experiment_id, best_result = select_best_experiment(grid_summary_df, experiment_results)

    print(f"Best experiment_id = {best_experiment_id}")
    print(best_result["param_label"])
    display(pd.Series(best_result["params"], name="value").to_frame())
    display(best_result["forward_summary"])
    display(best_result["event_count_by_symbol"].head(20).to_frame())
    display(best_result["event_meta"].head(10))

    if SHOW_RESULTS:
        plot_best_result(best_experiment_id, best_result)

    if EXPORT_RESULTS:
        export_scan_results(grid_summary_df, forward_summary_all_df, event_meta_all_df, best_result)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

# ## What is worth extending next
#
# The current notebook is now suitable for parameter exploration, and the next upgrades usually matter most:
#
# - Reconcile the notebook-only squeeze definition with `signals/timing/bollinger_bands.py` if you want research and backtest logic to match exactly.
# - Replace brute-force grids with staged search: core band parameters first, then breakout confirmation, then filter parameters.
# - Split results by market regime, index trend, or volume expansion to avoid mixing very different environments.
# - Add benchmark-relative returns so the path is not dominated by market beta.
# - Export a per-event label set for later modeling, ranking, or manual chart review.
