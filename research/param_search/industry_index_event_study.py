from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import json
import multiprocessing
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from research.param_search.conditions.registry import (  # noqa: E402
    BASE_CONDITION_PARAMS_MAP,
    CONDITION_PARAM_GRID_MAP,
    build_condition_frame,
)


CONDITION_NAME = "bollinger_squeeze"  # bollinger_squeeze | rsrs_breakout
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

INDEX_LIST_FILE = PROJECT_ROOT / "data" / "raw" / "akshare" / "industry_sector_index_list.csv"
INDEX_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "akshare" / "industry_sector_index"

INDEX_CODES = None  # ["881121", "881273"]
MAX_INDEXES = None

WINDOW_BEFORE = 30
WINDOW_AFTER = 30
MIN_GAP_BETWEEN_EVENTS = 20
EVENT_SELECTION = "first_in_run"  # first_in_run | all_true_days
REQUIRE_FULL_WINDOW = True
MIN_DATA_LENGTH = 120

FORWARD_HORIZONS = [1, 5, 10, 20, 30]
SHOW_SAMPLE_PLOTS = 8
SHOW_RESULTS = False
EXPORT_RESULTS = True
OUTPUT_DIR = PROJECT_ROOT / "output" / "industry_index_event_study"

RANK_BY_HORIZON = 20
TOP_N_EXPERIMENTS = 15
MIN_EVENTS_PER_EXPERIMENT = 6
MAX_EXPERIMENTS = None
USE_MULTIPROCESSING = True
MAX_WORKERS = 10
PROCESS_POOL_CHUNK_SIZE = 1

if CONDITION_NAME not in BASE_CONDITION_PARAMS_MAP:
    supported = ", ".join(sorted(BASE_CONDITION_PARAMS_MAP.keys()))
    raise ValueError(f"Unsupported CONDITION_NAME: {CONDITION_NAME}. Available: {supported}")

BASE_CONDITION_PARAMS = dict(BASE_CONDITION_PARAMS_MAP[CONDITION_NAME])
CONDITION_PARAM_GRID = dict(CONDITION_PARAM_GRID_MAP.get(CONDITION_NAME, {}))


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


WORKER_PRICE_CACHE = None
WORKER_SYMBOL_NAME_MAP = None
WORKER_CONDITION_NAME = None
WORKER_RANK_BY_HORIZON = None
WORKER_MIN_EVENTS_PER_EXPERIMENT = None


def is_missing(value):
    return value is None or value != value


def count_parameter_combinations(param_grid):
    if not param_grid:
        return 1
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def normalize_code(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def standardize_index_list(index_list_df):
    if index_list_df is None or index_list_df.empty:
        return pd.DataFrame(columns=["index_code", "industry_name", "link"])

    df = index_list_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}

    def pick_column(candidates):
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
            match = lower_to_original.get(candidate.lower())
            if match is not None:
                return match
        return None

    code_col = pick_column(["index_code", "code", "industry_code", "板块代码", "行业代码"])
    name_col = pick_column(["industry_name", "name", "board_name", "板块名称", "行业名称"])
    link_col = pick_column(["link", "source_link", "url"])

    if code_col is None or name_col is None:
        raise ValueError("industry_sector_index_list.csv must contain index_code and industry_name columns.")

    result = pd.DataFrame(
        {
            "index_code": df[code_col].map(normalize_code),
            "industry_name": df[name_col].map(normalize_text),
        }
    )
    if link_col is not None:
        result["link"] = df[link_col].map(normalize_text)
    else:
        result["link"] = ""

    result = result[(result["index_code"] != "") & (result["industry_name"] != "")]
    result = result.drop_duplicates(subset=["index_code"]).reset_index(drop=True)
    return result


def load_index_universe(index_list_file, index_codes=None, max_indexes=None):
    list_df = pd.read_csv(index_list_file, encoding="utf-8-sig")
    universe_df = standardize_index_list(list_df)

    if index_codes is not None:
        keep_codes = {normalize_code(code) for code in index_codes if normalize_code(code)}
        universe_df = universe_df[universe_df["index_code"].isin(keep_codes)].reset_index(drop=True)

    if max_indexes is not None and max_indexes > 0:
        universe_df = universe_df.head(max_indexes).reset_index(drop=True)

    return universe_df


def build_index_file_lookup(index_data_dir):
    pattern = re.compile(r"^(?P<code>\d{6})_(?P<start>\d{8})_(?P<end>\d{8})\.csv$")
    best_file_map = {}

    for csv_path in sorted(index_data_dir.glob("*.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        code = match.group("code")
        start_date = match.group("start")
        end_date = match.group("end")

        if code not in best_file_map:
            best_file_map[code] = (start_date, end_date, csv_path)
            continue

        old_start, old_end, _ = best_file_map[code]
        if end_date > old_end or (end_date == old_end and start_date < old_start):
            best_file_map[code] = (start_date, end_date, csv_path)

    return {code: payload[2] for code, payload in best_file_map.items()}


def standardize_index_price_frame(raw_df):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    rename_map = {}
    for column in df.columns:
        name = str(column).strip()
        name_lower = name.lower()

        if name in {"日期", "交易日期"} or name_lower in {"date", "datetime"}:
            rename_map[column] = "date"
        elif name in {"开盘价", "开盘"} or name_lower == "open":
            rename_map[column] = "open"
        elif name in {"最高价", "最高"} or name_lower == "high":
            rename_map[column] = "high"
        elif name in {"最低价", "最低"} or name_lower == "low":
            rename_map[column] = "low"
        elif name in {"收盘价", "收盘"} or name_lower == "close":
            rename_map[column] = "close"
        elif name in {"成交量"} or name_lower in {"volume", "vol"}:
            rename_map[column] = "volume"
        elif name in {"成交额"} or name_lower in {"amount", "turnover"}:
            rename_map[column] = "amount"
        elif name_lower in {"index_code", "code", "symbol"}:
            rename_map[column] = "code"
        elif name_lower in {"industry_name", "name"} or name in {"行业名称", "板块名称"}:
            rename_map[column] = "industry_name"

    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("Price csv is missing date column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_price_cache(universe_df, index_data_dir, start_date, end_date, min_data_length):
    file_lookup = build_index_file_lookup(index_data_dir)
    price_cache = {}
    symbol_name_map = {}
    cache_skip_rows = []

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for row in tqdm(universe_df.itertuples(index=False), total=len(universe_df), desc="Loading index cache", unit="index"):
        code = row.index_code
        industry_name = row.industry_name
        csv_path = file_lookup.get(code)

        if csv_path is None:
            cache_skip_rows.append({"symbol": code, "industry_name": industry_name, "reason": "missing_csv"})
            continue

        try:
            raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            price_df = standardize_index_price_frame(raw_df)
            price_df = price_df[(price_df.index >= start_ts) & (price_df.index <= end_ts)]

            required_columns = {"close", "high", "low"}
            if price_df.empty:
                cache_skip_rows.append({"symbol": code, "industry_name": industry_name, "reason": "empty_price_data"})
                continue
            if len(price_df) < min_data_length:
                cache_skip_rows.append({"symbol": code, "industry_name": industry_name, "reason": "insufficient_history"})
                continue
            if not required_columns.issubset(price_df.columns):
                cache_skip_rows.append({"symbol": code, "industry_name": industry_name, "reason": "missing_required_columns"})
                continue

            price_cache[code] = price_df
            symbol_name_map[code] = industry_name
        except Exception as exc:
            cache_skip_rows.append(
                {
                    "symbol": code,
                    "industry_name": industry_name,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    return price_cache, symbol_name_map, pd.DataFrame(cache_skip_rows)


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


def build_event_window(df, symbol, industry_name, event_date, window_before, window_after, require_full_window=True):
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
    window["industry_name"] = industry_name
    window["event_date"] = pd.Timestamp(event_date)
    window["event_close"] = event_close
    window["close_to_event"] = window["close"] / event_close - 1.0
    return window.reset_index().rename(columns={window.index.name or "index": "trade_date"})


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


def _get_event_value(analysis_df, event_date, column):
    if column not in analysis_df.columns:
        return None
    value = analysis_df.loc[event_date, column]
    if isinstance(value, pd.Series):
        return value.iloc[-1]
    return value


def build_event_meta_row(analysis_df, symbol, industry_name, event_date):
    row = {
        "symbol": symbol,
        "industry_name": industry_name,
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


def run_single_event_study(price_cache, symbol_name_map, condition_name, condition_params):
    event_windows = []
    event_rows = []
    skip_rows = []

    for symbol, price_df in price_cache.items():
        industry_name = symbol_name_map.get(symbol, "")
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
                skip_rows.append({"symbol": symbol, "industry_name": industry_name, "reason": "no_events"})
                continue

            for event_date in event_dates:
                window_df = build_event_window(
                    analysis_df,
                    symbol=symbol,
                    industry_name=industry_name,
                    event_date=event_date,
                    window_before=WINDOW_BEFORE,
                    window_after=WINDOW_AFTER,
                    require_full_window=REQUIRE_FULL_WINDOW,
                )

                if window_df is None:
                    continue

                event_windows.append(window_df)
                event_rows.append(build_event_meta_row(analysis_df, symbol, industry_name, event_date))
        except Exception as exc:
            skip_rows.append(
                {
                    "symbol": symbol,
                    "industry_name": industry_name,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

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
    rank_value = summary.get(rank_key, np.nan)
    summary["rank_metric"] = float(rank_value) if not is_missing(rank_value) else np.nan
    return summary


def resolve_worker_count(param_sets):
    if not param_sets:
        return 1

    requested_workers = MAX_WORKERS if MAX_WORKERS is not None else (os.cpu_count() or 1)
    return max(1, min(requested_workers, len(param_sets)))


def init_experiment_worker(price_cache, symbol_name_map, condition_name, rank_by_horizon, min_events_per_experiment):
    global WORKER_PRICE_CACHE
    global WORKER_SYMBOL_NAME_MAP
    global WORKER_CONDITION_NAME
    global WORKER_RANK_BY_HORIZON
    global WORKER_MIN_EVENTS_PER_EXPERIMENT

    WORKER_PRICE_CACHE = price_cache
    WORKER_SYMBOL_NAME_MAP = symbol_name_map
    WORKER_CONDITION_NAME = condition_name
    WORKER_RANK_BY_HORIZON = rank_by_horizon
    WORKER_MIN_EVENTS_PER_EXPERIMENT = min_events_per_experiment


def run_experiment_scan_task(task):
    experiment_id, param_label, params = task
    result = run_single_event_study(
        price_cache=WORKER_PRICE_CACHE,
        symbol_name_map=WORKER_SYMBOL_NAME_MAP,
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


def scan_parameters_serial(param_sets, price_cache, symbol_name_map):
    for experiment_id, param_label, params in tqdm(param_sets, desc="Exploring params", unit="experiment"):
        result = run_single_event_study(
            price_cache=price_cache,
            symbol_name_map=symbol_name_map,
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


def scan_parameters_parallel(param_sets, price_cache, symbol_name_map):
    worker_count = resolve_worker_count(param_sets)
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=init_experiment_worker,
        initargs=(price_cache, symbol_name_map, CONDITION_NAME, RANK_BY_HORIZON, MIN_EVENTS_PER_EXPERIMENT),
    ) as executor:
        results = executor.map(
            run_experiment_scan_task,
            param_sets,
            chunksize=PROCESS_POOL_CHUNK_SIZE,
        )
        for output in tqdm(results, total=len(param_sets), desc="Exploring params", unit="experiment"):
            yield output


def scan_parameters(param_sets, price_cache, symbol_name_map):
    should_use_multiprocessing = USE_MULTIPROCESSING and len(param_sets) > 1
    if should_use_multiprocessing:
        yield from scan_parameters_parallel(param_sets, price_cache, symbol_name_map)
        return
    yield from scan_parameters_serial(param_sets, price_cache, symbol_name_map)


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
        label = f"{row.symbol} {row.industry_name} | {row.event_date:%Y-%m-%d}"
        ax.plot(sample_path.index.to_numpy(dtype=int), sample_path.values, alpha=0.8, label=label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"Sample event paths | experiment {best_experiment_id}")
    ax.set_xlabel("Relative trading day")
    ax.set_ylabel("Return vs. event close")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    plt.show()


def export_scan_results(grid_summary_df, forward_summary_all_df, event_meta_all_df, best_result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"industry_{CONDITION_NAME}_{START_DATE}_{END_DATE}".replace(":", "-")
    grid_summary_df.to_csv(OUTPUT_DIR / f"grid_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    forward_summary_all_df.to_csv(OUTPUT_DIR / f"grid_forward_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    event_meta_all_df.to_csv(OUTPUT_DIR / f"grid_event_meta_{suffix}.csv", index=False, encoding="utf-8-sig")

    best_result["event_meta"].to_csv(OUTPUT_DIR / f"best_event_meta_{suffix}.csv", index=False, encoding="utf-8-sig")
    best_result["event_window_df"].to_csv(OUTPUT_DIR / f"best_event_windows_{suffix}.csv", index=False, encoding="utf-8-sig")
    best_result["forward_summary"].to_csv(OUTPUT_DIR / f"best_forward_summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    print(f"Exported results to: {OUTPUT_DIR}")


def build_runtime_summary(universe_df):
    return pd.Series(
        {
            "condition_name": CONDITION_NAME,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "index_list_file": str(INDEX_LIST_FILE),
            "index_data_dir": str(INDEX_DATA_DIR),
            "index_count": len(universe_df),
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


def main():
    if not INDEX_LIST_FILE.exists():
        raise FileNotFoundError(f"Industry list file not found: {INDEX_LIST_FILE}")
    if not INDEX_DATA_DIR.exists():
        raise FileNotFoundError(f"Industry index data directory not found: {INDEX_DATA_DIR}")

    universe_df = load_index_universe(
        index_list_file=INDEX_LIST_FILE,
        index_codes=INDEX_CODES,
        max_indexes=MAX_INDEXES,
    )
    if universe_df.empty:
        raise ValueError("No industry indexes found in universe after filters.")

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    display(build_runtime_summary(universe_df).to_frame("value"))
    # display(universe_df.head(12))

    price_cache, symbol_name_map, cache_skip_df = build_price_cache(
        universe_df=universe_df,
        index_data_dir=INDEX_DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE,
        min_data_length=MIN_DATA_LENGTH,
    )

    cache_summary = pd.Series(
        {
            "requested_indexes": len(universe_df),
            "cached_indexes": len(price_cache),
            "skipped_indexes": len(cache_skip_df),
        }
    )
    display(cache_summary.to_frame("value"))
    display(cache_skip_df.head(10))

    param_sets = list(iter_param_sets(BASE_CONDITION_PARAMS, CONDITION_PARAM_GRID, MAX_EXPERIMENTS))
    print(f"Running {len(param_sets)} experiments...")
    if USE_MULTIPROCESSING and len(param_sets) > 1:
        print(f"Using multiprocessing with {resolve_worker_count(param_sets)} workers.")
    else:
        print("Using serial parameter scan.")

    experiment_results = {}
    grid_summary_rows = []
    forward_summary_frames = []
    event_meta_frames = []

    for output in scan_parameters(param_sets, price_cache, symbol_name_map):
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
    # display(best_result["event_meta"].head(10))

    if SHOW_RESULTS:
        plot_best_result(best_experiment_id, best_result)

    if EXPORT_RESULTS:
        export_scan_results(grid_summary_df, forward_summary_all_df, event_meta_all_df, best_result)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
