from __future__ import annotations

import argparse
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
import os
import re
import sys
import time

import pandas as pd


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
    build_condition_frame,
)


CONDITION_NAME = "bollinger_squeeze"
START_DATE = "2025-01-01"
REFRESH_LOOKBACK_DAYS = 120
MIN_DATA_LENGTH = 120

INDEX_LIST_FILE = PROJECT_ROOT / "data" / "raw" / "akshare" / "industry_sector_index_list.csv"
INDEX_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "akshare" / "industry_sector_index"
OUTPUT_DIR = PROJECT_ROOT / "output" / "industry_index_daily_monitor"

INDEX_CODES = None
MAX_INDEXES = None

SLEEP_SECONDS = 2.0
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2.0
PRUNE_OLD_FILES = False

SCHEDULE_TIMES = ("14:30", "19:00")

REPORT_COLUMNS = [
    "run_label",
    "snapshot_time",
    "symbol",
    "industry_name",
    "trade_date",
    "is_today",
    "is_new_signal",
    "event_direction",
    "close",
    "daily_return",
    "volume_ratio",
    "bandwidth",
    "condition_threshold",
    "rsrs_score",
    "rsrs_zscore",
    "condition",
    "breakout_valid",
    "volume_confirmation",
    "trend_long_confirmation",
]


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


def safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def pick_column(df, candidates):
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    return None


def standardize_index_list(index_list_df):
    if index_list_df is None or index_list_df.empty:
        return pd.DataFrame(columns=["index_code", "industry_name", "link"])

    df = index_list_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    code_col = pick_column(df, ["index_code", "code", "industry_code", "板块代码", "行业代码"])
    name_col = pick_column(df, ["industry_name", "name", "board_name", "板块名称", "行业名称"])
    link_col = pick_column(df, ["link", "source_link", "url"])

    if code_col is None or name_col is None:
        raise ValueError("industry_sector_index_list.csv must contain index_code and industry_name columns.")

    result = pd.DataFrame(
        {
            "index_code": df[code_col].map(normalize_code),
            "industry_name": df[name_col].map(normalize_text),
            "link": df[link_col].map(normalize_text) if link_col is not None else "",
        }
    )
    result = result[(result["index_code"] != "") & (result["industry_name"] != "")]
    return result.drop_duplicates(subset=["index_code"]).reset_index(drop=True)


def load_index_universe(index_list_file, index_codes=None, max_indexes=None):
    list_df = pd.read_csv(index_list_file, encoding="utf-8-sig")
    universe_df = standardize_index_list(list_df)

    if index_codes is not None:
        keep_codes = {normalize_code(code) for code in index_codes if normalize_code(code)}
        universe_df = universe_df[universe_df["index_code"].isin(keep_codes)].reset_index(drop=True)

    if max_indexes is not None and max_indexes > 0:
        universe_df = universe_df.head(max_indexes).reset_index(drop=True)

    return universe_df


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
            rename_map[column] = "index_code"
        elif name_lower in {"industry_name", "name"} or name in {"行业名称", "板块名称"}:
            rename_map[column] = "industry_name"
        elif name_lower in {"source_link", "link", "url"}:
            rename_map[column] = "source_link"

    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        raise ValueError("Price data is missing date column.")

    df["date"] = pd.to_datetime(df["date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


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


def to_yyyymmdd(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


def choose_fetch_start(existing_df, default_start_date, refresh_lookback_days):
    default_start = pd.Timestamp(default_start_date)
    if existing_df.empty:
        return default_start

    last_date = pd.Timestamp(existing_df["date"].max())
    refresh_start = last_date - pd.Timedelta(days=max(refresh_lookback_days, 1))
    return min(max(default_start, refresh_start), last_date)


def output_price_frame(df, industry_name, index_code, link):
    out = df.copy()
    out["industry_name"] = industry_name
    out["index_code"] = index_code
    out["source_link"] = link
    out["日期"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    rename_map = {
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "volume": "成交量",
        "amount": "成交额",
    }
    out = out.rename(columns=rename_map)
    columns = [
        "industry_name",
        "index_code",
        "source_link",
        "日期",
        "开盘价",
        "最高价",
        "最低价",
        "收盘价",
        "成交量",
        "成交额",
    ]
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def write_merged_price_file(merged_df, index_data_dir, old_path, industry_name, index_code, link, prune_old_files):
    first_date = to_yyyymmdd(merged_df["date"].min())
    last_date = to_yyyymmdd(merged_df["date"].max())
    target_path = index_data_dir / f"{index_code}_{first_date}_{last_date}.csv"
    output_price_frame(merged_df, industry_name, index_code, link).to_csv(target_path, index=False, encoding="utf-8-sig")

    if prune_old_files:
        for csv_path in index_data_dir.glob(f"{index_code}_*.csv"):
            if csv_path != target_path:
                csv_path.unlink()

    if old_path is not None and old_path != target_path and prune_old_files:
        old_path.unlink(missing_ok=True)

    return target_path


def fetch_index_history(ak, industry_name, start_date, end_date, max_retries, retry_wait_seconds):
    last_error = None
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            hist_df = ak.stock_board_industry_index_ths(
                symbol=industry_name,
                start_date=start_date,
                end_date=end_date,
            )
            if hist_df is None or hist_df.empty:
                raise ValueError("empty dataframe")
            return hist_df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max(max_retries, 1):
                time.sleep(max(retry_wait_seconds, 0.0) * attempt)
    raise RuntimeError(f"failed to fetch {industry_name}: {last_error}") from last_error


def update_industry_data(universe_df, args):
    try:
        import akshare as ak
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("akshare is not installed. Install it first: pip install akshare") from exc

    INDEX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_lookup = build_index_file_lookup(INDEX_DATA_DIR)
    end_date = pd.Timestamp(args.end_date or datetime.now().date())
    update_rows = []

    total = len(universe_df)
    for position, row in enumerate(universe_df.itertuples(index=False), start=1):
        index_code = normalize_code(row.index_code)
        industry_name = normalize_text(row.industry_name)
        link = normalize_text(getattr(row, "link", ""))
        old_path = file_lookup.get(index_code)
        existing_df = pd.DataFrame()

        if old_path is not None:
            existing_df = standardize_index_price_frame(pd.read_csv(old_path, encoding="utf-8-sig"))

        fetch_start = choose_fetch_start(existing_df, args.start_date, args.refresh_lookback_days)
        fetch_start_text = to_yyyymmdd(fetch_start)
        end_date_text = to_yyyymmdd(end_date)

        status = "updated"
        message = ""
        latest_date = None
        output_path = old_path
        try:
            print(f"[{position}/{total}] Fetching {industry_name} ({index_code}) {fetch_start_text}->{end_date_text}")
            fresh_df = fetch_index_history(
                ak=ak,
                industry_name=industry_name,
                start_date=fetch_start_text,
                end_date=end_date_text,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
            )
            fresh_df = standardize_index_price_frame(fresh_df)
            merged_df = pd.concat([existing_df, fresh_df], ignore_index=True)
            merged_df = standardize_index_price_frame(merged_df)
            output_path = write_merged_price_file(
                merged_df=merged_df,
                index_data_dir=INDEX_DATA_DIR,
                old_path=old_path,
                industry_name=industry_name,
                index_code=index_code,
                link=link,
                prune_old_files=args.prune_old_files,
            )
            latest_date = pd.Timestamp(merged_df["date"].max()).date().isoformat()
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            if not existing_df.empty:
                latest_date = pd.Timestamp(existing_df["date"].max()).date().isoformat()

        update_rows.append(
            {
                "symbol": index_code,
                "industry_name": industry_name,
                "status": status,
                "latest_date": latest_date,
                "fetch_start": fetch_start_text,
                "fetch_end": end_date_text,
                "output_path": str(output_path) if output_path is not None else "",
                "message": message,
            }
        )
        time.sleep(max(args.sleep_seconds, 0.0))

    return pd.DataFrame(update_rows)


def build_monitor_rows(universe_df, condition_name, condition_params, run_label, snapshot_time):
    file_lookup = build_index_file_lookup(INDEX_DATA_DIR)
    today = pd.Timestamp(snapshot_time.date())
    monitor_rows = []
    skip_rows = []

    for row in universe_df.itertuples(index=False):
        symbol = normalize_code(row.index_code)
        industry_name = normalize_text(row.industry_name)
        csv_path = file_lookup.get(symbol)

        if csv_path is None:
            skip_rows.append({"symbol": symbol, "industry_name": industry_name, "reason": "missing_csv"})
            continue

        try:
            price_df = standardize_index_price_frame(pd.read_csv(csv_path, encoding="utf-8-sig"))
            price_df = price_df.set_index("date").sort_index()
            if len(price_df) < MIN_DATA_LENGTH:
                skip_rows.append({"symbol": symbol, "industry_name": industry_name, "reason": "insufficient_history"})
                continue

            required_columns = {"close", "high", "low"}
            if not required_columns.issubset(price_df.columns):
                skip_rows.append({"symbol": symbol, "industry_name": industry_name, "reason": "missing_required_columns"})
                continue

            condition_frame = build_condition_frame(price_df, condition_name, condition_params)
            analysis_df = price_df.join(condition_frame)
            latest = analysis_df.dropna(subset=["close"]).iloc[-1]
            latest_date = pd.Timestamp(latest.name)
            previous_condition = False
            if len(analysis_df) >= 2 and "condition" in analysis_df.columns:
                previous_condition = safe_bool(analysis_df["condition"].iloc[-2])

            volume_ratio = None
            if "volume" in analysis_df.columns:
                volume_ma = analysis_df["volume"].rolling(20, min_periods=5).mean()
                if len(volume_ma) and volume_ma.iloc[-1] and not pd.isna(volume_ma.iloc[-1]):
                    volume_ratio = safe_float(analysis_df["volume"].iloc[-1] / volume_ma.iloc[-1])

            row_payload = {
                "run_label": run_label,
                "snapshot_time": snapshot_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "industry_name": industry_name,
                "trade_date": latest_date.date().isoformat(),
                "is_today": bool(latest_date.normalize() == today),
                "is_new_signal": safe_bool(latest.get("condition")) and not previous_condition,
                "event_direction": normalize_text(latest.get("event_direction")),
                "close": safe_float(latest.get("close")),
                "daily_return": safe_float(analysis_df["close"].pct_change().iloc[-1]),
                "volume_ratio": volume_ratio,
            }
            boolean_columns = {
                "condition",
                "breakout_valid",
                "volume_confirmation",
                "trend_long_confirmation",
            }
            for column in [
                "bandwidth",
                "condition_threshold",
                "rsrs_score",
                "rsrs_zscore",
                "condition",
                "breakout_valid",
                "volume_confirmation",
                "trend_long_confirmation",
            ]:
                value = latest.get(column)
                row_payload[column] = safe_bool(value) if column in boolean_columns else safe_float(value)

            monitor_rows.append(row_payload)
        except Exception as exc:  # noqa: BLE001
            skip_rows.append(
                {
                    "symbol": symbol,
                    "industry_name": industry_name,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    monitor_df = pd.DataFrame(monitor_rows)
    skip_df = pd.DataFrame(skip_rows)
    if not monitor_df.empty:
        monitor_df = monitor_df.sort_values(
            by=["is_today", "is_new_signal", "condition", "daily_return", "volume_ratio"],
            ascending=[False, False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        monitor_df = monitor_df[[column for column in REPORT_COLUMNS if column in monitor_df.columns]]
    return monitor_df, skip_df


def export_outputs(update_df, monitor_df, skip_df, run_label, snapshot_time):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{snapshot_time:%Y%m%d_%H%M}_{run_label}"

    update_path = OUTPUT_DIR / f"industry_update_{suffix}.csv"
    monitor_path = OUTPUT_DIR / f"industry_opportunities_{suffix}.csv"
    skip_path = OUTPUT_DIR / f"industry_monitor_skip_{suffix}.csv"
    latest_path = OUTPUT_DIR / "latest_industry_opportunities.csv"

    update_df.to_csv(update_path, index=False, encoding="utf-8-sig")
    monitor_df.to_csv(monitor_path, index=False, encoding="utf-8-sig")
    monitor_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    skip_df.to_csv(skip_path, index=False, encoding="utf-8-sig")

    print(f"Saved update log: {update_path}")
    print(f"Saved monitor report: {monitor_path}")
    print(f"Saved latest report: {latest_path}")
    if not skip_df.empty:
        print(f"Saved skipped symbols: {skip_path}")


def parse_schedule_time(text):
    hour, minute = text.split(":", 1)
    return datetime_time(hour=int(hour), minute=int(minute))


def next_run_datetime(now, schedule_times):
    candidates = []
    parsed_times = [parse_schedule_time(value) for value in schedule_times]
    for day_offset in (0, 1):
        base_date = now.date() + timedelta(days=day_offset)
        for run_time in parsed_times:
            candidate = datetime.combine(base_date, run_time)
            if candidate > now:
                candidates.append(candidate)
    return min(candidates)


def infer_run_label(snapshot_time):
    current = snapshot_time.time()
    if current < datetime_time(16, 0):
        return "1430_intraday"
    return "1900_close"


def run_once(args):
    if args.condition_name not in BASE_CONDITION_PARAMS_MAP:
        raise ValueError(f"Unsupported condition: {args.condition_name}")
    if not INDEX_LIST_FILE.exists():
        raise FileNotFoundError(f"Industry list file not found: {INDEX_LIST_FILE}")

    universe_df = load_index_universe(
        index_list_file=INDEX_LIST_FILE,
        index_codes=args.index_codes or INDEX_CODES,
        max_indexes=args.max_indexes if args.max_indexes is not None else MAX_INDEXES,
    )
    if universe_df.empty:
        raise ValueError("No industry indexes found in universe after filters.")

    snapshot_time = datetime.now()
    run_label = args.run_label or infer_run_label(snapshot_time)
    print(f"Run label: {run_label}")
    print(f"Universe size: {len(universe_df)}")

    update_df = update_industry_data(universe_df, args)
    monitor_df, skip_df = build_monitor_rows(
        universe_df=universe_df,
        condition_name=args.condition_name,
        condition_params=dict(BASE_CONDITION_PARAMS_MAP[args.condition_name]),
        run_label=run_label,
        snapshot_time=snapshot_time,
    )

    export_outputs(update_df, monitor_df, skip_df, run_label, snapshot_time)
    print("")
    print("Top opportunities:")
    if monitor_df.empty:
        print("No monitor rows generated.")
    else:
        print(monitor_df.head(args.show_top).to_string(index=False))


def run_daemon(args):
    print(f"Scheduler started. Daily run times: {', '.join(args.schedule_times)}")
    while True:
        now = datetime.now()
        next_run = next_run_datetime(now, args.schedule_times)
        sleep_seconds = max((next_run - now).total_seconds(), 0.0)
        print(f"Next run: {next_run:%Y-%m-%d %H:%M:%S}")
        time.sleep(sleep_seconds)
        args.run_label = infer_run_label(datetime.now())
        run_once(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update THS industry index data and monitor condition changes."
    )
    parser.add_argument("--daemon", action="store_true", help="Keep running and execute at scheduled times.")
    parser.add_argument("--schedule-times", nargs="+", default=list(SCHEDULE_TIMES), help="Daily HH:MM run times.")
    parser.add_argument("--run-label", default="", help="Label for this run, e.g. 1430_intraday or 1900_close.")
    parser.add_argument("--condition-name", default=CONDITION_NAME, choices=sorted(BASE_CONDITION_PARAMS_MAP.keys()))
    parser.add_argument("--start-date", default=START_DATE, help="Default full-history start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Fetch end date, default today.")
    parser.add_argument("--refresh-lookback-days", type=int, default=REFRESH_LOOKBACK_DAYS)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-wait-seconds", type=float, default=RETRY_WAIT_SECONDS)
    parser.add_argument("--prune-old-files", action="store_true", default=PRUNE_OLD_FILES)
    parser.add_argument("--index-codes", nargs="+", default=None, help="Optional industry index code filter.")
    parser.add_argument("--max-indexes", type=int, default=None, help="Optional max industries for testing.")
    parser.add_argument("--show-top", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.daemon:
        run_daemon(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
