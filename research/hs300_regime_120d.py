"""
Classify A-share index bull/bear/sideways regimes with 120-day returns.

Default data source is AkShare. The script fetches HS300, CSI500, CSI1000,
CSI All Share, and ChiNext daily index data, calculates 120-trading-day
returns, labels each trading day, compresses daily labels into stages, and
exports cross-index comparison reports.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INDEXES = [
    ("000300", "沪深300"),
    ("000905", "中证500"),
    ("000852", "中证1000"),
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_BULL_THRESHOLD = 0.05
DEFAULT_BEAR_THRESHOLD = -0.05
DEFAULT_SMOOTH_DAYS = 5
DEFAULT_MIN_SEGMENT_DAYS = 20
REGIME_DIRECTION = {
    "bull": "牛市",
    "bear": "熊市",
    "sideways": "震荡",
    "mixed": "分化",
    "unknown": "样本不足",
}


def parse_date_text(date_text: str) -> pd.Timestamp:
    parsed = pd.to_datetime(date_text, errors="coerce")
    if parsed != parsed:
        raise ValueError(f"Invalid date: {date_text}")
    return parsed.normalize()


def compact_date(date_text: str) -> str:
    return parse_date_text(date_text).strftime("%Y%m%d")


def log_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def split_csv_text(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def normalize_index_code(index_code: str) -> str:
    code = str(index_code).strip().lower()
    code = code.replace("sh", "").replace("sz", "")
    if "." in code:
        code = code.split(".", 1)[0]
    return code


def parse_index_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.index_code:
        return [(normalize_index_code(args.index_code), args.index_name or normalize_index_code(args.index_code))]

    if args.index_codes:
        codes = split_csv_text(args.index_codes)
        names = split_csv_text(args.index_names) if args.index_names else []
        if names and len(names) != len(codes):
            raise ValueError("--index-names count must match --index-codes count.")
        if not names:
            name_lookup = {code: name for code, name in DEFAULT_INDEXES}
            names = [name_lookup.get(normalize_index_code(code), normalize_index_code(code)) for code in codes]
        return [(normalize_index_code(code), name) for code, name in zip(codes, names)]

    return DEFAULT_INDEXES.copy()


def parse_input_csv_map(text: str) -> Dict[str, Path]:
    result = {}
    for item in split_csv_text(text):
        if "=" not in item:
            raise ValueError("--input-csv-map items must use code=path format.")
        code, path = item.split("=", 1)
        result[normalize_index_code(code)] = Path(path.strip())
    return result


def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def standardize_index_frame(data: pd.DataFrame, index_code: str, index_name: str) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("Index data is empty.")

    frame = data.copy()
    date_column = find_first_column(frame.columns, ["date", "日期", "trade_date", "交易日期"])
    close_column = find_first_column(frame.columns, ["close", "收盘", "收盘价"])
    open_column = find_first_column(frame.columns, ["open", "开盘", "开盘价"])
    high_column = find_first_column(frame.columns, ["high", "最高", "最高价"])
    low_column = find_first_column(frame.columns, ["low", "最低", "最低价"])
    volume_column = find_first_column(frame.columns, ["volume", "成交量"])
    amount_column = find_first_column(frame.columns, ["amount", "成交额"])

    if date_column is None or close_column is None:
        raise ValueError(
            "Index data must contain date and close columns. "
            f"Found columns: {list(frame.columns)}"
        )

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    result["index_code"] = normalize_index_code(index_code)
    result["index_name"] = index_name
    result["open"] = pd.to_numeric(frame[open_column], errors="coerce") if open_column else np.nan
    result["close"] = pd.to_numeric(frame[close_column], errors="coerce")
    result["high"] = pd.to_numeric(frame[high_column], errors="coerce") if high_column else np.nan
    result["low"] = pd.to_numeric(frame[low_column], errors="coerce") if low_column else np.nan
    result["volume"] = pd.to_numeric(frame[volume_column], errors="coerce") if volume_column else np.nan
    result["amount"] = pd.to_numeric(frame[amount_column], errors="coerce") if amount_column else np.nan

    result = result.dropna(subset=["date", "close"])
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if result.empty:
        raise ValueError("Index data has no valid date/close rows.")
    return result.reset_index(drop=True)


def to_akshare_daily_symbol(index_code: str) -> str:
    code = normalize_index_code(index_code)
    if code.startswith("399"):
        return f"sz{code}"
    return f"sh{code}"


def fetch_index_data_from_akshare(
    index_code: str,
    index_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "akshare is required to fetch index data. "
            "Install it with `pip install akshare`, or provide --input-csv/--input-csv-map."
        ) from exc

    start = compact_date(start_date)
    end = compact_date(end_date)
    pure_code = normalize_index_code(index_code)

    if hasattr(ak, "index_zh_a_hist"):
        try:
            raw = ak.index_zh_a_hist(
                symbol=pure_code,
                period="daily",
                start_date=start,
                end_date=end,
            )
            if raw is not None and not raw.empty:
                return standardize_index_frame(raw, pure_code, index_name)
        except TypeError:
            raw = ak.index_zh_a_hist(symbol=pure_code, period="daily")
            if raw is not None and not raw.empty:
                data = standardize_index_frame(raw, pure_code, index_name)
                start_ts = parse_date_text(start_date)
                end_ts = parse_date_text(end_date)
                return data[pd.to_datetime(data["date"]).between(start_ts, end_ts, inclusive="both")]
        except Exception as exc:
            log_step(f"ak.index_zh_a_hist failed for {pure_code}, fallback to stock_zh_index_daily: {exc}")

    if not hasattr(ak, "stock_zh_index_daily"):
        raise RuntimeError("akshare does not expose index_zh_a_hist or stock_zh_index_daily.")

    daily_symbol = to_akshare_daily_symbol(pure_code)
    raw = ak.stock_zh_index_daily(symbol=daily_symbol)
    data = standardize_index_frame(raw, pure_code, index_name)
    start_ts = parse_date_text(start_date)
    end_ts = parse_date_text(end_date)
    return data[pd.to_datetime(data["date"]).between(start_ts, end_ts, inclusive="both")]


def load_single_index_data(
    index_code: str,
    index_name: str,
    args: argparse.Namespace,
    input_csv_map: Dict[str, Path],
) -> pd.DataFrame:
    code = normalize_index_code(index_code)
    if code in input_csv_map:
        input_path = input_csv_map[code]
        log_step(f"Reading {index_name}({code}) input csv: {input_path}")
        return standardize_index_frame(read_csv_with_fallback(input_path), code, index_name)

    cache_dir = PROJECT_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}_{compact_date(args.start)}_{compact_date(args.end)}.csv"
    if cache_path.exists() and not args.refresh:
        log_step(f"Reading cached {index_name}({code}) data: {cache_path}")
        return standardize_index_frame(read_csv_with_fallback(cache_path), code, index_name)

    log_step(f"Fetching {index_name}({code}) daily data from AkShare")
    data = fetch_index_data_from_akshare(code, index_name, args.start, args.end)
    if data.empty:
        raise ValueError(f"AkShare returned empty data for {index_name}({code}).")
    data.to_csv(cache_path, index=False, encoding="utf-8-sig")
    log_step(f"Saved index cache: {cache_path}")
    return data


def load_index_data(index_pairs: List[Tuple[str, str]], args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    input_csv_map = parse_input_csv_map(args.input_csv_map) if args.input_csv_map else {}
    if args.input_csv:
        if len(index_pairs) != 1:
            raise ValueError("--input-csv can only be used with one index. Use --input-csv-map for multiple indexes.")
        input_csv_map[normalize_index_code(index_pairs[0][0])] = Path(args.input_csv)

    result = {}
    for code, name in index_pairs:
        data = load_single_index_data(code, name, args, input_csv_map)
        start = parse_date_text(args.start)
        end = parse_date_text(args.end)
        data = data[pd.to_datetime(data["date"]).between(start, end, inclusive="both")].copy()
        if len(data) <= args.lookback_days:
            raise ValueError(
                f"Not enough rows for {name}({code}) {args.lookback_days}-day returns: got {len(data)} rows."
            )
        result[normalize_index_code(code)] = data
    return result


def label_regime(return_value: float, bull_threshold: float, bear_threshold: float) -> str:
    if return_value != return_value:
        return "unknown"
    if return_value >= bull_threshold:
        return "bull"
    if return_value <= bear_threshold:
        return "bear"
    return "sideways"


def smooth_regime_labels(labels: pd.Series, smooth_days: int) -> pd.Series:
    if smooth_days <= 1:
        return labels.copy()

    smoothed = []
    values = labels.tolist()
    for index in range(len(values)):
        window_values = values[max(0, index - smooth_days + 1) : index + 1]
        clean_values = [value for value in window_values if value != "unknown"]
        if not clean_values:
            smoothed.append("unknown")
            continue
        counts = pd.Series(clean_values).value_counts()
        smoothed.append(str(counts.index[0]))
    return pd.Series(smoothed, index=labels.index)


def calculate_daily_regime(
    data: pd.DataFrame,
    lookback_days: int,
    bull_threshold: float,
    bear_threshold: float,
    smooth_days: int,
) -> pd.DataFrame:
    frame = data.copy().sort_values("date").reset_index(drop=True)
    return_column = f"return_{lookback_days}d"
    ma_column = f"ma_{lookback_days}d"
    above_ma_column = f"above_ma_{lookback_days}d"

    frame[return_column] = frame["close"] / frame["close"].shift(lookback_days) - 1.0
    frame[ma_column] = frame["close"].rolling(lookback_days).mean()
    frame[above_ma_column] = frame["close"] >= frame[ma_column]
    frame["daily_return"] = frame["close"].pct_change()
    frame["raw_regime"] = frame[return_column].apply(
        lambda value: label_regime(value, bull_threshold, bear_threshold)
    )
    frame["regime"] = smooth_regime_labels(frame["raw_regime"], smooth_days)
    frame["direction"] = frame["regime"].map(REGIME_DIRECTION)
    return frame


def max_drawdown(close: pd.Series) -> float:
    values = close.astype(float).dropna()
    if values.empty:
        return np.nan
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def annualized_return(total_return: float, trading_days: int) -> float:
    if total_return != total_return or trading_days <= 0 or total_return <= -1:
        return np.nan
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def build_raw_segments(daily: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    rows = []
    if daily.empty:
        return pd.DataFrame(rows)

    return_column = f"return_{lookback_days}d"
    segment_id = (daily["regime"] != daily["regime"].shift()).cumsum()
    for _, group in daily.groupby(segment_id):
        regime = str(group["regime"].iloc[0])
        if regime == "unknown":
            continue
        start_close = float(group["close"].iloc[0])
        end_close = float(group["close"].iloc[-1])
        rows.append(
            {
                "index_code": str(group["index_code"].iloc[0]),
                "index_name": str(group["index_name"].iloc[0]),
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "regime": regime,
                "direction": str(group["direction"].iloc[0]),
                "trading_days": int(len(group)),
                "start_close": start_close,
                "end_close": end_close,
                "period_return": end_close / start_close - 1.0 if start_close else np.nan,
                "mean_120d_return": float(group[return_column].mean()),
                "start_120d_return": float(group[return_column].iloc[0]),
                "end_120d_return": float(group[return_column].iloc[-1]),
                "annualized_volatility": float(group["daily_return"].std() * np.sqrt(252)),
                "max_drawdown": max_drawdown(group["close"]),
            }
        )
    return pd.DataFrame(rows)


def merge_short_segments(segments: pd.DataFrame, min_segment_days: int) -> pd.DataFrame:
    if segments.empty or min_segment_days <= 1:
        return segments.copy()

    merged = segments.copy().reset_index(drop=True)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, row in merged.iterrows():
            if int(row["trading_days"]) >= min_segment_days:
                continue

            if index == 0:
                target = 1
            elif index == len(merged) - 1:
                target = index - 1
            else:
                previous_days = int(merged.loc[index - 1, "trading_days"])
                next_days = int(merged.loc[index + 1, "trading_days"])
                target = index - 1 if previous_days >= next_days else index + 1

            merged.loc[index, "regime"] = merged.loc[target, "regime"]
            merged.loc[index, "direction"] = merged.loc[target, "direction"]
            changed = True
            break

        if changed:
            parts = []
            group_id = (merged["regime"] != merged["regime"].shift()).cumsum()
            for _, group in merged.groupby(group_id):
                first = group.iloc[0].copy()
                first["start_date"] = group["start_date"].iloc[0]
                first["end_date"] = group["end_date"].iloc[-1]
                first["trading_days"] = int(group["trading_days"].sum())
                first["start_close"] = float(group["start_close"].iloc[0])
                first["end_close"] = float(group["end_close"].iloc[-1])
                first["period_return"] = first["end_close"] / first["start_close"] - 1.0
                first["mean_120d_return"] = float(
                    np.average(group["mean_120d_return"], weights=group["trading_days"])
                )
                first["start_120d_return"] = float(group["start_120d_return"].iloc[0])
                first["end_120d_return"] = float(group["end_120d_return"].iloc[-1])
                first["annualized_volatility"] = np.nan
                first["max_drawdown"] = np.nan
                parts.append(first)
            merged = pd.DataFrame(parts).reset_index(drop=True)

    return merged


def recalculate_segment_risk(segments: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return segments

    output = segments.copy()
    for index, row in output.iterrows():
        start = pd.to_datetime(row["start_date"])
        end = pd.to_datetime(row["end_date"])
        mask = pd.to_datetime(daily["date"]).between(start, end, inclusive="both")
        subset = daily.loc[mask]
        output.loc[index, "max_drawdown"] = max_drawdown(subset["close"])
        output.loc[index, "annualized_volatility"] = float(subset["daily_return"].std() * np.sqrt(252))
    return output


def build_segments(daily: pd.DataFrame, lookback_days: int, min_segment_days: int) -> pd.DataFrame:
    segments = build_raw_segments(daily, lookback_days)
    segments = merge_short_segments(segments, min_segment_days)
    segments = recalculate_segment_risk(segments, daily)
    if segments.empty:
        return segments

    segments = segments.reset_index(drop=True)
    segments.insert(0, "stage_id", np.arange(1, len(segments) + 1))
    for column in ("start_date", "end_date"):
        segments[column] = pd.to_datetime(segments[column]).dt.strftime("%Y-%m-%d")
    return segments


def build_all_daily_and_segments(
    index_data: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames = []
    segment_frames = []
    for code, data in index_data.items():
        daily = calculate_daily_regime(
            data,
            lookback_days=args.lookback_days,
            bull_threshold=args.bull_threshold,
            bear_threshold=args.bear_threshold,
            smooth_days=args.smooth_days,
        )
        segments = build_segments(daily, args.lookback_days, args.min_segment_days)
        daily_frames.append(daily)
        segment_frames.append(segments)

    all_daily = pd.concat(daily_frames, ignore_index=True).sort_values(["index_code", "date"])
    all_segments = pd.concat(segment_frames, ignore_index=True).sort_values(["index_code", "stage_id"])
    return all_daily.reset_index(drop=True), all_segments.reset_index(drop=True)


def build_index_summary(all_daily: pd.DataFrame, all_segments: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    rows = []
    return_column = f"return_{lookback_days}d"
    for (code, name), group in all_daily.groupby(["index_code", "index_name"], sort=False):
        group = group.sort_values("date")
        start_close = float(group["close"].iloc[0])
        end_close = float(group["close"].iloc[-1])
        total_return = end_close / start_close - 1.0 if start_close else np.nan
        valid = group[group["regime"] != "unknown"]
        regime_days = valid["regime"].value_counts().to_dict()
        segment_slice = all_segments[all_segments["index_code"].astype(str) == str(code)]
        rows.append(
            {
                "index_code": code,
                "index_name": name,
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "trading_days": int(len(group)),
                "valid_regime_days": int(len(valid)),
                "start_close": start_close,
                "end_close": end_close,
                "total_return": float(total_return),
                "annualized_return": annualized_return(total_return, len(group)),
                "annualized_volatility": float(group["daily_return"].std() * np.sqrt(252)),
                "max_drawdown": max_drawdown(group["close"]),
                "mean_120d_return": float(group[return_column].mean()),
                "median_120d_return": float(group[return_column].median()),
                "bull_days": int(regime_days.get("bull", 0)),
                "bear_days": int(regime_days.get("bear", 0)),
                "sideways_days": int(regime_days.get("sideways", 0)),
                "bull_day_ratio": float(regime_days.get("bull", 0) / len(valid)) if len(valid) else np.nan,
                "bear_day_ratio": float(regime_days.get("bear", 0) / len(valid)) if len(valid) else np.nan,
                "sideways_day_ratio": float(regime_days.get("sideways", 0) / len(valid)) if len(valid) else np.nan,
                "stage_count": int(len(segment_slice)),
                "last_regime": str(valid["regime"].iloc[-1]) if not valid.empty else "unknown",
                "last_direction": str(valid["direction"].iloc[-1]) if not valid.empty else REGIME_DIRECTION["unknown"],
            }
        )
    return pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)


def choose_majority_regime(counts: Dict[str, int]) -> str:
    if not counts:
        return "unknown"
    max_count = max(counts.values())
    winners = [regime for regime, count in counts.items() if count == max_count]
    if len(winners) == 1:
        return winners[0]
    return "mixed"


def build_consensus_daily(all_daily: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    rows = []
    return_column = f"return_{lookback_days}d"
    for date, group in all_daily.groupby("date"):
        valid = group[group["regime"] != "unknown"].copy()
        counts = valid["regime"].value_counts().to_dict()
        majority_regime = choose_majority_regime(counts)
        available_count = int(len(valid))
        top_row = valid.loc[valid[return_column].idxmax()] if available_count else None
        weak_row = valid.loc[valid[return_column].idxmin()] if available_count else None
        rows.append(
            {
                "date": date,
                "available_index_count": available_count,
                "bull_count": int(counts.get("bull", 0)),
                "bear_count": int(counts.get("bear", 0)),
                "sideways_count": int(counts.get("sideways", 0)),
                "bull_ratio": float(counts.get("bull", 0) / available_count) if available_count else np.nan,
                "bear_ratio": float(counts.get("bear", 0) / available_count) if available_count else np.nan,
                "sideways_ratio": float(counts.get("sideways", 0) / available_count) if available_count else np.nan,
                "majority_regime": majority_regime,
                "majority_direction": REGIME_DIRECTION.get(majority_regime, majority_regime),
                "dominant_120d_index_code": str(top_row["index_code"]) if top_row is not None else "",
                "dominant_120d_index_name": str(top_row["index_name"]) if top_row is not None else "",
                "dominant_120d_return": float(top_row[return_column]) if top_row is not None else np.nan,
                "weakest_120d_index_code": str(weak_row["index_code"]) if weak_row is not None else "",
                "weakest_120d_index_name": str(weak_row["index_name"]) if weak_row is not None else "",
                "weakest_120d_return": float(weak_row[return_column]) if weak_row is not None else np.nan,
                "return_120d_spread": float(top_row[return_column] - weak_row[return_column])
                if top_row is not None and weak_row is not None
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_consensus_segments(consensus_daily: pd.DataFrame, min_segment_days: int) -> pd.DataFrame:
    valid = consensus_daily[consensus_daily["majority_regime"] != "unknown"].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["direction"] = valid["majority_regime"].map(REGIME_DIRECTION)
    segment_id = (valid["majority_regime"] != valid["majority_regime"].shift()).cumsum()
    rows = []
    for _, group in valid.groupby(segment_id):
        dominant_counts = group["dominant_120d_index_name"].replace("", np.nan).dropna().value_counts()
        rows.append(
            {
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "regime": str(group["majority_regime"].iloc[0]),
                "direction": str(group["direction"].iloc[0]),
                "trading_days": int(len(group)),
                "mean_bull_ratio": float(group["bull_ratio"].mean()),
                "mean_bear_ratio": float(group["bear_ratio"].mean()),
                "mean_sideways_ratio": float(group["sideways_ratio"].mean()),
                "mean_120d_spread": float(group["return_120d_spread"].mean()),
                "dominant_index_by_days": str(dominant_counts.index[0]) if not dominant_counts.empty else "",
            }
        )

    segments = pd.DataFrame(rows)
    if min_segment_days > 1:
        segments = merge_consensus_short_segments(segments, min_segment_days)
    segments = segments.reset_index(drop=True)
    segments.insert(0, "stage_id", np.arange(1, len(segments) + 1))
    for column in ("start_date", "end_date"):
        segments[column] = pd.to_datetime(segments[column]).dt.strftime("%Y-%m-%d")
    return segments


def merge_consensus_short_segments(segments: pd.DataFrame, min_segment_days: int) -> pd.DataFrame:
    merged = segments.copy().reset_index(drop=True)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, row in merged.iterrows():
            if int(row["trading_days"]) >= min_segment_days:
                continue
            if index == 0:
                target = 1
            elif index == len(merged) - 1:
                target = index - 1
            else:
                previous_days = int(merged.loc[index - 1, "trading_days"])
                next_days = int(merged.loc[index + 1, "trading_days"])
                target = index - 1 if previous_days >= next_days else index + 1
            merged.loc[index, "regime"] = merged.loc[target, "regime"]
            merged.loc[index, "direction"] = merged.loc[target, "direction"]
            changed = True
            break

        if changed:
            parts = []
            group_id = (merged["regime"] != merged["regime"].shift()).cumsum()
            for _, group in merged.groupby(group_id):
                first = group.iloc[0].copy()
                first["start_date"] = group["start_date"].iloc[0]
                first["end_date"] = group["end_date"].iloc[-1]
                first["trading_days"] = int(group["trading_days"].sum())
                for column in ("mean_bull_ratio", "mean_bear_ratio", "mean_sideways_ratio", "mean_120d_spread"):
                    first[column] = float(np.average(group[column], weights=group["trading_days"]))
                dominant_counts = group["dominant_index_by_days"].replace("", np.nan).dropna().value_counts()
                first["dominant_index_by_days"] = str(dominant_counts.index[0]) if not dominant_counts.empty else ""
                parts.append(first)
            merged = pd.DataFrame(parts).reset_index(drop=True)
    return merged


def summarize_outputs(
    all_daily: pd.DataFrame,
    all_segments: pd.DataFrame,
    index_summary: pd.DataFrame,
    consensus_daily: pd.DataFrame,
    consensus_segments: pd.DataFrame,
    index_pairs: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    consensus_valid = consensus_daily[consensus_daily["majority_regime"] != "unknown"]
    return {
        "index_count": len(index_pairs),
        "indexes": [{"index_code": code, "index_name": name} for code, name in index_pairs],
        "start": args.start,
        "end": args.end,
        "lookback_days": int(args.lookback_days),
        "bull_threshold": float(args.bull_threshold),
        "bear_threshold": float(args.bear_threshold),
        "smooth_days": int(args.smooth_days),
        "min_segment_days": int(args.min_segment_days),
        "daily_rows": int(len(all_daily)),
        "single_index_stage_count": int(len(all_segments)),
        "consensus_stage_count": int(len(consensus_segments)),
        "consensus_regime_days": {
            str(regime): int(days)
            for regime, days in consensus_valid["majority_regime"].value_counts().items()
        },
        "best_total_return_index": index_summary.iloc[0][["index_code", "index_name", "total_return"]].to_dict()
        if not index_summary.empty
        else {},
        "worst_total_return_index": index_summary.iloc[-1][["index_code", "index_name", "total_return"]].to_dict()
        if not index_summary.empty
        else {},
    }


def export_outputs(
    all_daily: pd.DataFrame,
    all_segments: pd.DataFrame,
    index_summary: pd.DataFrame,
    consensus_daily: pd.DataFrame,
    consensus_segments: pd.DataFrame,
    index_pairs: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / args.output_dir / f"a_share_index_regime_120d_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_frames = {
        "index_daily_regime.csv": all_daily,
        "index_regime_stages.csv": all_segments,
        "index_comparison_summary.csv": index_summary,
        "consensus_daily_regime.csv": consensus_daily,
        "consensus_regime_stages.csv": consensus_segments,
    }
    for filename, frame in export_frames.items():
        export_frame = frame.copy()
        for column in ("date", "start_date", "end_date"):
            if column in export_frame.columns:
                export_frame[column] = pd.to_datetime(export_frame[column]).dt.strftime("%Y-%m-%d")
        path = output_dir / filename
        export_frame.to_csv(path, index=False, encoding="utf-8-sig")
        log_step(f"Saved {filename}: {path}")

    summary = summarize_outputs(
        all_daily,
        all_segments,
        index_summary,
        consensus_daily,
        consensus_segments,
        index_pairs,
        args,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"Saved summary.json: {summary_path}")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use 120-day returns to classify bull/bear/sideways regimes for multiple A-share indexes."
    )
    parser.add_argument(
        "--index-codes",
        default=",".join(code for code, _ in DEFAULT_INDEXES),
        help="Comma-separated index codes. Default: 000300,000905,000852,000985,399006.",
    )
    parser.add_argument(
        "--index-names",
        default=",".join(name for _, name in DEFAULT_INDEXES),
        help="Comma-separated index names matching --index-codes.",
    )
    parser.add_argument("--index-code", default="", help="Single-index compatibility option.")
    parser.add_argument("--index-name", default="", help="Single-index display name.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date, default: 2018-01-01.")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="End date, default: 2025-12-31.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Return lookback days.")
    parser.add_argument("--bull-threshold", type=float, default=DEFAULT_BULL_THRESHOLD, help="Bull threshold.")
    parser.add_argument("--bear-threshold", type=float, default=DEFAULT_BEAR_THRESHOLD, help="Bear threshold.")
    parser.add_argument("--smooth-days", type=int, default=DEFAULT_SMOOTH_DAYS, help="Rolling mode smoothing days.")
    parser.add_argument("--min-segment-days", type=int, default=DEFAULT_MIN_SEGMENT_DAYS, help="Merge shorter stages.")
    parser.add_argument("--input-csv", default="", help="Optional local csv path for one index.")
    parser.add_argument(
        "--input-csv-map",
        default="",
        help="Optional csv map for multiple indexes, for example 000300=a.csv,000905=b.csv.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/akshare/index_history",
        help="Index cache directory relative to project root.",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory relative to project root.")
    parser.add_argument("--refresh", action="store_true", help="Refresh AkShare cache.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.lookback_days <= 0:
        raise ValueError("--lookback-days must be positive.")
    if args.bear_threshold >= args.bull_threshold:
        raise ValueError("--bear-threshold must be smaller than --bull-threshold.")

    index_pairs = parse_index_pairs(args)
    index_data = load_index_data(index_pairs, args)
    all_daily, all_segments = build_all_daily_and_segments(index_data, args)
    index_summary = build_index_summary(all_daily, all_segments, args.lookback_days)
    consensus_daily = build_consensus_daily(all_daily, args.lookback_days)
    consensus_segments = build_consensus_segments(consensus_daily, args.min_segment_days)
    output_dir = export_outputs(
        all_daily,
        all_segments,
        index_summary,
        consensus_daily,
        consensus_segments,
        index_pairs,
        args,
    )

    print("\nIndex comparison summary:")
    summary_columns = [
        "index_code",
        "index_name",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "bull_day_ratio",
        "bear_day_ratio",
        "sideways_day_ratio",
        "last_direction",
    ]
    print(index_summary[summary_columns].to_string(index=False))

    print("\nConsensus regime stages:")
    if consensus_segments.empty:
        print("No valid consensus stages.")
    else:
        display_columns = [
            "stage_id",
            "start_date",
            "end_date",
            "direction",
            "trading_days",
            "mean_bull_ratio",
            "mean_bear_ratio",
            "dominant_index_by_days",
        ]
        print(consensus_segments[display_columns].to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
