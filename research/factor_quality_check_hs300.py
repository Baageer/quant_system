"""
Offline factor data-quality checks for the HS300 universe.

The script reads local cached price-history CSV files, builds the unified factor
panel, and exports data-quality reports to output/.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_panel import (  # noqa: E402
    AVAILABLE_FACTOR_SPECS,
    calculate_single_stock_factors,
    list_available_factors,
)

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    tqdm = None


STANDARD_COLUMNS = ["date", "code", "open", "close", "high", "low", "volume", "amount"]
NUMERIC_COLUMNS = ["open", "close", "high", "low", "volume", "amount"]
DEFAULT_FORWARD_HORIZONS = [1, 3, 5, 10, 20]


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def progress_iter(iterable, desc: str, total: Optional[int] = None, enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="item")


def _resolve_worker_count(max_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if max_workers <= 0:
        return max(1, min(os.cpu_count() or 1, task_count))
    return max(1, min(max_workers, task_count))


def _calculate_symbol_factor_frame(task: Tuple[str, pd.DataFrame, Tuple[str, ...]]) -> pd.DataFrame:
    symbol, data, factor_names = task
    factors = calculate_single_stock_factors(data, factor_names=factor_names, include_ohlcv=False)
    factors["symbol"] = str(symbol)
    factors["date"] = factors.index
    return factors.set_index(["date", "symbol"])


def read_symbols(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def market_suffix(symbol: str) -> str:
    return "SH" if symbol.startswith(("5", "6", "9")) else "SZ"


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


def find_price_file(symbol: str, cache_dir: Path) -> Optional[Path]:
    suffix = market_suffix(symbol)
    candidates = [
        cache_dir / f"{symbol}.{suffix}_20050101_20241231.csv",
        cache_dir / f"{symbol}.{suffix}.csv",
        cache_dir / f"{symbol}_20050101_20241231.csv",
        cache_dir / f"{symbol}.csv",
    ]
    candidates.extend(sorted(cache_dir.glob(f"{symbol}.{suffix}_*.csv")))
    candidates.extend(sorted(cache_dir.glob(f"{symbol}_*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def standardize_price_frame(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    alias_map = {
        "trade_date": "date",
        "ts_code": "code",
        "vol": "volume",
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw_df.rename(columns=alias_map).copy()
    unnamed_columns = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(unnamed_columns, axis=1)

    if "date" not in df.columns:
        raise ValueError(f"{symbol}: missing date column")
    if "code" not in df.columns:
        df["code"] = f"{symbol}.{market_suffix(symbol)}"

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notnull()].copy()
    df["code"] = df["code"].fillna(f"{symbol}.{market_suffix(symbol)}").astype(str)
    df = df[STANDARD_COLUMNS].sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date")
    return df


def load_price_data(
    symbols: Iterable[str],
    cache_dir: Path,
    start_date: str,
    end_date: str,
    show_progress: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    stock_data = {}
    rows = []
    symbol_list = list(symbols)

    for symbol in progress_iter(symbol_list, "Loading price cache", total=len(symbol_list), enabled=show_progress):
        price_file = find_price_file(symbol, cache_dir)
        if price_file is None:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "missing_file",
                    "file": "",
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                }
            )
            continue

        try:
            raw_df = read_csv_with_fallback(price_file)
            df = standardize_price_frame(raw_df, symbol)
            df = df[(df.index >= start) & (df.index <= end)].copy()
            if not df.empty:
                stock_data[symbol] = df
            rows.append(
                {
                    "symbol": symbol,
                    "status": "ok" if not df.empty else "empty_window",
                    "file": str(price_file.relative_to(PROJECT_ROOT)),
                    "rows": int(len(df)),
                    "first_date": "" if df.empty else df.index.min().strftime("%Y-%m-%d"),
                    "last_date": "" if df.empty else df.index.max().strftime("%Y-%m-%d"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "read_error",
                    "file": str(price_file.relative_to(PROJECT_ROOT)),
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                    "error": repr(exc),
                }
            )

    return stock_data, pd.DataFrame(rows)


def build_factor_panel_with_progress(
    stock_data: Dict[str, pd.DataFrame],
    factor_names: List[str],
    show_progress: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    frames = []
    items = sorted(stock_data.items())
    worker_count = _resolve_worker_count(max_workers, len(items))
    tasks = [(symbol, data, tuple(factor_names)) for symbol, data in items]

    if worker_count == 1:
        frame_iter = (_calculate_symbol_factor_frame(task) for task in tasks)
        for frame in progress_iter(frame_iter, "Calculating factors", total=len(items), enabled=show_progress):
            frames.append(frame)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_symbol = {
                executor.submit(_calculate_symbol_factor_frame, task): task[0]
                for task in tasks
            }
            completed_futures = as_completed(future_to_symbol)
            for future in progress_iter(completed_futures, "Calculating factors", total=len(items), enabled=show_progress):
                symbol = future_to_symbol[future]
                try:
                    frames.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"{symbol}: factor calculation failed") from exc

    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
    return pd.concat(frames).sort_index()


def calculate_date_missing(panel: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    missing = panel[factor_names].isnull().groupby(level="date").mean()
    missing.columns = [f"{column}_missing_rate" for column in missing.columns]
    missing["factor_mean_missing_rate"] = missing.mean(axis=1)
    return missing.reset_index()


def calculate_symbol_missing(panel: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    missing = panel[factor_names].isnull().groupby(level="symbol").mean()
    missing.columns = [f"{column}_missing_rate" for column in missing.columns]
    missing["factor_mean_missing_rate"] = missing.mean(axis=1)
    return missing.reset_index()


def calculate_factor_missing_summary(
    panel: pd.DataFrame,
    factor_names: List[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    total_rows = len(panel)
    for factor in progress_iter(factor_names, "Missing summary", total=len(factor_names), enabled=show_progress):
        values = panel[factor]
        valid = values.notnull()
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "required_columns": ",".join(AVAILABLE_FACTOR_SPECS[factor].required_columns),
                "total_rows": int(total_rows),
                "valid_rows": int(valid.sum()),
                "missing_rows": int((~valid).sum()),
                "missing_rate": float((~valid).mean()) if total_rows else np.nan,
                "first_valid_date": "" if not valid.any() else values[valid].index.get_level_values("date").min().strftime("%Y-%m-%d"),
                "last_valid_date": "" if not valid.any() else values[valid].index.get_level_values("date").max().strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "factor"], ascending=[False, True])


def calculate_factor_symbol_detail(
    panel: pd.DataFrame,
    factor_names: List[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    groups = list(panel[factor_names].groupby(level="symbol"))
    for symbol, symbol_frame in progress_iter(groups, "Symbol missing detail", total=len(groups), enabled=show_progress):
        symbol_dates = symbol_frame.index.get_level_values("date")
        for factor in factor_names:
            values = symbol_frame[factor]
            valid = values.notnull()
            first_valid_date = None
            if valid.any():
                first_valid_date = values[valid].index.get_level_values("date").min()
                after_first = symbol_dates >= first_valid_date
                missing_after_first = int(values[after_first].isnull().sum())
            else:
                missing_after_first = int(len(values))
            rows.append(
                {
                    "symbol": symbol,
                    "factor": factor,
                    "rows": int(len(values)),
                    "missing_rows": int(values.isnull().sum()),
                    "missing_rate": float(values.isnull().mean()) if len(values) else np.nan,
                    "first_valid_date": "" if first_valid_date is None else first_valid_date.strftime("%Y-%m-%d"),
                    "initial_or_warmup_missing_rows": int((~valid).cumprod().sum()) if len(values) else 0,
                    "missing_after_first_valid_rows": missing_after_first,
                }
            )
    return pd.DataFrame(rows)


def calculate_extreme_summary(
    panel: pd.DataFrame,
    factor_names: List[str],
    extreme_abs_threshold: float,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    for factor in progress_iter(factor_names, "Extreme summary", total=len(factor_names), enabled=show_progress):
        values = panel[factor]
        finite = np.isfinite(values.astype(float))
        non_missing = values.notnull()
        finite_values = values[finite]
        by_symbol = values.groupby(level="symbol")
        all_zero_symbols = by_symbol.apply(lambda series: bool((series.dropna() == 0).all()) if len(series.dropna()) else False)
        constant_symbols = by_symbol.apply(lambda series: int(series.dropna().nunique()) <= 1 if len(series.dropna()) else False)
        rows.append(
            {
                "factor": factor,
                "non_missing_rows": int(non_missing.sum()),
                "nan_rows": int(values.isnull().sum()),
                "inf_rows": int((non_missing & ~finite).sum()),
                "extreme_abs_threshold": float(extreme_abs_threshold),
                "extreme_abs_rows": int((finite_values.abs() > extreme_abs_threshold).sum()),
                "zero_rows": int((finite_values == 0).sum()),
                "zero_rate_among_finite": float((finite_values == 0).mean()) if len(finite_values) else np.nan,
                "all_zero_symbol_count": int(all_zero_symbols.sum()),
                "constant_symbol_count": int(constant_symbols.sum()),
                "min": float(finite_values.min()) if len(finite_values) else np.nan,
                "p01": float(finite_values.quantile(0.01)) if len(finite_values) else np.nan,
                "p50": float(finite_values.quantile(0.50)) if len(finite_values) else np.nan,
                "p99": float(finite_values.quantile(0.99)) if len(finite_values) else np.nan,
                "max": float(finite_values.max()) if len(finite_values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["inf_rows", "extreme_abs_rows", "constant_symbol_count"], ascending=False)


def calculate_daily_coverage(panel: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    valid_counts = panel[factor_names].notnull().groupby(level="date").sum()
    total_counts = panel[factor_names].groupby(level="date").size()
    coverage = valid_counts.copy()
    for factor in factor_names:
        coverage[f"{factor}_coverage_rate"] = valid_counts[factor] / total_counts
    coverage["price_observed_symbol_count"] = total_counts
    return coverage.reset_index()


def calculate_coverage_summary(
    daily_coverage: pd.DataFrame,
    factor_names: List[str],
    min_coverage_rate: float,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    observed = daily_coverage["price_observed_symbol_count"]
    for factor in progress_iter(factor_names, "Coverage summary", total=len(factor_names), enabled=show_progress):
        valid_count = daily_coverage[factor]
        coverage_rate = daily_coverage[f"{factor}_coverage_rate"]
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "min_valid_count": int(valid_count.min()),
                "median_valid_count": float(valid_count.median()),
                "max_valid_count": int(valid_count.max()),
                "min_coverage_rate": float(coverage_rate.min()),
                "median_coverage_rate": float(coverage_rate.median()),
                "low_coverage_date_count": int((coverage_rate < min_coverage_rate).sum()),
                "low_coverage_date_rate": float((coverage_rate < min_coverage_rate).mean()),
                "median_price_observed_symbol_count": float(observed.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_coverage_rate", "factor"])


def calculate_calendar_reports(
    stock_data: Dict[str, pd.DataFrame],
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(set().union(*[set(df.index) for df in stock_data.values()])) if stock_data else []
    calendar = pd.DatetimeIndex(all_dates)
    symbol_rows = []
    date_rows = []

    first_last = {}
    items = sorted(stock_data.items())
    for symbol, df in progress_iter(items, "Calendar by symbol", total=len(items), enabled=show_progress):
        dates = pd.DatetimeIndex(df.index)
        first_date = dates.min()
        last_date = dates.max()
        first_last[symbol] = (first_date, last_date)
        active_calendar = calendar[(calendar >= first_date) & (calendar <= last_date)]
        observed = set(dates)
        missing_inside = [date for date in active_calendar if date not in observed]
        symbol_rows.append(
            {
                "symbol": symbol,
                "first_date": first_date.strftime("%Y-%m-%d"),
                "last_date": last_date.strftime("%Y-%m-%d"),
                "observed_days": int(len(dates)),
                "active_calendar_days": int(len(active_calendar)),
                "missing_inside_active_days": int(len(missing_inside)),
                "missing_inside_active_rate": float(len(missing_inside) / len(active_calendar)) if len(active_calendar) else np.nan,
                "pre_listing_or_pre_cache_days": int((calendar < first_date).sum()),
                "post_last_cache_days": int((calendar > last_date).sum()),
            }
        )

    for date in progress_iter(calendar, "Calendar by date", total=len(calendar), enabled=show_progress):
        observed_symbols = [symbol for symbol, df in stock_data.items() if date in df.index]
        active_symbols = [
            symbol
            for symbol, (first_date, last_date) in first_last.items()
            if first_date <= date <= last_date
        ]
        date_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "observed_symbol_count": int(len(observed_symbols)),
                "active_symbol_count": int(len(active_symbols)),
                "active_missing_symbol_count": int(len(active_symbols) - len(observed_symbols)),
            }
        )

    return pd.DataFrame(symbol_rows), pd.DataFrame(date_rows)


def calculate_forward_return_alignment(
    stock_data: Dict[str, pd.DataFrame],
    horizons: List[int],
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    for horizon in progress_iter(horizons, "Forward labels", total=len(horizons), enabled=show_progress):
        total_rows = 0
        missing_rows = 0
        expected_terminal_missing = 0
        unexpected_missing = 0
        for symbol, df in stock_data.items():
            close = df["close"].astype(float)
            forward_return = close.shift(-horizon) / close - 1.0
            total_rows += int(len(forward_return))
            missing = int(forward_return.isnull().sum())
            expected = min(int(horizon), int(len(forward_return)))
            missing_rows += missing
            expected_terminal_missing += expected
            unexpected_missing += max(0, missing - expected)
        rows.append(
            {
                "horizon": int(horizon),
                "label_formula": f"close.shift(-{horizon}) / close - 1",
                "total_rows": int(total_rows),
                "missing_rows": int(missing_rows),
                "expected_terminal_missing_rows": int(expected_terminal_missing),
                "unexpected_missing_rows": int(unexpected_missing),
                "status": "pass" if unexpected_missing == 0 else "check",
            }
        )
    return pd.DataFrame(rows)


def export_report(frame: pd.DataFrame, output_dir: Path, filename: str) -> None:
    frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def export_reports(output_dir: Path, reports: Dict[str, pd.DataFrame], show_progress: bool = True) -> None:
    items = list(reports.items())
    for filename, frame in progress_iter(items, "Exporting reports", total=len(items), enabled=show_progress):
        export_report(frame, output_dir, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HS300 factor data-quality checks.")
    parser.add_argument("--stock-file", default="data/HS300.txt")
    parser.add_argument("--cache-dir", default="data/raw/tushare/price_history/qfq")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--min-coverage-rate", type=float, default=0.70)
    parser.add_argument("--extreme-abs-threshold", type=float, default=1e6)
    parser.add_argument("--factors", default="all", help="Comma-separated factor names or all.")
    parser.add_argument("--forward-horizons", default="1,3,5,10,20")
    parser.add_argument(
        "--factor-workers",
        type=int,
        default=0,
        help="Factor calculation worker processes. Use 0 for auto, 1 for serial.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    stock_file = PROJECT_ROOT / args.stock_file
    cache_dir = PROJECT_ROOT / args.cache_dir
    factor_names = (
        sorted(AVAILABLE_FACTOR_SPECS)
        if args.factors == "all"
        else [name.strip() for name in args.factors.split(",") if name.strip()]
    )
    forward_horizons = [int(item.strip()) for item in args.forward_horizons.split(",") if item.strip()]

    symbols = read_symbols(stock_file)
    log_step(f"Loading {len(symbols)} symbols from {stock_file.relative_to(PROJECT_ROOT)}")
    log_step(f"Using cache dir: {cache_dir.relative_to(PROJECT_ROOT)}")
    stock_data, load_report = load_price_data(
        symbols,
        cache_dir,
        args.start,
        args.end,
        show_progress=show_progress,
    )
    log_step(f"Loaded {len(stock_data)} symbols with price rows")

    factor_worker_count = _resolve_worker_count(args.factor_workers, len(stock_data))
    log_step(f"Building factor panel with {len(factor_names)} factors using {factor_worker_count} worker(s)")
    panel = build_factor_panel_with_progress(
        stock_data,
        factor_names=factor_names,
        show_progress=show_progress,
        max_workers=args.factor_workers,
    )
    log_step(f"Factor panel rows: {len(panel)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"factor_quality_hs300_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step("Calculating quality reports")
    factor_catalog = list_available_factors()
    date_missing = calculate_date_missing(panel, factor_names)
    symbol_missing = calculate_symbol_missing(panel, factor_names)
    missing_summary = calculate_factor_missing_summary(panel, factor_names, show_progress=show_progress)
    factor_symbol_detail = calculate_factor_symbol_detail(panel, factor_names, show_progress=show_progress)
    extreme_summary = calculate_extreme_summary(
        panel,
        factor_names,
        args.extreme_abs_threshold,
        show_progress=show_progress,
    )
    daily_coverage = calculate_daily_coverage(panel, factor_names)
    coverage_summary = calculate_coverage_summary(
        daily_coverage,
        factor_names,
        args.min_coverage_rate,
        show_progress=show_progress,
    )
    calendar_by_symbol, calendar_by_date = calculate_calendar_reports(stock_data, show_progress=show_progress)
    forward_alignment = calculate_forward_return_alignment(
        stock_data,
        forward_horizons,
        show_progress=show_progress,
    )

    panel_row_count_expected = int(sum(len(df) for df in stock_data.values()))
    time_alignment_checks = pd.DataFrame(
        [
            {
                "check": "panel_does_not_reindex_or_fill_missing_trading_days",
                "status": "pass" if len(panel) == panel_row_count_expected else "check",
                "detail": f"panel_rows={len(panel)}, observed_price_rows={panel_row_count_expected}",
            },
            {
                "check": "future_return_labels_use_negative_shift",
                "status": "pass" if (forward_alignment["unexpected_missing_rows"] == 0).all() else "check",
                "detail": "See forward_return_alignment.csv",
            },
            {
                "check": "factor_rows_do_not_exceed_observed_price_rows",
                "status": "pass" if len(panel) <= panel_row_count_expected else "check",
                "detail": "Factors are calculated only on observed per-symbol price rows.",
            },
        ]
    )

    log_step(f"Exporting reports to {output_dir.relative_to(PROJECT_ROOT)}")
    export_reports(
        output_dir,
        {
            "factor_catalog.csv": factor_catalog,
            "price_load_report.csv": load_report,
            "factor_missing_by_date.csv": date_missing,
            "factor_missing_by_symbol.csv": symbol_missing,
            "factor_missing_summary.csv": missing_summary,
            "factor_symbol_missing_detail.csv": factor_symbol_detail,
            "factor_extreme_summary.csv": extreme_summary,
            "factor_daily_coverage.csv": daily_coverage,
            "factor_coverage_summary.csv": coverage_summary,
            "price_calendar_by_symbol.csv": calendar_by_symbol,
            "price_calendar_by_date.csv": calendar_by_date,
            "forward_return_alignment.csv": forward_alignment,
            "time_alignment_checks.csv": time_alignment_checks,
        },
        show_progress=show_progress,
    )

    key_factors = [factor for factor in ["rsrs_score", "mfi_14"] if factor in factor_names]
    summary = {
        "stock_file": str(stock_file.relative_to(PROJECT_ROOT)),
        "cache_dir": str(cache_dir.relative_to(PROJECT_ROOT)),
        "start": args.start,
        "end": args.end,
        "requested_symbol_count": len(symbols),
        "loaded_symbol_count": len(stock_data),
        "panel_rows": int(len(panel)),
        "factor_count": len(factor_names),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "worst_missing_factors": missing_summary.head(10).to_dict(orient="records"),
        "lowest_coverage_factors": coverage_summary.head(10).to_dict(orient="records"),
        "key_factor_coverage": coverage_summary[coverage_summary["factor"].isin(key_factors)].to_dict(orient="records"),
    }
    with open(output_dir / "quality_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

    log_step("Quality check completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
