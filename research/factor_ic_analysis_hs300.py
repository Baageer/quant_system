"""
Single-factor Rank IC analysis for the HS300 universe.

The script reuses the local price-cache loader and unified factor panel builder
from factor_quality_check_hs300.py. For each trading date it calculates the
cross-sectional Spearman correlation between factor values and future returns.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_panel import (  # noqa: E402
    AVAILABLE_FACTOR_SPECS,
    standardize_panel,
    winsorize_panel,
)
from research.feature_engineering import FeatureEngineer  # noqa: E402
from research.factor_quality_check_hs300 import (  # noqa: E402
    build_factor_panel_with_progress,
    export_reports,
    load_price_data,
    log_step,
    progress_iter,
    read_symbols,
)


DEFAULT_FORWARD_HORIZONS = [1, 3, 5, 10, 20]


def parse_factor_names(text: str) -> List[str]:
    if text == "all":
        return sorted(AVAILABLE_FACTOR_SPECS)
    factor_names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in factor_names if name not in AVAILABLE_FACTOR_SPECS]
    if unknown:
        raise ValueError(f"Unknown factors: {unknown}")
    return factor_names


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    values = sorted(set(value for value in values if value > 0))
    if not values:
        raise ValueError("At least one positive horizon is required.")
    return values


def build_forward_return_panel(
    stock_data: Dict[str, pd.DataFrame],
    horizons: List[int],
    show_progress: bool = True,
) -> pd.DataFrame:
    frames = []
    items = sorted(stock_data.items())
    for symbol, data in progress_iter(items, "Building forward returns", total=len(items), enabled=show_progress):
        close = data["close"].astype(float)
        frame = pd.DataFrame(index=data.index)
        for horizon in horizons:
            frame[f"forward_return_{horizon}d"] = close.shift(-horizon) / close - 1.0
        frame["date"] = frame.index
        frame["symbol"] = str(symbol)
        frames.append(frame.set_index(["date", "symbol"]))

    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
    return pd.concat(frames).sort_index()


def _valid_ic_sample(factor_values: pd.Series, return_values: pd.Series, min_obs: int) -> pd.DataFrame:
    sample = pd.DataFrame({"factor": factor_values, "future_return": return_values}).dropna()
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < min_obs:
        return pd.DataFrame()
    if sample["factor"].nunique() <= 1 or sample["future_return"].nunique() <= 1:
        return pd.DataFrame()
    return sample


def calculate_spearman_ic(
    engineer: FeatureEngineer,
    factor_values: pd.Series,
    return_values: pd.Series,
) -> float:
    try:
        return engineer.calculate_ic(factor_values, return_values, method="spearman")
    except ModuleNotFoundError:
        factor_rank = factor_values.rank(method="average")
        return_rank = return_values.rank(method="average")
        return factor_rank.corr(return_rank, method="pearson")


def calculate_rank_ic_by_date(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: List[str],
    horizons: List[int],
    min_obs: int = 30,
    show_progress: bool = True,
) -> pd.DataFrame:
    engineer = FeatureEngineer()
    combined = factor_panel[factor_names].join(forward_panel, how="inner")
    dates = sorted(set(combined.index.get_level_values("date")))
    rows = []

    tasks = [(factor, horizon) for factor in factor_names for horizon in horizons]
    for factor, horizon in progress_iter(tasks, "Calculating Rank IC", total=len(tasks), enabled=show_progress):
        return_column = f"forward_return_{horizon}d"
        columns = [factor, return_column]
        factor_slice = combined[columns]

        for date in dates:
            try:
                date_frame = factor_slice.xs(date, level="date")
            except KeyError:
                continue
            sample = _valid_ic_sample(date_frame[factor], date_frame[return_column], min_obs=min_obs)
            if sample.empty:
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": horizon,
                        "rank_ic": np.nan,
                        "n_obs": int(len(date_frame.dropna())),
                        "status": "insufficient_or_constant",
                    }
                )
                continue

            rank_ic = calculate_spearman_ic(engineer, sample["factor"], sample["future_return"])
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "horizon": horizon,
                    "rank_ic": float(rank_ic) if rank_ic == rank_ic else np.nan,
                    "n_obs": int(len(sample)),
                    "status": "ok" if rank_ic == rank_ic else "nan_ic",
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["factor", "horizon", "date"]).reset_index(drop=True)


def calculate_ic_summary(ic_by_date: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, horizon), group in ic_by_date.groupby(["factor", "horizon"]):
        values = group["rank_ic"].dropna()
        ic_count = int(len(values))
        ic_mean = float(values.mean()) if ic_count else np.nan
        ic_std = float(values.std()) if ic_count > 1 else np.nan
        icir = ic_mean / ic_std if ic_std == ic_std and ic_std != 0 else np.nan
        t_value = ic_mean / (ic_std / math.sqrt(ic_count)) if ic_std == ic_std and ic_std != 0 and ic_count > 1 else np.nan
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "horizon": int(horizon),
                "ic_count": ic_count,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_win_rate": float((values > 0).mean()) if ic_count else np.nan,
                "positive_ic_count": int((values > 0).sum()) if ic_count else 0,
                "negative_ic_count": int((values < 0).sum()) if ic_count else 0,
                "t_value": t_value,
                "mean_n_obs": float(group["n_obs"].mean()) if len(group) else np.nan,
                "valid_date_rate": float(ic_count / len(group)) if len(group) else np.nan,
                "direction": "positive" if ic_mean == ic_mean and ic_mean > 0 else ("negative" if ic_mean == ic_mean and ic_mean < 0 else "neutral"),
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon", "icir", "ic_mean"], ascending=[True, False, False])


def calculate_ic_by_year(ic_by_date: pd.DataFrame) -> pd.DataFrame:
    frame = ic_by_date.copy()
    frame["year"] = frame["date"].dt.year
    rows = []
    for (factor, horizon, year), group in frame.groupby(["factor", "horizon", "year"]):
        values = group["rank_ic"].dropna()
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "year": int(year),
                "ic_count": int(len(values)),
                "ic_mean": float(values.mean()) if len(values) else np.nan,
                "ic_std": float(values.std()) if len(values) > 1 else np.nan,
                "ic_win_rate": float((values > 0).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["factor", "horizon", "year"])


def calculate_rolling_ic(ic_by_date: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    rows = []
    for (factor, horizon), group in ic_by_date.groupby(["factor", "horizon"]):
        ordered = group.sort_values("date").copy()
        rolling_mean = ordered["rank_ic"].rolling(rolling_window, min_periods=max(5, rolling_window // 3)).mean()
        rolling_std = ordered["rank_ic"].rolling(rolling_window, min_periods=max(5, rolling_window // 3)).std()
        rolling_icir = rolling_mean / rolling_std.replace(0, np.nan)
        frame = pd.DataFrame(
            {
                "date": ordered["date"].values,
                "factor": factor,
                "horizon": int(horizon),
                "rolling_window": int(rolling_window),
                "rolling_ic_mean": rolling_mean.values,
                "rolling_ic_std": rolling_std.values,
                "rolling_icir": rolling_icir.values,
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["factor", "horizon", "date"])


def build_selection_view(ic_summary: pd.DataFrame) -> pd.DataFrame:
    frame = ic_summary.copy()
    frame["abs_ic_mean"] = frame["ic_mean"].abs()
    frame["abs_icir"] = frame["icir"].abs()
    frame["suggestion"] = "watch"
    keep_mask = (
        (frame["abs_ic_mean"] >= 0.02)
        & (frame["abs_icir"] >= 0.30)
        & (frame["valid_date_rate"] >= 0.80)
    )
    frame.loc[keep_mask, "suggestion"] = "keep"
    frame.loc[(frame["ic_count"] < 30) | (frame["valid_date_rate"] < 0.50), "suggestion"] = "drop_low_sample"
    frame.loc[(frame["abs_ic_mean"] < 0.005) & (frame["valid_date_rate"] >= 0.80), "suggestion"] = "drop_weak_ic"
    return frame.sort_values(["horizon", "suggestion", "abs_icir", "abs_ic_mean"], ascending=[True, True, False, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-factor Rank IC analysis for HS300.")
    parser.add_argument("--stock-file", default="data/ZZ_500.txt")
    parser.add_argument("--cache-dir", default="data/raw/tushare/price_history/hfq")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--factors", default="all", help="Comma-separated factor names or all.")
    parser.add_argument("--forward-horizons", default="5,10,20")
    parser.add_argument("--min-obs", type=int, default=30, help="Minimum stocks required for one daily cross-section.")
    parser.add_argument("--rolling-window", type=int, default=60, help="Trading-day window for rolling IC.")
    parser.add_argument("--winsorize", action="store_true", help="Apply cross-sectional factor winsorization before IC.")
    parser.add_argument("--standardize", action="store_true", help="Apply cross-sectional factor z-score before IC.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    stock_file = PROJECT_ROOT / args.stock_file
    cache_dir = PROJECT_ROOT / args.cache_dir
    factor_names = parse_factor_names(args.factors)
    horizons = parse_int_list(args.forward_horizons)

    log_step(f"Loading symbols from {stock_file.relative_to(PROJECT_ROOT)}")
    symbols = read_symbols(stock_file)
    log_step(f"Loading local price cache for {len(symbols)} symbols")
    stock_data, load_report = load_price_data(
        symbols,
        cache_dir,
        args.start,
        args.end,
        show_progress=show_progress,
    )
    log_step(f"Loaded {len(stock_data)} symbols")

    log_step(f"Building factor panel: {len(factor_names)} factors")
    factor_panel = build_factor_panel_with_progress(
        stock_data,
        factor_names=factor_names,
        show_progress=show_progress,
    )
    if args.winsorize:
        log_step("Applying cross-sectional winsorization")
        factor_panel = winsorize_panel(factor_panel, factor_columns=factor_names)
    if args.standardize:
        log_step("Applying cross-sectional standardization")
        factor_panel = standardize_panel(factor_panel, factor_columns=factor_names)

    log_step(f"Building forward return labels: horizons={horizons}")
    forward_panel = build_forward_return_panel(stock_data, horizons, show_progress=show_progress)

    log_step("Calculating daily Rank IC")
    ic_by_date = calculate_rank_ic_by_date(
        factor_panel,
        forward_panel,
        factor_names,
        horizons,
        min_obs=args.min_obs,
        show_progress=show_progress,
    )
    log_step("Summarizing IC")
    ic_summary = calculate_ic_summary(ic_by_date)
    ic_by_year = calculate_ic_by_year(ic_by_date)
    rolling_ic = calculate_rolling_ic(ic_by_date, args.rolling_window)
    selection_view = build_selection_view(ic_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"factor_ic_hs300_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"Exporting IC reports to {output_dir.relative_to(PROJECT_ROOT)}")
    export_reports(
        output_dir,
        {
            "price_load_report.csv": load_report,
            "factor_ic_by_date.csv": ic_by_date,
            "factor_ic_summary.csv": ic_summary,
            "factor_ic_by_year.csv": ic_by_year,
            "factor_rolling_ic.csv": rolling_ic,
            "factor_selection_view.csv": selection_view,
        },
        show_progress=show_progress,
    )

    summary = {
        "stock_file": str(stock_file.relative_to(PROJECT_ROOT)),
        "cache_dir": str(cache_dir.relative_to(PROJECT_ROOT)),
        "start": args.start,
        "end": args.end,
        "loaded_symbol_count": len(stock_data),
        "factor_count": len(factor_names),
        "horizons": horizons,
        "min_obs": args.min_obs,
        "rolling_window": args.rolling_window,
        "winsorize": bool(args.winsorize),
        "standardize": bool(args.standardize),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "top_abs_icir": selection_view.sort_values("abs_icir", ascending=False).head(10).to_dict(orient="records"),
    }
    with open(output_dir / "ic_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

    log_step("Rank IC analysis completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
