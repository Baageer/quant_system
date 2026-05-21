"""
Single-factor quantile group return analysis for the HS300 universe.

For each trading date, stocks are sorted by one factor and split into N groups.
The script then summarizes future returns for Q1..QN, Top-Bottom spreads,
long-group win rates, monotonicity, and multiple holding horizons.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_panel import standardize_panel, winsorize_panel  # noqa: E402
from research.factor_ic_analysis_hs300 import (  # noqa: E402
    build_forward_return_panel,
    parse_factor_names,
    parse_int_list,
)
from research.factor_quality_check_hs300 import (  # noqa: E402
    build_factor_panel_with_progress,
    export_reports,
    load_price_data,
    log_step,
    progress_iter,
    read_symbols,
)


def _valid_group_sample(factor_values: pd.Series, return_values: pd.Series, min_obs: int) -> pd.DataFrame:
    sample = pd.DataFrame({"factor": factor_values, "future_return": return_values}).dropna()
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < min_obs:
        return pd.DataFrame()
    if sample["factor"].nunique() <= 1:
        return pd.DataFrame()
    return sample


def assign_quantile_groups(sample: pd.DataFrame, group_count: int) -> pd.Series:
    if len(sample) < group_count:
        return pd.Series(index=sample.index, dtype=float)
    ranked = sample["factor"].rank(method="first")
    labels = pd.qcut(ranked, q=group_count, labels=False) + 1
    return labels.astype(int)


def calculate_group_monotonicity(group_returns: pd.Series) -> Dict[str, object]:
    ordered = group_returns.sort_index().dropna()
    if len(ordered) < 2:
        return {
            "monotonicity_corr": np.nan,
            "is_monotonic_increasing": False,
            "is_monotonic_decreasing": False,
            "monotonicity_direction": "unknown",
        }

    group_numbers = pd.Series(ordered.index.astype(float), index=ordered.index)
    group_rank = group_numbers.rank(method="average")
    return_rank = ordered.rank(method="average")
    monotonicity_corr = group_rank.corr(return_rank, method="pearson")
    diffs = ordered.diff().dropna()
    is_increasing = bool((diffs > 0).all())
    is_decreasing = bool((diffs < 0).all())
    if is_increasing:
        direction = "increasing"
    elif is_decreasing:
        direction = "decreasing"
    elif monotonicity_corr == monotonicity_corr and monotonicity_corr > 0:
        direction = "mostly_increasing"
    elif monotonicity_corr == monotonicity_corr and monotonicity_corr < 0:
        direction = "mostly_decreasing"
    else:
        direction = "mixed"

    return {
        "monotonicity_corr": float(monotonicity_corr) if monotonicity_corr == monotonicity_corr else np.nan,
        "is_monotonic_increasing": is_increasing,
        "is_monotonic_decreasing": is_decreasing,
        "monotonicity_direction": direction,
    }


def calculate_group_returns_by_date(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: List[str],
    horizons: List[int],
    group_count: int,
    min_obs: int,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = factor_panel[factor_names].join(forward_panel, how="inner")
    dates = sorted(set(combined.index.get_level_values("date")))
    group_rows = []
    spread_rows = []
    tasks = [(factor, horizon) for factor in factor_names for horizon in horizons]

    for factor, horizon in progress_iter(tasks, "Calculating group returns", total=len(tasks), enabled=show_progress):
        return_column = f"forward_return_{horizon}d"
        factor_slice = combined[[factor, return_column]]

        for date in dates:
            try:
                date_frame = factor_slice.xs(date, level="date")
            except KeyError:
                continue

            sample = _valid_group_sample(date_frame[factor], date_frame[return_column], min_obs=min_obs)
            if sample.empty:
                continue

            try:
                sample = sample.copy()
                sample["group"] = assign_quantile_groups(sample, group_count)
            except ValueError:
                continue
            sample = sample.dropna(subset=["group"])
            if sample.empty:
                continue
            sample["group"] = sample["group"].astype(int)

            daily_group = sample.groupby("group")["future_return"].agg(["mean", "median", "count"])
            daily_win_rate = sample.groupby("group")["future_return"].apply(lambda values: float((values > 0).mean()))

            for group_number, row in daily_group.iterrows():
                group_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": int(horizon),
                        "group_count": int(group_count),
                        "group": int(group_number),
                        "group_label": f"Q{int(group_number)}",
                        "mean_return": float(row["mean"]),
                        "median_return": float(row["median"]),
                        "win_rate": float(daily_win_rate.loc[group_number]),
                        "sample_count": int(row["count"]),
                    }
                )

            if 1 in daily_group.index and group_count in daily_group.index:
                group_means = daily_group["mean"]
                top_group = sample[sample["group"] == group_count]["future_return"]
                bottom_group = sample[sample["group"] == 1]["future_return"]
                monotonicity = calculate_group_monotonicity(group_means)
                spread_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": int(horizon),
                        "group_count": int(group_count),
                        "top_group": f"Q{group_count}",
                        "bottom_group": "Q1",
                        "top_mean_return": float(group_means.loc[group_count]),
                        "bottom_mean_return": float(group_means.loc[1]),
                        "top_bottom_return": float(group_means.loc[group_count] - group_means.loc[1]),
                        "long_group_win_rate": float((top_group > 0).mean()),
                        "bottom_group_win_rate": float((bottom_group > 0).mean()),
                        "top_sample_count": int(len(top_group)),
                        "bottom_sample_count": int(len(bottom_group)),
                        **monotonicity,
                    }
                )

    group_by_date = pd.DataFrame(group_rows)
    spread_by_date = pd.DataFrame(spread_rows)
    if not group_by_date.empty:
        group_by_date["date"] = pd.to_datetime(group_by_date["date"])
        group_by_date = group_by_date.sort_values(["factor", "horizon", "date", "group"]).reset_index(drop=True)
    if not spread_by_date.empty:
        spread_by_date["date"] = pd.to_datetime(spread_by_date["date"])
        spread_by_date = spread_by_date.sort_values(["factor", "horizon", "date"]).reset_index(drop=True)
    return group_by_date, spread_by_date


def summarize_group_returns(group_by_date: pd.DataFrame) -> pd.DataFrame:
    if group_by_date.empty:
        return pd.DataFrame()

    rows = []
    for (factor, horizon, group), frame in group_by_date.groupby(["factor", "horizon", "group"]):
        weighted_sum = (frame["mean_return"] * frame["sample_count"]).sum()
        sample_sum = frame["sample_count"].sum()
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "group": int(group),
                "group_label": f"Q{int(group)}",
                "daily_count": int(len(frame)),
                "avg_daily_mean_return": float(frame["mean_return"].mean()),
                "median_daily_mean_return": float(frame["mean_return"].median()),
                "stock_weighted_mean_return": float(weighted_sum / sample_sum) if sample_sum else np.nan,
                "avg_daily_win_rate": float(frame["win_rate"].mean()),
                "positive_daily_rate": float((frame["mean_return"] > 0).mean()),
                "avg_sample_count": float(frame["sample_count"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["factor", "horizon", "group"])


def summarize_spreads(spread_by_date: pd.DataFrame) -> pd.DataFrame:
    if spread_by_date.empty:
        return pd.DataFrame()

    rows = []
    for (factor, horizon), frame in spread_by_date.groupby(["factor", "horizon"]):
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "daily_count": int(len(frame)),
                "avg_top_return": float(frame["top_mean_return"].mean()),
                "avg_bottom_return": float(frame["bottom_mean_return"].mean()),
                "avg_top_bottom_return": float(frame["top_bottom_return"].mean()),
                "median_top_bottom_return": float(frame["top_bottom_return"].median()),
                "top_bottom_win_rate": float((frame["top_bottom_return"] > 0).mean()),
                "long_group_positive_daily_rate": float((frame["top_mean_return"] > 0).mean()),
                "avg_long_group_win_rate": float(frame["long_group_win_rate"].mean()),
                "avg_monotonicity_corr": float(frame["monotonicity_corr"].mean()),
                "monotonic_increase_rate": float(frame["is_monotonic_increasing"].mean()),
                "monotonic_decrease_rate": float(frame["is_monotonic_decreasing"].mean()),
                "direction": (
                    "top_better"
                    if frame["top_bottom_return"].mean() > 0
                    else ("bottom_better" if frame["top_bottom_return"].mean() < 0 else "flat")
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["abs_avg_top_bottom_return"] = result["avg_top_bottom_return"].abs()
    return result.sort_values(["horizon", "abs_avg_top_bottom_return"], ascending=[True, False])


def build_group_selection_view(spread_summary: pd.DataFrame) -> pd.DataFrame:
    if spread_summary.empty:
        return pd.DataFrame()

    frame = spread_summary.copy()
    frame["suggestion"] = "watch"
    keep_mask = (
        (frame["abs_avg_top_bottom_return"] >= 0.005)
        & (frame["top_bottom_win_rate"] >= 0.53)
        & (frame["avg_monotonicity_corr"].abs() >= 0.30)
    )
    frame.loc[keep_mask, "suggestion"] = "keep"
    frame.loc[
        (frame["abs_avg_top_bottom_return"] < 0.001)
        & (frame["top_bottom_win_rate"].between(0.47, 0.53)),
        "suggestion",
    ] = "drop_weak_spread"
    frame.loc[frame["daily_count"] < 30, "suggestion"] = "drop_low_sample"
    return frame.sort_values(
        ["horizon", "suggestion", "abs_avg_top_bottom_return"],
        ascending=[True, True, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HS300 single-factor quantile group return analysis.")
    parser.add_argument("--stock-file", default="data/HS300.txt")
    parser.add_argument("--cache-dir", default="data/raw/tushare/price_history/qfq")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--factors", default="all", help="Comma-separated factor names or all.")
    parser.add_argument("--forward-horizons", default="1,3,5,10,20")
    parser.add_argument("--groups", type=int, default=5, choices=[5, 10], help="Quantile group count.")
    parser.add_argument("--min-obs", type=int, default=30, help="Minimum stocks required for one daily cross-section.")
    parser.add_argument("--winsorize", action="store_true", help="Apply cross-sectional factor winsorization before grouping.")
    parser.add_argument("--standardize", action="store_true", help="Apply cross-sectional factor z-score before grouping.")
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

    log_step(f"Calculating Q{args.groups} group returns")
    group_by_date, spread_by_date = calculate_group_returns_by_date(
        factor_panel,
        forward_panel,
        factor_names,
        horizons,
        group_count=args.groups,
        min_obs=args.min_obs,
        show_progress=show_progress,
    )
    log_step("Summarizing group returns")
    group_summary = summarize_group_returns(group_by_date)
    spread_summary = summarize_spreads(spread_by_date)
    selection_view = build_group_selection_view(spread_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"factor_group_return_hs300_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"Exporting group-return reports to {output_dir.relative_to(PROJECT_ROOT)}")
    export_reports(
        output_dir,
        {
            "price_load_report.csv": load_report,
            "factor_group_return_by_date.csv": group_by_date,
            "factor_group_return_summary.csv": group_summary,
            "factor_top_bottom_by_date.csv": spread_by_date,
            "factor_top_bottom_summary.csv": spread_summary,
            "factor_group_selection_view.csv": selection_view,
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
        "groups": args.groups,
        "min_obs": args.min_obs,
        "winsorize": bool(args.winsorize),
        "standardize": bool(args.standardize),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "top_abs_spreads": selection_view.head(10).to_dict(orient="records") if not selection_view.empty else [],
    }
    with open(output_dir / "group_return_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

    log_step("Group return analysis completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

