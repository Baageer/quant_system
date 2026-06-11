"""
Single-factor Rank IC analysis for the HS300 universe.

The script reuses the local price-cache loader and unified factor panel builder
from factor_quality_check_hs300.py. For each trading date it calculates the
cross-sectional Spearman correlation between factor values and future returns.
"""
import argparse
import json
import math
import re
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


def stock_file_slug(stock_file: Path) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "", stock_file.stem).lower()
    return slug or "stocks"


def list_factor_categories() -> List[str]:
    return sorted({spec.category for spec in AVAILABLE_FACTOR_SPECS.values()})


def filter_factor_names_by_category(factor_names: List[str], category_text: str) -> List[str]:
    if category_text == "all":
        return factor_names

    categories = [category.strip() for category in category_text.split(",") if category.strip()]
    available_categories = list_factor_categories()
    unknown = [category for category in categories if category not in available_categories]
    if unknown:
        raise ValueError(f"Unknown factor categories: {unknown}. Available categories: {available_categories}")

    selected_categories = set(categories)
    filtered = [
        factor
        for factor in factor_names
        if AVAILABLE_FACTOR_SPECS[factor].category in selected_categories
    ]
    if not filtered:
        raise ValueError(
            f"No factors left after category filter {categories}. "
            f"Selected factors: {factor_names}"
        )
    return filtered


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    values = sorted(set(value for value in values if value > 0))
    if not values:
        raise ValueError("At least one positive horizon is required.")
    return values


def parse_quantile_groups(text: str) -> List[int]:
    values = parse_int_list(text)
    invalid = [value for value in values if value < 2]
    if invalid:
        raise ValueError(f"Quantile group counts must be >= 2: {invalid}")
    return values


def _safe_prod_return(values: pd.Series) -> float:
    clean_values = values.dropna()
    if clean_values.empty:
        return np.nan
    return float((1.0 + clean_values).prod() - 1.0)


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 2 or sample["left"].nunique() <= 1 or sample["right"].nunique() <= 1:
        return np.nan
    return float(sample["left"].rank(method="average").corr(sample["right"].rank(method="average"), method="pearson"))


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


def add_effective_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["effective_factor_multiplier"] = np.where(result["direction"] == "negative", -1, 1)
    result["effective_factor"] = np.where(
        result["effective_factor_multiplier"] == -1,
        result["factor"].astype(str) + "_neg",
        result["factor"].astype(str),
    )
    result["effective_factor_formula"] = np.where(
        result["effective_factor_multiplier"] == -1,
        "-" + result["factor"].astype(str),
        result["factor"].astype(str),
    )
    result["effective_ic_mean"] = result["ic_mean"] * result["effective_factor_multiplier"]
    result["effective_icir"] = result["icir"] * result["effective_factor_multiplier"]
    return result


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

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = add_effective_factor_columns(result)
    return result.sort_values(["horizon", "effective_icir", "effective_ic_mean"], ascending=[True, False, False])


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
    frame = add_effective_factor_columns(frame)
    return frame.sort_values(["horizon", "suggestion", "abs_icir", "abs_ic_mean"], ascending=[True, True, False, False])


def build_effective_factor_mapping(selection_view: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "raw_factor",
        "category",
        "effective_factor",
        "effective_factor_multiplier",
        "effective_factor_formula",
        "direction",
        "source_horizon",
        "suggestion",
        "ic_mean",
        "icir",
        "effective_ic_mean",
        "effective_icir",
        "valid_date_rate",
        "ic_count",
    ]
    if selection_view.empty:
        return pd.DataFrame(columns=columns)

    frame = selection_view.copy()
    suggestion_priority = {
        "keep": 0,
        "watch": 1,
        "drop_weak_ic": 2,
        "drop_low_sample": 3,
    }
    frame["suggestion_priority"] = frame["suggestion"].map(suggestion_priority).fillna(9)
    frame = frame.sort_values(
        ["factor", "suggestion_priority", "abs_icir", "abs_ic_mean", "valid_date_rate"],
        ascending=[True, True, False, False, False],
    )
    mapping = frame.groupby("factor", as_index=False).head(1).copy()
    mapping = mapping.rename(columns={"factor": "raw_factor", "horizon": "source_horizon"})
    return mapping[columns].sort_values(["suggestion", "effective_icir", "effective_ic_mean"], ascending=[True, False, False])


def _effective_multiplier_by_factor_horizon(ic_summary: pd.DataFrame) -> Dict[tuple, int]:
    if ic_summary.empty:
        return {}
    return {
        (row["factor"], int(row["horizon"])): int(row["effective_factor_multiplier"])
        for _, row in ic_summary.iterrows()
    }


def _assign_quantile_groups(sample: pd.DataFrame, group_count: int) -> pd.DataFrame:
    if len(sample) < group_count or sample["effective_factor_value"].nunique() < group_count:
        return pd.DataFrame()
    result = sample.copy()
    result["quantile"] = pd.qcut(
        result["effective_factor_value"].rank(method="first"),
        group_count,
        labels=list(range(1, group_count + 1)),
    ).astype(int)
    return result


def calculate_quantile_backtest(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: List[str],
    horizons: List[int],
    quantile_groups: List[int],
    ic_summary: pd.DataFrame,
    min_obs: int,
    transaction_cost_bps: float,
    show_progress: bool = True,
) -> tuple:
    quantile_columns = [
        "date",
        "factor",
        "category",
        "horizon",
        "group_count",
        "quantile",
        "effective_factor_multiplier",
        "n_obs",
        "raw_return",
        "turnover",
        "cost_return",
        "net_return",
    ]
    long_short_columns = [
        "date",
        "factor",
        "category",
        "horizon",
        "group_count",
        "effective_factor_multiplier",
        "long_quantile",
        "short_quantile",
        "long_return",
        "short_return",
        "long_short_return",
        "long_turnover",
        "short_turnover",
        "long_short_turnover",
        "cost_return",
        "net_long_short_return",
    ]
    if factor_panel.empty or forward_panel.empty:
        return pd.DataFrame(columns=quantile_columns), pd.DataFrame(columns=long_short_columns)

    return_columns = [f"forward_return_{horizon}d" for horizon in horizons]
    combined = factor_panel[factor_names].join(forward_panel[return_columns], how="inner")
    dates = sorted(set(combined.index.get_level_values("date")))
    multiplier_lookup = _effective_multiplier_by_factor_horizon(ic_summary)
    cost_rate = float(transaction_cost_bps) / 10000.0
    quantile_rows = []
    long_short_rows = []

    tasks = [(factor, horizon, group_count) for factor in factor_names for horizon in horizons for group_count in quantile_groups]
    for factor, horizon, group_count in progress_iter(tasks, "Quantile backtest", total=len(tasks), enabled=show_progress):
        return_column = f"forward_return_{horizon}d"
        multiplier = multiplier_lookup.get((factor, int(horizon)), 1)
        previous_members = {}

        for date in dates:
            try:
                date_frame = combined[[factor, return_column]].xs(date, level="date")
            except KeyError:
                continue

            sample = date_frame.rename(columns={factor: "factor_value", return_column: "future_return"})
            sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < max(min_obs, group_count):
                continue

            sample["effective_factor_value"] = sample["factor_value"] * multiplier
            grouped_sample = _assign_quantile_groups(sample, group_count)
            if grouped_sample.empty:
                continue

            group_stats = {}
            for quantile in range(1, group_count + 1):
                group = grouped_sample[grouped_sample["quantile"] == quantile]
                if group.empty:
                    continue
                members = set(group.index.astype(str))
                previous = previous_members.get(quantile)
                turnover = 1.0 if previous is None else 1.0 - (len(members & previous) / len(members) if members else np.nan)
                raw_return = float(group["future_return"].mean())
                cost_return = cost_rate * turnover if turnover == turnover else np.nan
                net_return = raw_return - cost_return if cost_return == cost_return else np.nan
                previous_members[quantile] = members
                group_stats[quantile] = {
                    "raw_return": raw_return,
                    "turnover": turnover,
                    "cost_return": cost_return,
                    "net_return": net_return,
                    "n_obs": int(len(group)),
                }
                quantile_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "category": AVAILABLE_FACTOR_SPECS[factor].category,
                        "horizon": int(horizon),
                        "group_count": int(group_count),
                        "quantile": int(quantile),
                        "effective_factor_multiplier": int(multiplier),
                        "n_obs": int(len(group)),
                        "raw_return": raw_return,
                        "turnover": turnover,
                        "cost_return": cost_return,
                        "net_return": net_return,
                    }
                )

            if 1 in group_stats and group_count in group_stats:
                long_stats = group_stats[group_count]
                short_stats = group_stats[1]
                long_short_return = long_stats["raw_return"] - short_stats["raw_return"]
                long_short_turnover = long_stats["turnover"] + short_stats["turnover"]
                cost_return = cost_rate * long_short_turnover
                long_short_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "category": AVAILABLE_FACTOR_SPECS[factor].category,
                        "horizon": int(horizon),
                        "group_count": int(group_count),
                        "effective_factor_multiplier": int(multiplier),
                        "long_quantile": int(group_count),
                        "short_quantile": 1,
                        "long_return": long_stats["raw_return"],
                        "short_return": short_stats["raw_return"],
                        "long_short_return": long_short_return,
                        "long_turnover": long_stats["turnover"],
                        "short_turnover": short_stats["turnover"],
                        "long_short_turnover": long_short_turnover,
                        "cost_return": cost_return,
                        "net_long_short_return": long_short_return - cost_return,
                    }
                )

    quantile_returns = pd.DataFrame(quantile_rows, columns=quantile_columns)
    long_short_returns = pd.DataFrame(long_short_rows, columns=long_short_columns)
    if not quantile_returns.empty:
        quantile_returns["date"] = pd.to_datetime(quantile_returns["date"])
        quantile_returns = quantile_returns.sort_values(["factor", "horizon", "group_count", "date", "quantile"])
    if not long_short_returns.empty:
        long_short_returns["date"] = pd.to_datetime(long_short_returns["date"])
        long_short_returns = long_short_returns.sort_values(["factor", "horizon", "group_count", "date"])
    return quantile_returns, long_short_returns


def summarize_quantile_returns(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor",
        "category",
        "horizon",
        "group_count",
        "quantile",
        "date_count",
        "mean_n_obs",
        "mean_return",
        "return_std",
        "win_rate",
        "cumulative_return",
        "mean_turnover",
        "mean_cost_return",
        "net_mean_return",
        "net_cumulative_return",
    ]
    if quantile_returns.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (factor, horizon, group_count, quantile), group in quantile_returns.groupby(["factor", "horizon", "group_count", "quantile"]):
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "horizon": int(horizon),
                "group_count": int(group_count),
                "quantile": int(quantile),
                "date_count": int(len(group)),
                "mean_n_obs": float(group["n_obs"].mean()),
                "mean_return": float(group["raw_return"].mean()),
                "return_std": float(group["raw_return"].std()) if len(group) > 1 else np.nan,
                "win_rate": float((group["raw_return"] > 0).mean()),
                "cumulative_return": _safe_prod_return(group["raw_return"]),
                "mean_turnover": float(group["turnover"].mean()),
                "mean_cost_return": float(group["cost_return"].mean()),
                "net_mean_return": float(group["net_return"].mean()),
                "net_cumulative_return": _safe_prod_return(group["net_return"]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["factor", "horizon", "group_count", "quantile"])


def summarize_long_short_returns(long_short_returns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor",
        "category",
        "horizon",
        "group_count",
        "date_count",
        "mean_long_return",
        "mean_short_return",
        "mean_long_short_return",
        "long_short_std",
        "long_short_win_rate",
        "long_short_cumulative_return",
        "mean_long_short_turnover",
        "mean_cost_return",
        "net_mean_long_short_return",
        "net_long_short_cumulative_return",
    ]
    if long_short_returns.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (factor, horizon, group_count), group in long_short_returns.groupby(["factor", "horizon", "group_count"]):
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "horizon": int(horizon),
                "group_count": int(group_count),
                "date_count": int(len(group)),
                "mean_long_return": float(group["long_return"].mean()),
                "mean_short_return": float(group["short_return"].mean()),
                "mean_long_short_return": float(group["long_short_return"].mean()),
                "long_short_std": float(group["long_short_return"].std()) if len(group) > 1 else np.nan,
                "long_short_win_rate": float((group["long_short_return"] > 0).mean()),
                "long_short_cumulative_return": _safe_prod_return(group["long_short_return"]),
                "mean_long_short_turnover": float(group["long_short_turnover"].mean()),
                "mean_cost_return": float(group["cost_return"].mean()),
                "net_mean_long_short_return": float(group["net_long_short_return"].mean()),
                "net_long_short_cumulative_return": _safe_prod_return(group["net_long_short_return"]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["factor", "horizon", "group_count"])


def calculate_monotonicity_summary(quantile_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor",
        "category",
        "horizon",
        "group_count",
        "quantile_count",
        "monotonic_spearman",
        "net_monotonic_spearman",
        "is_monotonic_increasing",
        "is_net_monotonic_increasing",
        "top_bottom_mean_return",
        "net_top_bottom_mean_return",
    ]
    if quantile_summary.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (factor, horizon, group_count), group in quantile_summary.groupby(["factor", "horizon", "group_count"]):
        ordered = group.sort_values("quantile")
        quantiles = ordered["quantile"].astype(float)
        mean_returns = ordered["mean_return"].astype(float)
        net_mean_returns = ordered["net_mean_return"].astype(float)
        top = ordered[ordered["quantile"] == group_count]
        bottom = ordered[ordered["quantile"] == 1]
        rows.append(
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "horizon": int(horizon),
                "group_count": int(group_count),
                "quantile_count": int(len(ordered)),
                "monotonic_spearman": _spearman_corr(quantiles, mean_returns) if len(ordered) > 1 else np.nan,
                "net_monotonic_spearman": _spearman_corr(quantiles, net_mean_returns) if len(ordered) > 1 else np.nan,
                "is_monotonic_increasing": bool((mean_returns.diff().dropna() >= 0).all()) if len(ordered) > 1 else False,
                "is_net_monotonic_increasing": bool((net_mean_returns.diff().dropna() >= 0).all()) if len(ordered) > 1 else False,
                "top_bottom_mean_return": np.nan if top.empty or bottom.empty else float(top["mean_return"].iloc[0] - bottom["mean_return"].iloc[0]),
                "net_top_bottom_mean_return": np.nan if top.empty or bottom.empty else float(top["net_mean_return"].iloc[0] - bottom["net_mean_return"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["horizon", "group_count", "net_monotonic_spearman"], ascending=[True, True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-factor Rank IC analysis for HS300.")
    parser.add_argument("--stock-file", default="data/ZZ_1000.txt")
    parser.add_argument("--cache-dir", default="data/raw/tushare/price_history/raw_hfq_pct")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--factors", default="all", help="Comma-separated factor names or all.")
    parser.add_argument(
        "--factors-category",
        default="all",
        help="Factor category to test, or comma-separated categories. Use all to disable filtering.",
    )
    parser.add_argument("--forward-horizons", default="5,10,20")
    parser.add_argument("--min-obs", type=int, default=30, help="Minimum stocks required for one daily cross-section.")
    parser.add_argument("--rolling-window", type=int, default=60, help="Trading-day window for rolling IC.")
    parser.add_argument("--quantile-groups", default="5,10", help="Comma-separated quantile group counts for stratified backtest.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="One-way rebalance transaction cost in basis points.")
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
    factor_names = filter_factor_names_by_category(factor_names, args.factors_category)
    horizons = parse_int_list(args.forward_horizons)
    quantile_groups = parse_quantile_groups(args.quantile_groups)

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

    log_step(f"Building factor panel: {len(factor_names)} factors, category={args.factors_category}")
    factor_panel = build_factor_panel_with_progress(
        stock_data,
        factor_names=factor_names,
        show_progress=show_progress,
        max_workers=6,
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
    effective_mapping = build_effective_factor_mapping(selection_view)

    log_step(f"Running quantile backtest: groups={quantile_groups}, cost_bps={args.transaction_cost_bps}")
    quantile_returns, long_short_returns = calculate_quantile_backtest(
        factor_panel,
        forward_panel,
        factor_names,
        horizons,
        quantile_groups,
        ic_summary,
        min_obs=args.min_obs,
        transaction_cost_bps=args.transaction_cost_bps,
        show_progress=show_progress,
    )
    quantile_summary = summarize_quantile_returns(quantile_returns)
    long_short_summary = summarize_long_short_returns(long_short_returns)
    monotonicity_summary = calculate_monotonicity_summary(quantile_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stock_slug = stock_file_slug(stock_file)
    output_dir = PROJECT_ROOT / "output" / f"factor_ic_{stock_slug}_{timestamp}"
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
            "factor_effective_mapping.csv": effective_mapping,
            "factor_quantile_returns.csv": quantile_returns,
            "factor_quantile_summary.csv": quantile_summary,
            "factor_long_short_returns.csv": long_short_returns,
            "factor_long_short_summary.csv": long_short_summary,
            "factor_monotonicity_summary.csv": monotonicity_summary,
        },
        show_progress=show_progress,
    )

    summary = {
        "stock_file": str(stock_file.relative_to(PROJECT_ROOT)),
        "stock_file_slug": stock_slug,
        "cache_dir": str(cache_dir.relative_to(PROJECT_ROOT)),
        "start": args.start,
        "end": args.end,
        "loaded_symbol_count": len(stock_data),
        "factor_count": len(factor_names),
        "factors_category": args.factors_category,
        "factor_names": factor_names,
        "horizons": horizons,
        "quantile_groups": quantile_groups,
        "transaction_cost_bps": args.transaction_cost_bps,
        "min_obs": args.min_obs,
        "rolling_window": args.rolling_window,
        "winsorize": bool(args.winsorize),
        "standardize": bool(args.standardize),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "top_abs_icir": selection_view.sort_values("abs_icir", ascending=False).head(10).to_dict(orient="records"),
        "top_long_short": long_short_summary.sort_values(
            "net_mean_long_short_return",
            ascending=False,
        ).head(10).to_dict(orient="records"),
        "top_monotonicity": monotonicity_summary.sort_values(
            "net_monotonic_spearman",
            ascending=False,
        ).head(10).to_dict(orient="records"),
        "negative_effective_factors": effective_mapping[
            effective_mapping["effective_factor_multiplier"] == -1
        ].to_dict(orient="records"),
    }
    with open(output_dir / "ic_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

    log_step("Rank IC analysis completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
