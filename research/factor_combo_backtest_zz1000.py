"""
20-day rebalance backtest for an effective multi-factor ZZ1000 long portfolio.

The script is intentionally research-oriented: it reuses the local price loader
and factor panel builder, constructs cross-sectional effective factor scores,
then simulates a non-overlapping rebalance portfolio with transaction costs.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_panel import AVAILABLE_FACTOR_SPECS, standardize_panel, winsorize_panel  # noqa: E402
from research.factor_ic_analysis_hs300 import parse_factor_names, stock_file_slug  # noqa: E402
from research.factor_quality_check_hs300 import (  # noqa: E402
    build_factor_panel_with_progress,
    export_reports,
    load_price_data,
    log_step,
    read_symbols,
)


DEFAULT_FACTORS = "vwap_distance,volatility_20,sma_ratio_60,ret_20d,rsrs_beta"


def parse_weights(text: str, factor_names: List[str]) -> Dict[str, float]:
    if text == "equal":
        weight = 1.0 / len(factor_names)
        return {factor: weight for factor in factor_names}

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != len(factor_names):
        raise ValueError(f"Expected {len(factor_names)} factor weights, got {len(values)}.")
    if any(value < 0 for value in values):
        raise ValueError("Factor weights must be non-negative.")

    total = sum(values)
    if total <= 0:
        raise ValueError("At least one factor weight must be positive.")
    return {factor: value / total for factor, value in zip(factor_names, values)}


def load_effective_multipliers(
    factor_names: List[str],
    mapping_file: Optional[Path],
    default_negative: bool,
) -> Dict[str, int]:
    if mapping_file is None:
        if default_negative:
            return {factor: -1 for factor in factor_names}
        return {factor: 1 for factor in factor_names}

    mapping = pd.read_csv(mapping_file)
    required = {"raw_factor", "effective_factor_multiplier"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file missing columns: {sorted(missing)}")

    lookup = {
        str(row["raw_factor"]): int(row["effective_factor_multiplier"])
        for _, row in mapping.iterrows()
    }
    unknown = [factor for factor in factor_names if factor not in lookup]
    if unknown:
        raise ValueError(f"Mapping file has no multiplier for factors: {unknown}")
    return {factor: lookup[factor] for factor in factor_names}


def build_close_panel(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, data in sorted(stock_data.items()):
        frame = pd.DataFrame(
            {
                "date": data.index,
                "symbol": str(symbol),
                "close": data["close"].astype(float).values,
            }
        )
        frames.append(frame.set_index(["date", "symbol"]))
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]), columns=["close"])
    result = pd.concat(frames).sort_index()
    result.index = result.index.set_names(["date", "symbol"])
    return result


def build_rebalance_dates(close_panel: pd.DataFrame, rebalance_days: int, offset: int) -> List[pd.Timestamp]:
    all_dates = sorted(pd.to_datetime(close_panel.index.get_level_values("date").unique()))
    if offset < 0 or offset >= rebalance_days:
        raise ValueError("--rebalance-offset must be between 0 and rebalance_days - 1.")
    return all_dates[offset::rebalance_days]


def _safe_period_returns(close_panel: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    try:
        start_close = close_panel.xs(start_date, level="date")["close"]
        end_close = close_panel.xs(end_date, level="date")["close"]
    except KeyError:
        return pd.Series(dtype=float)
    returns = end_close / start_close - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.index = returns.index.astype(str)
    return returns


def _portfolio_return(weights: Dict[str, float], returns: pd.Series) -> tuple:
    if not weights or returns.empty:
        return np.nan, 0, 0

    weight_series = pd.Series(weights, dtype=float)
    common_symbols = weight_series.index.intersection(returns.index)
    missing_count = int(len(weight_series) - len(common_symbols))
    if len(common_symbols) == 0:
        return np.nan, 0, missing_count

    valid_weights = weight_series.loc[common_symbols]
    valid_weights = valid_weights / valid_weights.sum()
    return float((valid_weights * returns.loc[common_symbols]).sum()), int(len(common_symbols)), missing_count


def _one_way_turnover(previous_weights: Optional[Dict[str, float]], new_weights: Dict[str, float]) -> float:
    if not new_weights:
        return np.nan
    if previous_weights is None:
        return 1.0

    symbols = sorted(set(previous_weights) | set(new_weights))
    previous = pd.Series({symbol: previous_weights.get(symbol, 0.0) for symbol in symbols}, dtype=float)
    new = pd.Series({symbol: new_weights.get(symbol, 0.0) for symbol in symbols}, dtype=float)
    return float(0.5 * (new - previous).abs().sum())


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return np.nan
    cumulative_with_initial = pd.concat([pd.Series([0.0]), cumulative.reset_index(drop=True)], ignore_index=True)
    wealth = 1.0 + cumulative_with_initial
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def calculate_performance_summary(
    returns: pd.DataFrame,
    rebalance_days: int,
    top_fraction: float,
    top_n: Optional[int],
    transaction_cost_bps: float,
) -> pd.DataFrame:
    columns = [
        "period_count",
        "rebalance_days",
        "annual_periods",
        "top_fraction",
        "top_n",
        "transaction_cost_bps",
        "mean_gross_return",
        "mean_net_return",
        "net_return_std",
        "net_win_rate",
        "total_net_return",
        "annualized_net_return",
        "annualized_net_volatility",
        "net_sharpe",
        "max_drawdown",
        "benchmark_total_return",
        "excess_total_return",
        "mean_turnover",
        "mean_selected_count",
        "mean_valid_return_count",
    ]
    if returns.empty:
        return pd.DataFrame(columns=columns)

    annual_periods = 252.0 / float(rebalance_days)
    net_returns = returns["net_return"].dropna()
    benchmark_returns = returns["benchmark_return"].dropna()
    total_net_return = float((1.0 + net_returns).prod() - 1.0) if len(net_returns) else np.nan
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0) if len(benchmark_returns) else np.nan
    annualized_net_return = (
        float((1.0 + total_net_return) ** (annual_periods / len(net_returns)) - 1.0)
        if len(net_returns) and total_net_return > -1
        else np.nan
    )
    net_std = float(net_returns.std()) if len(net_returns) > 1 else np.nan
    annualized_net_volatility = net_std * math.sqrt(annual_periods) if net_std == net_std else np.nan
    net_sharpe = (
        float(net_returns.mean() / net_std * math.sqrt(annual_periods))
        if net_std == net_std and net_std != 0
        else np.nan
    )

    row = {
        "period_count": int(len(returns)),
        "rebalance_days": int(rebalance_days),
        "annual_periods": annual_periods,
        "top_fraction": float(top_fraction),
        "top_n": "" if top_n is None else int(top_n),
        "transaction_cost_bps": float(transaction_cost_bps),
        "mean_gross_return": float(returns["gross_return"].mean()),
        "mean_net_return": float(returns["net_return"].mean()),
        "net_return_std": net_std,
        "net_win_rate": float((returns["net_return"] > 0).mean()),
        "total_net_return": total_net_return,
        "annualized_net_return": annualized_net_return,
        "annualized_net_volatility": annualized_net_volatility,
        "net_sharpe": net_sharpe,
        "max_drawdown": _max_drawdown(returns["net_cumulative_return"]),
        "benchmark_total_return": benchmark_total_return,
        "excess_total_return": total_net_return - benchmark_total_return,
        "mean_turnover": float(returns["turnover"].mean()),
        "mean_selected_count": float(returns["selected_count"].mean()),
        "mean_valid_return_count": float(returns["valid_return_count"].mean()),
    }
    return pd.DataFrame([row], columns=columns)


def build_combo_scores(
    factor_panel: pd.DataFrame,
    factor_names: List[str],
    multipliers: Dict[str, int],
    factor_weights: Dict[str, float],
) -> pd.DataFrame:
    frame = factor_panel[factor_names].copy()
    effective_columns = []
    for factor in factor_names:
        column = f"{factor}_effective"
        frame[column] = frame[factor] * multipliers[factor]
        effective_columns.append(column)
    frame["combo_score"] = 0.0
    for factor in factor_names:
        frame["combo_score"] = frame["combo_score"] + frame[f"{factor}_effective"] * factor_weights[factor]
    return frame[effective_columns + ["combo_score"]]


def select_portfolio(
    date_scores: pd.DataFrame,
    top_fraction: float,
    top_n: Optional[int],
    min_obs: int,
) -> pd.DataFrame:
    sample = date_scores.replace([np.inf, -np.inf], np.nan).dropna(subset=["combo_score"]).copy()
    sample = sample.sort_values("combo_score", ascending=False)
    if len(sample) < min_obs:
        return pd.DataFrame()

    selected_count = top_n if top_n is not None else max(1, int(math.ceil(len(sample) * top_fraction)))
    selected_count = min(selected_count, len(sample))
    selected = sample.head(selected_count).copy()
    selected["weight"] = 1.0 / float(selected_count)
    selected["rank"] = np.arange(1, selected_count + 1)
    return selected


def run_combo_backtest(
    factor_scores: pd.DataFrame,
    close_panel: pd.DataFrame,
    rebalance_days: int,
    top_fraction: float,
    top_n: Optional[int],
    min_obs: int,
    transaction_cost_bps: float,
    rebalance_offset: int,
) -> tuple:
    cost_rate = float(transaction_cost_bps) / 10000.0
    rebalance_dates = build_rebalance_dates(close_panel, rebalance_days, rebalance_offset)
    rows = []
    holding_rows = []
    previous_weights = None

    for position, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[position + 1]
        try:
            date_scores = factor_scores.xs(date, level="date")
        except KeyError:
            continue

        selected = select_portfolio(date_scores, top_fraction=top_fraction, top_n=top_n, min_obs=min_obs)
        if selected.empty:
            continue

        weights = {str(symbol): float(weight) for symbol, weight in selected["weight"].items()}
        period_returns = _safe_period_returns(close_panel, date, next_date)
        gross_return, valid_return_count, missing_return_count = _portfolio_return(weights, period_returns)
        if gross_return != gross_return:
            continue

        benchmark_return = float(period_returns.mean()) if len(period_returns) else np.nan
        turnover = _one_way_turnover(previous_weights, weights)
        cost_return = cost_rate * turnover if turnover == turnover else np.nan
        net_return = gross_return - cost_return if cost_return == cost_return else np.nan
        previous_weights = weights

        rows.append(
            {
                "date": date,
                "next_date": next_date,
                "holding_days": int((next_date - date).days),
                "trading_day_holding_period": int(rebalance_days),
                "score_universe_count": int(len(date_scores.dropna(subset=["combo_score"]))),
                "selected_count": int(len(selected)),
                "valid_return_count": int(valid_return_count),
                "missing_return_count": int(missing_return_count),
                "gross_return": gross_return,
                "benchmark_return": benchmark_return,
                "excess_gross_return": gross_return - benchmark_return if benchmark_return == benchmark_return else np.nan,
                "turnover": turnover,
                "cost_return": cost_return,
                "net_return": net_return,
                "excess_net_return": net_return - benchmark_return if benchmark_return == benchmark_return else np.nan,
            }
        )

        holding_frame = selected.reset_index().rename(columns={"index": "symbol"})
        if "symbol" not in holding_frame.columns:
            holding_frame = holding_frame.rename(columns={holding_frame.columns[0]: "symbol"})
        holding_frame["date"] = date
        holding_frame["next_date"] = next_date
        holding_frame["selected_count"] = int(len(selected))
        holding_rows.append(holding_frame)

    returns = pd.DataFrame(rows)
    holdings = pd.concat(holding_rows, ignore_index=True) if holding_rows else pd.DataFrame()
    if not returns.empty:
        returns["date"] = pd.to_datetime(returns["date"])
        returns["next_date"] = pd.to_datetime(returns["next_date"])
        returns = returns.sort_values("date").reset_index(drop=True)
        returns["gross_cumulative_return"] = (1.0 + returns["gross_return"]).cumprod() - 1.0
        returns["net_cumulative_return"] = (1.0 + returns["net_return"]).cumprod() - 1.0
        returns["benchmark_cumulative_return"] = (1.0 + returns["benchmark_return"]).cumprod() - 1.0
        returns["excess_net_cumulative_return"] = (1.0 + returns["excess_net_return"]).cumprod() - 1.0
    if not holdings.empty:
        holdings["date"] = pd.to_datetime(holdings["date"])
        holdings["next_date"] = pd.to_datetime(holdings["next_date"])
        holdings = holdings.sort_values(["date", "rank"]).reset_index(drop=True)
    return returns, holdings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-day ZZ1000 effective factor combo backtest.")
    parser.add_argument("--stock-file", default="data/ZZ_1000.txt")
    parser.add_argument("--cache-dir", default="data/raw/tushare/price_history/raw_hfq_pct")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--factors", default=DEFAULT_FACTORS, help="Comma-separated raw factor names.")
    parser.add_argument("--factor-weights", default="equal", help="equal or comma-separated non-negative weights.")
    parser.add_argument("--mapping-file", default="", help="Optional factor_effective_mapping.csv from IC analysis.")
    parser.add_argument(
        "--default-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use negative direction for factors when no mapping file is supplied.",
    )
    parser.add_argument("--rebalance-days", type=int, default=20)
    parser.add_argument("--rebalance-offset", type=int, default=0)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=0, help="Fixed selected stock count. Use 0 to select by top fraction.")
    parser.add_argument("--min-obs", type=int, default=30)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--winsorize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--factor-workers", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    stock_file = PROJECT_ROOT / args.stock_file
    cache_dir = PROJECT_ROOT / args.cache_dir
    mapping_file = PROJECT_ROOT / args.mapping_file if args.mapping_file else None
    factor_names = parse_factor_names(args.factors)
    factor_weights = parse_weights(args.factor_weights, factor_names)
    multipliers = load_effective_multipliers(factor_names, mapping_file, default_negative=args.default_negative)
    top_n = args.top_n if args.top_n and args.top_n > 0 else None
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1].")
    if args.rebalance_days <= 0:
        raise ValueError("--rebalance-days must be positive.")

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

    log_step(f"Building factor panel: {factor_names}")
    factor_panel = build_factor_panel_with_progress(
        stock_data,
        factor_names=factor_names,
        show_progress=show_progress,
        max_workers=args.factor_workers,
    )
    if args.winsorize:
        log_step("Applying cross-sectional winsorization")
        factor_panel = winsorize_panel(factor_panel, factor_columns=factor_names)
    if args.standardize:
        log_step("Applying cross-sectional standardization")
        factor_panel = standardize_panel(factor_panel, factor_columns=factor_names)

    log_step("Building close panel and combo scores")
    close_panel = build_close_panel(stock_data)
    combo_scores = build_combo_scores(factor_panel, factor_names, multipliers, factor_weights)

    log_step(
        f"Running combo backtest: rebalance_days={args.rebalance_days}, "
        f"top_fraction={args.top_fraction}, top_n={top_n}, cost_bps={args.transaction_cost_bps}"
    )
    combo_returns, combo_holdings = run_combo_backtest(
        combo_scores,
        close_panel,
        rebalance_days=args.rebalance_days,
        top_fraction=args.top_fraction,
        top_n=top_n,
        min_obs=args.min_obs,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_offset=args.rebalance_offset,
    )
    combo_summary = calculate_performance_summary(
        combo_returns,
        rebalance_days=args.rebalance_days,
        top_fraction=args.top_fraction,
        top_n=top_n,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    factor_config = pd.DataFrame(
        [
            {
                "factor": factor,
                "category": AVAILABLE_FACTOR_SPECS[factor].category,
                "effective_factor_multiplier": int(multipliers[factor]),
                "factor_weight": float(factor_weights[factor]),
                "effective_factor_formula": f"-{factor}" if multipliers[factor] == -1 else factor,
            }
            for factor in factor_names
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"factor_combo_{stock_file_slug(stock_file)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"Exporting combo reports to {output_dir.relative_to(PROJECT_ROOT)}")
    export_reports(
        output_dir,
        {
            "price_load_report.csv": load_report,
            "combo_factor_config.csv": factor_config,
            "combo_rebalance_returns.csv": combo_returns,
            "combo_holdings.csv": combo_holdings,
            "combo_summary.csv": combo_summary,
        },
        show_progress=show_progress,
    )

    summary = {
        "stock_file": str(stock_file.relative_to(PROJECT_ROOT)),
        "stock_file_slug": stock_file_slug(stock_file),
        "cache_dir": str(cache_dir.relative_to(PROJECT_ROOT)),
        "start": args.start,
        "end": args.end,
        "loaded_symbol_count": len(stock_data),
        "factor_count": len(factor_names),
        "factor_names": factor_names,
        "factor_weights": factor_weights,
        "effective_multipliers": multipliers,
        "mapping_file": "" if mapping_file is None else str(mapping_file.relative_to(PROJECT_ROOT)),
        "rebalance_days": args.rebalance_days,
        "rebalance_offset": args.rebalance_offset,
        "top_fraction": args.top_fraction,
        "top_n": top_n,
        "transaction_cost_bps": args.transaction_cost_bps,
        "winsorize": bool(args.winsorize),
        "standardize": bool(args.standardize),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "performance": combo_summary.to_dict(orient="records")[0] if not combo_summary.empty else {},
    }
    with open(output_dir / "combo_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

    log_step("Combo backtest completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

# python research/factor_combo_backtest_zz1000.py --mapping-file output/factor_ic_zz1000_20260611_135734/factor_effective_mapping.csv --no-progress
if __name__ == "__main__":
    main()
