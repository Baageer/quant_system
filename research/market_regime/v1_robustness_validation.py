"""Robustness validation of the v1 HS300-excess ranking strategy.

v1 (``hs300_excess_v1_outcome``) reports net +45.98%, Sharpe 0.90, maxDD -13.8% on
2022-07-20..2025-12-02.  Before that number is treated as a deliverable, two
independent questions have to be answered.

**A. Attribution** (``--mode attribution``) -- is the result concentrated in one
period?  The daily equity curve is decomposed by calendar year and by the causal
``market_state``, and a moving-block bootstrap over daily returns gives an
interval for total return and Sharpe, so "one lucky stretch" can be ruled in or
out directly.

**B. Walk-forward retrain** (``--mode walk_forward``) -- v1 fits its Ridge on
2005-2018 and never refreshes, so 2022-2025 is predicted by a model that has seen
nothing after 2018-12.  Here the model is re-fit at every fold boundary on an
expanding window (train <= boundary - validation_months, validation = the last
``validation_months`` before the boundary), and each fold predicts only the
``step_months`` after its boundary.  Fold predictions are stitched into one
continuous score table in the exact schema ``trade_outcome_ranking_evaluation``
consumes, so the downstream backtest is byte-for-byte the same engine and the two
arms are directly comparable.

Example::

    python -m research.market_regime.v1_robustness_validation --mode all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.market_environment import build_causal_market_features, load_benchmark_history
from research.market_regime.tradability_model import load_trend_feature_columns
from research.market_regime.trade_outcome_model import (
    load_outcome_split,
    target_columns,
    train_trade_outcome_experiment,
)


V1_MODEL_ID = "hs300_excess_v1_outcome"
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
TRADING_DAYS = 252
BOOTSTRAP_BLOCKS = (5, 20, 60)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260911


def load_equity(path: Path) -> pd.DataFrame:
    """Daily equity curve with a computed daily return column."""

    equity = pd.read_csv(path)
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity = equity.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    equity["daily_return"] = equity["portfolio_value"].pct_change().fillna(0.0)
    return equity


def load_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, dtype={"symbol": str})
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
    return trades.dropna(subset=["date"])


def _period_stats(returns: np.ndarray) -> Dict[str, float]:
    """Return/vol/Sharpe/maxDD over a set of daily returns.

    For the ``year`` key the days are contiguous, so max drawdown is a real
    path statistic.  For ``market_state`` the days are scattered, so its
    drawdown figures describe the concatenated sub-series, not a tradeable path.
    """

    if returns.size == 0:
        return {"total_return": np.nan, "annual_return": np.nan, "annual_volatility": np.nan,
                "sharpe_zero_rate": np.nan, "max_drawdown": np.nan, "days": 0}
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.min(cumulative / peak - 1.0))
    total_return = float(cumulative[-1] - 1.0)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else np.nan
    mean_annual = float(np.mean(returns) * TRADING_DAYS)
    return {
        "total_return": total_return,
        "annual_return": (1.0 + total_return) ** (TRADING_DAYS / returns.size) - 1.0,
        "annual_volatility": volatility,
        "sharpe_zero_rate": mean_annual / volatility if volatility and volatility > 0 else np.nan,
        "max_drawdown": max_drawdown,
        "days": int(returns.size),
    }


def period_breakdown(equity: pd.DataFrame, trades: pd.DataFrame, key: str) -> pd.DataFrame:
    """Net return / Sharpe / maxDD / turnover per period, keyed by year or market state."""

    working = equity.copy()
    if key == "market_state":
        benchmark = load_benchmark_history("data/benchmarks/000300.csv")
        market = build_causal_market_features(benchmark)[["market_state"]].reset_index()
        market.columns = ["date", "market_state"]
        working = working.merge(market, on="date", how="left")
        working = working.dropna(subset=["market_state"])
    elif key == "year":
        working["year"] = working["date"].dt.year
    else:
        raise ValueError("Unsupported period key: {}".format(key))

    trades_keyed = trades.copy()
    trades_keyed["year"] = trades_keyed["date"].dt.year
    if key == "market_state":
        trades_keyed = trades_keyed.merge(working[["date", "market_state"]], on="date", how="left")

    rows = []
    for value, group in working.groupby(key, sort=True):
        fills = trades_keyed[trades_keyed[key] == value]
        rows.append({
            key: value,
            **_period_stats(np.asarray(group["daily_return"], dtype=float)),
            "filled_trades": int(len(fills)),
            "turnover_multiple": float(fills["trade_value"].sum() / INITIAL_CAPITAL) if len(fills) else 0.0,
        })
    return pd.DataFrame(rows)


def block_bootstrap(returns: np.ndarray, blocks: Sequence[int] = BOOTSTRAP_BLOCKS,
                    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Moving-block bootstrap of total return / Sharpe / max drawdown.

    Daily strategy returns are autocorrelated through overlapping 10-day holding
    periods, so an i.i.d. bootstrap would understate the spread.  Resampling
    contiguous blocks preserves most of that structure.
    """

    rng = np.random.default_rng(seed)
    n = returns.size
    rows = []
    for block in blocks:
        n_blocks = int(np.ceil(n / block))
        max_start = n - block
        if max_start < 1:
            continue
        totals = np.empty(draws)
        sharpes = np.empty(draws)
        drawdowns = np.empty(draws)
        for draw in range(draws):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            sample = np.concatenate([returns[start:start + block] for start in starts])[:n]
            cumulative = np.cumprod(1.0 + sample)
            totals[draw] = cumulative[-1] - 1.0
            volatility = np.std(sample, ddof=1) * np.sqrt(TRADING_DAYS)
            sharpes[draw] = np.mean(sample) * TRADING_DAYS / volatility if volatility > 0 else np.nan
            peak = np.maximum.accumulate(cumulative)
            drawdowns[draw] = np.min(cumulative / peak - 1.0)
        rows.append({
            "block": int(block),
            "draws": int(draws),
            "total_return_p05": float(np.nanpercentile(totals, 5)),
            "total_return_p50": float(np.nanpercentile(totals, 50)),
            "total_return_p95": float(np.nanpercentile(totals, 95)),
            "total_return_below_zero": float(np.mean(totals <= 0)),
            "sharpe_p05": float(np.nanpercentile(sharpes, 5)),
            "sharpe_p50": float(np.nanpercentile(sharpes, 50)),
            "sharpe_p95": float(np.nanpercentile(sharpes, 95)),
            "sharpe_below_zero": float(np.mean(sharpes <= 0)),
            "max_drawdown_p50": float(np.nanpercentile(drawdowns, 50)),
            "max_drawdown_p95_worst": float(np.nanpercentile(drawdowns, 5)),
        })
    return pd.DataFrame(rows)


def walk_forward_folds(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                       step_months: int, validation_months: int) -> List[Dict[str, pd.Timestamp]]:
    """Expanding-window folds covering ``start``..``end``, one per ``step_months``."""

    boundaries = []
    boundary = start
    while boundary <= end:
        boundaries.append(boundary)
        boundary = boundary + pd.DateOffset(months=step_months)
    folds = []
    for boundary in boundaries:
        folds.append({
            "boundary": boundary,
            "train_end": boundary - pd.DateOffset(months=validation_months),
            "validation_start": boundary - pd.DateOffset(months=validation_months),
            "validation_end": boundary,
            "test_start": boundary,
            "test_end": min(boundary + pd.DateOffset(months=step_months), end),
        })
    return folds


def run_walk_forward(
    dataset_dir: Path, dataset_id: str, trend_model_id: str, horizon: int,
    start: str, end: str, step_months: int, validation_months: int,
    alpha_values: Sequence[float], risk_penalties: Sequence[float], top_k: float,
) -> Dict[str, object]:
    """Re-fit the dual-Ridge outcome model at every fold and stitch the predictions."""

    features = load_trend_feature_columns(dataset_dir, trend_model_id)
    splits = [
        load_outcome_split(dataset_dir, dataset_id, split, features, horizon)
        for split in ("train", "validation", "test")
    ]
    samples = pd.concat(splits, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    folds = walk_forward_folds(samples["date"], start_ts, end_ts, step_months, validation_months)

    test_parts, validation_parts, fold_rows = [], [], []
    for fold in folds:
        train = samples[samples["date"] <= fold["train_end"]]
        validation = samples[(samples["date"] > fold["validation_start"]) & (samples["date"] <= fold["validation_end"])]
        test = samples[(samples["date"] > fold["test_start"]) & (samples["date"] <= fold["test_end"])]
        if train.empty or validation.empty or test.empty:
            continue
        report = train_trade_outcome_experiment(
            train, validation, test, feature_columns=features, horizon=horizon,
            alpha_values=alpha_values, risk_penalties=risk_penalties, top_k=top_k,
        )
        selected = report["metrics"].query("split == 'test'").iloc[0]
        fold_rows.append({
            "boundary": fold["boundary"],
            "train_rows": int(len(train)),
            "train_end": fold["train_end"],
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "selected_alpha": report["selected_alpha"],
            "selected_risk_penalty": report["selected_risk_penalty"],
            "fold_test_excess_over_universe": float(selected["selected_excess_over_universe"]),
            "fold_test_rank_ic": float(selected["excess_return_rank_ic"]),
        })
        test_parts.append(report["test_predictions"])
        validation_parts.append(report["validation_predictions"])

    if not test_parts:
        raise ValueError("Walk-forward produced no folds")
    test_predictions = pd.concat(test_parts, ignore_index=True)
    # Fold test windows are disjoint by construction, so any duplicate here is a
    # real bug rather than an expected overlap -- fail loudly instead of deduping.
    duplicated_test = int(test_predictions.duplicated(["date", "symbol"]).sum())
    if duplicated_test:
        raise ValueError("Walk-forward test folds overlap: {} duplicate rows".format(duplicated_test))

    validation_predictions = pd.concat(validation_parts, ignore_index=True)
    # Validation windows *do* overlap (validation_months > step_months), so keep the
    # latest fold's score for each row -- that is the most recently trained model,
    # which is the causally correct one to carry forward.
    validation_predictions = validation_predictions.drop_duplicates(["date", "symbol"], keep="last")
    return {
        "feature_columns": features,
        "fold_table": pd.DataFrame(fold_rows),
        "test_predictions": test_predictions.sort_values(["date", "symbol"]).reset_index(drop=True),
        "validation_predictions": validation_predictions.sort_values(["date", "symbol"]).reset_index(drop=True),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="excess_hs300_v1")
    parser.add_argument("--trend-model-id", default="hs300_logistic_v2")
    parser.add_argument("--v1-model-id", default=V1_MODEL_ID)
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="v1_robustness")
    parser.add_argument("--mode", choices=("attribution", "walk_forward", "all"), default="all")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--start", default="2022-07-20")
    parser.add_argument("--end", default="2025-12-02")
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument("--risk-penalties", default="0,0.25,0.5,1")
    parser.add_argument("--top-k", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "v1_model_id": args.v1_model_id,
        "question": "is the v1 result a single-period artefact, and does it survive causal model refresh",
    }

    if args.mode in ("attribution", "all"):
        equity = load_equity(dataset_dir / "trade_outcome_ranking_{}_equity_net.csv".format(args.v1_model_id))
        trades = load_trades(dataset_dir / "trade_outcome_ranking_{}_trades_net.csv".format(args.v1_model_id))
        by_year = period_breakdown(equity, trades, "year")
        by_state = period_breakdown(equity, trades, "market_state")
        returns = np.asarray(equity["daily_return"], dtype=float)
        bootstrap = block_bootstrap(returns)
        # Concentration: how much of the total return survives dropping the single best period.
        year_returns = [np.asarray(equity.loc[equity["date"].dt.year == year, "daily_return"], dtype=float)
                        for year in sorted(equity["date"].dt.year.unique())]
        year_totals = sorted((float(np.prod(1.0 + r) - 1.0), int(y)) for r, y in
                             zip(year_returns, sorted(equity["date"].dt.year.unique())))
        full_total = float(np.prod(1.0 + returns) - 1.0)
        leave_one_out = []
        for total, year in year_totals:
            kept = np.concatenate([r for r, y in zip(year_returns, sorted(equity["date"].dt.year.unique()))
                                   if y != year])
            leave_one_out.append({"dropped_year": year, "dropped_year_return": total,
                                  "remaining_total_return": float(np.prod(1.0 + kept) - 1.0)})

        for name, table in (("attribution_by_year", by_year), ("attribution_by_state", by_state),
                            ("bootstrap", bootstrap), ("leave_one_out_year", pd.DataFrame(leave_one_out))):
            path = output_dir / "{}_{}.csv".format(args.run_id, name)
            table.to_csv(path, index=False, encoding="utf-8-sig")
            manifest[name + "_path"] = str(path)
        manifest["full_period_total_return"] = full_total

        pd.set_option("display.width", 220)
        print("=== v1 attribution by year ===")
        print(by_year.round(4).to_string(index=False))
        print("\n=== v1 attribution by market_state ===")
        print(by_state.round(4).to_string(index=False))
        print("\n=== moving-block bootstrap (daily returns) ===")
        print(bootstrap.round(4).to_string(index=False))
        print("\n=== leave-one-year-out total return ===")
        print(pd.DataFrame(leave_one_out).round(4).to_string(index=False))

    if args.mode in ("walk_forward", "all"):
        report = run_walk_forward(
            dataset_dir, args.dataset_id, args.trend_model_id, args.horizon,
            args.start, args.end, args.step_months, args.validation_months,
            [float(value) for value in args.alphas.split(",") if value.strip()],
            [float(value) for value in args.risk_penalties.split(",") if value.strip()],
            args.top_k,
        )
        walk_id = "{}_walkforward".format(args.run_id)
        report["test_predictions"].to_csv(
            output_dir / "trade_outcome_test_predictions_{}.csv".format(walk_id), index=False, encoding="utf-8-sig")
        report["validation_predictions"].to_csv(
            output_dir / "trade_outcome_validation_predictions_{}.csv".format(walk_id), index=False, encoding="utf-8-sig")
        fold_path = output_dir / "{}_{}.csv".format(args.run_id, "walkforward_folds")
        report["fold_table"].to_csv(fold_path, index=False, encoding="utf-8-sig")
        manifest["walkforward_model_id"] = walk_id
        manifest["walkforward_folds_path"] = str(fold_path)
        manifest["walkforward_folds"] = int(len(report["fold_table"]))
        print("\n=== walk-forward folds ===")
        print(report["fold_table"].round(4).to_string(index=False))
        print("\nWrote walk-forward predictions as model_id={} -- run the same backtest with:\n"
              "  python -m research.market_regime.trade_outcome_ranking_evaluation --model-id {}".format(walk_id, walk_id))

    manifest_path = output_dir / "{}_manifest.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nmanifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
