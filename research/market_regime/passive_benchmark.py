"""Compare a strategy's equity curve against the real 沪深300 index.

Every evaluation module in this package reports an "excess vs passive" number,
and that passive arm is ``passive_equal_weight_sample_net``: 20 symbols picked by
an alphabetical stride across the scored universe, bought once and never
rebalanced (``probability_trade_evaluation.select_passive_symbols``).  That is a
*convenience* baseline, not a benchmark -- it is 20 arbitrary names rather than
the 300-name index the model's own excess label is measured against, it is held
statically, and it is deterministic-arbitrary rather than an unbiased sample.

The label side was always right: ``future_excess_return_{h}d`` is already the
return over the real 000300 index (``excess_target_dataset.load_hs300_benchmark``).
So a strategy trained against the index was being *reported* against a basket.
That mismatch is what this module closes.

Two uses:

* ``index_arm_row`` -- an index row shaped exactly like ``summarize_backtest``'s
  output, so it can be appended to any ``_backtest_metrics.csv`` and read
  alongside the strategy arms.  It synthesizes a buy-and-hold equity curve at the
  same ``initial_capital`` over the strategy's *own* session index and hands it
  to ``summarize_backtest``, so annualization, volatility, Sharpe and drawdown
  use one code path and cannot drift apart.
* ``audit_equity_vs_index`` -- the retrospective check: take an already-written
  equity curve and ask what it would have looked like against the index.

**Dividend convention (read this before interpreting any excess).**  Stock prices
come from the ``raw_hfq_pct`` adjustment, i.e. they are 后复权 and therefore
include reinvested dividends.  The 000300 CSV is a *price* index and does not.
The strategy side is credited with roughly the index dividend yield (~2%/yr for
HS300) that the benchmark never receives, so every "excess vs index" number here
is flattered by that amount.  A result that only clears the index by a couple of
points a year is inside this wedge and should be treated as no result.

Example::

    python -m research.market_regime.passive_benchmark --audit
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.excess_target_dataset import load_hs300_benchmark
from research.market_regime.probability_trade_evaluation import summarize_backtest
from research.market_regime.output_paths import DATASET_DIR


DEFAULT_BENCHMARK_FILE = "data/benchmarks/000300.csv"
INDEX_ARM_NAME = "hs300_index_buy_hold"
#: Sessions the index may be missing from the strategy's own session index before
#: the window is rejected rather than silently compared over a different span.
MAX_MISSING_SESSIONS = 5


def load_index_levels(path=DEFAULT_BENCHMARK_FILE) -> pd.Series:
    """Date-indexed close level of the benchmark index."""

    return load_hs300_benchmark(Path(path))


def align_levels(levels: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Restrict ``levels`` to ``sessions``, rejecting windows it does not cover.

    The two series are built from the same A-share calendar, but a strategy window
    that runs past the end of the benchmark file (or starts before it) would
    otherwise be compared over a shorter span than the strategy traded.  A few
    isolated gaps are forward-filled; a window with more than
    ``MAX_MISSING_SESSIONS`` of them is an error, not a comparison.
    """

    sessions = pd.DatetimeIndex(pd.to_datetime(sessions))
    if len(sessions) == 0:
        raise ValueError("sessions is empty")
    if levels is None or len(levels) == 0:
        raise ValueError("index level series is empty")
    missing = sessions.difference(levels.index)
    # An endpoint the index does not have is a window mismatch, never a gap to
    # fill: back-filling the tail would invent a flat return for a stretch the
    # index simply does not reach, and report the strategy as beating nothing.
    endpoints = [session for session in (sessions.min(), sessions.max()) if session in set(missing)]
    if endpoints:
        raise ValueError(
            "index is missing the window endpoint {} of [{}, {}]".format(
                endpoints[0].date(), sessions.min().date(), sessions.max().date())
        )
    if len(missing) > MAX_MISSING_SESSIONS:
        raise ValueError(
            "index is missing {} of {} sessions in [{}, {}]; first missing {}".format(
                len(missing), len(sessions), sessions.min().date(), sessions.max().date(),
                missing[0].date() if len(missing) else "-",
            )
        )
    window = levels.reindex(sessions).ffill()
    if window.isna().any():
        raise ValueError("index level series has no value covering the window")
    return window


def index_arm_row(
    levels: pd.Series,
    sessions: pd.DatetimeIndex,
    initial_capital: float = 500000.0,
    name: str = INDEX_ARM_NAME,
) -> Dict[str, object]:
    """A buy-and-hold index row shaped like ``summarize_backtest``'s output.

    Trade and cost columns are zero by construction: an index cannot be traded
    through the engine, so it fills nothing and costs nothing.  That makes the
    comparison gross-of-cost on the benchmark side and net on the strategy side
    -- which is the conservative direction, since it favours the index.
    """

    window = align_levels(levels, sessions)
    curve = (window / window.iloc[0] * float(initial_capital)).rename("portfolio_value")
    row = summarize_backtest(name, curve.to_frame(), pd.DataFrame(), float(initial_capital))
    row["turnover_multiple"] = 0.0
    return row


def append_index_arm(
    metrics: pd.DataFrame,
    results: pd.DataFrame,
    initial_capital: float,
    benchmark_file=DEFAULT_BENCHMARK_FILE,
    levels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Append the real-index buy-and-hold row to a ``_backtest_metrics`` table.

    ``results`` is any engine equity frame; its ``portfolio_value`` index sets
    the window, so the index is measured over exactly the sessions the strategy
    traded.  The existing ``passive_equal_weight_sample_net`` row is deliberately
    left in place -- this adds a benchmark, it does not remove the diagnostic.

    ``levels`` overrides the file load; it exists so callers that already hold
    the index series (and tests) can skip the disk read.
    """

    if levels is None:
        levels = load_index_levels(benchmark_file)
    row = index_arm_row(levels, results.index, float(initial_capital))
    return pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)


def benchmark_comparison(
    metrics: pd.DataFrame,
    strategy: str = None,
    sample_arm: str = "passive_equal_weight_sample_net",
    index_arm: str = INDEX_ARM_NAME,
) -> Dict[str, float]:
    """Strategy excess over the index and over the equal-weight sample.

    The two are reported side by side so the size of the old
    convenience-baseline error stays visible instead of being quietly replaced.
    ``strategy`` defaults to the first row of ``metrics``.
    """

    table = metrics.set_index("strategy")
    if strategy is None:
        strategy = str(metrics.iloc[0]["strategy"])
    missing = [name for name in (strategy, index_arm) if name not in table.index]
    if missing:
        raise ValueError("metrics table is missing arms: {}".format(missing))
    net = table.loc[strategy]
    index_total = float(table.loc[index_arm, "total_return"])
    out = {
        "strategy": strategy,
        "net_total_return": float(net["total_return"]),
        "index_total_return": index_total,
        "net_total_return_excess_vs_index": float(net["total_return"] - index_total),
        "index_max_drawdown": float(table.loc[index_arm, "max_drawdown"]),
        "index_sharpe": float(table.loc[index_arm, "sharpe_zero_rate"]),
    }
    if sample_arm in table.index:
        sample_total = float(table.loc[sample_arm, "total_return"])
        out["sample_total_return"] = sample_total
        out["net_total_return_excess_vs_sample"] = float(net["total_return"] - sample_total)
        out["sample_minus_index_gap"] = float(sample_total - index_total)
    return out


def index_window_return(levels: pd.Series, start, end) -> float:
    """Cumulative index return between two dates."""

    window = levels.loc[(levels.index >= pd.Timestamp(start)) & (levels.index <= pd.Timestamp(end))]
    if len(window) < 2:
        raise ValueError("index has fewer than two sessions in [{}, {}]".format(start, end))
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def excess_vs_index(equity: pd.DataFrame, levels: pd.Series, strategy_row: Optional[Dict] = None) -> Dict[str, float]:
    """Strategy-vs-index totals: return, drawdown and Sharpe deltas.

    ``equity`` is a backtest results frame (``date``/``portfolio_value`` columns,
    or a date index).  ``strategy_row`` may be passed to avoid recomputing the
    strategy's own metrics.
    """

    if "date" in getattr(equity, "columns", []):
        frame = equity.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    else:
        frame = equity.copy()
        frame.index = pd.DatetimeIndex(frame.index)
    values = frame["portfolio_value"].astype(float)
    # A backtest equity curve opens flat at the initial capital, so its first
    # mark *is* the capital base -- no need to be told it separately.
    capital = float(values.iloc[0])
    strategy = strategy_row or summarize_backtest("strategy", frame, pd.DataFrame(), capital)
    index_row = index_arm_row(levels, values.index, capital)
    return {
        "net_total_return": float(strategy["total_return"]),
        "index_total_return": float(index_row["total_return"]),
        "net_total_return_excess_vs_index": float(strategy["total_return"] - index_row["total_return"]),
        "index_max_drawdown": float(index_row["max_drawdown"]),
        "max_drawdown_excess_vs_index": float(strategy["max_drawdown"] - index_row["max_drawdown"]),
        "index_sharpe": float(index_row["sharpe_zero_rate"]),
        "sharpe_excess_vs_index": float(strategy["sharpe_zero_rate"] - index_row["sharpe_zero_rate"]),
        "sessions": int(len(values)),
        "initial_capital": capital,
    }


def audit_equity_vs_index(
    equity_path: Path,
    levels: pd.Series,
    initial_capital: Optional[float] = None,
) -> Dict[str, object]:
    """Compare one written equity curve against the index over its own window."""

    frame = pd.read_csv(equity_path)
    if "portfolio_value" not in frame.columns:
        raise ValueError("{} has no portfolio_value column".format(equity_path))
    frame["date"] = pd.to_datetime(frame["date"])
    values = frame.set_index("date")["portfolio_value"].astype(float)
    capital = float(initial_capital) if initial_capital is not None else float(values.iloc[0])
    strategy = summarize_backtest("strategy", frame.set_index("date"), pd.DataFrame(), capital)
    index_row = index_arm_row(levels, values.index, capital)
    return {
        "equity_path": str(equity_path),
        "start_date": strategy["start_date"],
        "end_date": strategy["end_date"],
        "sessions": int(len(values)),
        "strategy_total_return": float(strategy["total_return"]),
        "index_total_return": float(index_row["total_return"]),
        "excess_vs_index": float(strategy["total_return"] - index_row["total_return"]),
        "strategy_max_drawdown": float(strategy["max_drawdown"]),
        "index_max_drawdown": float(index_row["max_drawdown"]),
        "strategy_sharpe": float(strategy["sharpe_zero_rate"]),
        "index_sharpe": float(index_row["sharpe_zero_rate"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity-dir", default=DATASET_DIR)
    parser.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--output", default=DATASET_DIR + "/passive_benchmark_audit.csv")
    parser.add_argument("--audit", action="store_true",
                        help="audit every *_equity_net.csv under --equity-dir")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    levels = load_index_levels(args.benchmark_file)
    if not args.audit:
        raise SystemExit("nothing to do: pass --audit")
    rows = []
    for path in sorted(Path(args.equity_dir).glob("*_equity_net.csv")):
        try:
            rows.append(audit_equity_vs_index(path, levels))
        except ValueError as exc:
            print("[skip] {}: {}".format(path.name, exc), flush=True)
    table = pd.DataFrame(rows).sort_values("excess_vs_index", ascending=False)
    columns = ["strategy_total_return", "index_total_return", "excess_vs_index",
               "strategy_max_drawdown", "index_max_drawdown", "strategy_sharpe", "index_sharpe"]
    pd.set_option("display.width", 240)
    print("\n=== every equity curve vs the real 000300 index, over its own window ===")
    print(table[["equity_path"] + columns].round(4).to_string(index=False))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nrows={} written={}".format(len(table), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
