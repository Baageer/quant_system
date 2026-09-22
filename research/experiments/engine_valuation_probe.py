"""How much did the halt-day valuation defect move the published risk numbers?

``BacktestEngine._calculate_portfolio_value`` used to value the book from the
day's *execution* cross-section, which omits a held name that has no quote that
day.  ``load_local_price_history`` drops NaN-close rows, so a halted name has no
row at all rather than a NaN row -- the position silently vanished from the
equity curve for the session and came back the next.  That is a one-day phantom
loss the size of the name's weight: it understates Sharpe and overstates max
drawdown.  Total return is almost always untouched (gap on the final day is
needed to move it), which is why every earlier audit missed it.

Fixed in ``_build_optimized_data_structures`` (a ``valuation_matrix`` that
forward-fills the mark, used only for valuation -- execution still sees the
gap).  This probe replays an arm both ways and reports what changed.

Example::

    python -m research.experiments.engine_valuation_probe --arm hs300 --split test
    python -m research.experiments.engine_valuation_probe --arm zz1000 --split validation
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

import backtest.engine as engine_module
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


ARMS = {
    "hs300": "hs300_excess_v1_outcome",
    "zz1000": "zz1000_excess_v1_outcome",
}

REPORT_METRICS = ("total_return", "annual_volatility", "sharpe_zero_rate", "max_drawdown")


def _pre_fix_build(self, data: Dict[str, pd.DataFrame]) -> tuple:
    """The old behaviour: value the book from the execution cross-section."""

    date_to_symbols, price_matrix, signal_matrix, _ = _ORIGINAL_BUILD(self, data)
    return date_to_symbols, price_matrix, signal_matrix, price_matrix


_ORIGINAL_BUILD = engine_module.BacktestEngine._build_optimized_data_structures


def run_arm(
    dataset_dir: Path,
    model_id: str,
    split: str,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    config_path: str = "config/settings.yaml",
) -> Dict[str, object]:
    """Run the production arms with the fixed and the pre-fix valuation."""

    scores = load_outcome_scores(dataset_dir, model_id, split, horizon)
    raw_market_data, _ = build_rank_market_data(scores, None, SCORE_COLUMN)
    if not raw_market_data:
        raise ValueError("No price data for {} {}".format(model_id, split))
    selected = select_top_k(scores, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(raw_market_data, scores, selected)
    start_date, end_date = scores["date"].min(), scores["date"].max()
    target_notional = initial_capital / max_positions

    def run():
        return _run_engine(
            prepared,
            make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
            initial_capital,
            config_path,
            start_date,
            end_date,
        )

    results, trades = run()
    fixed = summarize_backtest("fixed", results, trades, initial_capital)

    engine_module.BacktestEngine._build_optimized_data_structures = _pre_fix_build
    try:
        old_results, old_trades = run()
    finally:
        engine_module.BacktestEngine._build_optimized_data_structures = _ORIGINAL_BUILD
    before = summarize_backtest("pre_fix", old_results, old_trades, initial_capital)

    v_fixed = results["portfolio_value"].astype(float)
    v_old = old_results["portfolio_value"].astype(float)
    gap = (v_fixed - v_old).abs()
    return {
        "model_id": model_id,
        "split": split,
        "sessions": int(len(v_fixed)),
        "stale_mark_days": int((gap > 1e-6).sum()),
        "worst_gap": float(gap.max()),
        "worst_gap_frac_of_capital": float(gap.max() / initial_capital),
        "before": {key: float(before[key]) for key in REPORT_METRICS},
        "after": {key: float(fixed[key]) for key in REPORT_METRICS},
    }


def _print_report(report: Dict[str, object]) -> None:
    print("\n{}  split={}  {} sessions, {} with a stale mark, worst {:,.0f} ({:.2%} of capital)".format(
        report["model_id"], report["split"], report["sessions"], report["stale_mark_days"],
        report["worst_gap"], report["worst_gap_frac_of_capital"]))
    print("  {:<20} {:>14} {:>14} {:>12}".format("metric", "pre-fix", "fixed", "delta"))
    for metric in REPORT_METRICS:
        before = report["before"][metric]
        after = report["after"][metric]
        print("  {:<20} {:>14.6f} {:>14.6f} {:>12.6f}".format(metric, before, after, after - before))


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="hs300")
    parser.add_argument("--model-id", default=None, help="Override the arm's model id.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    report = run_arm(
        Path(args.dataset_dir), args.model_id or ARMS[args.arm], args.split, args.top_k,
        args.horizon, args.max_positions, args.initial_capital, args.config_path,
    )
    _print_report(report)


if __name__ == "__main__":
    main()
