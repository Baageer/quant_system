"""Is the walk-forward arm actually better, or did it just use a narrower Top-K?

``v1_robustness_validation_summary.md`` §5 recommends making annual walk-forward
retraining the production default, on the strength of +47.90% vs +45.98% and
Sharpe 0.937 vs 0.898.  But §6 bullet 1 flags that this is **not a like-for-like
comparison**: each arm's Top-K was chosen on its own (different) validation
window, and the walk-forward arm happened to land on 5% while v1 landed on 10%.

Worse, the two "validation" windows are not the same thing at all:

* v1 grid validation      : 2019-07-17 -> 2021-12-02
* walk-forward grid        : 2020-07-21 -> 2025-07-18  (the 4 folds' 24-month
  pre-boundary windows concatenated -- this **overlaps the test split**, so the
  walk-forward arm's Top-K was picked using test-period data)

This script removes both confounds: it backtests **both models on the same test
split at the same forced Top-K**, so the only thing that differs is the score
column.  Whatever gap survives is attributable to the model.

Example::

    python -m research.market_regime.walkforward_topk_ab
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores

INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
DEFAULT_MODELS = "hs300_excess_v1_outcome,v1_robustness_walkforward"


def backtest_test_split(
    model_id: str,
    dataset_dir: Path,
    top_k: float,
    horizon: int,
    config_path: str,
) -> Dict[str, float]:
    """One test-split backtest at a forced Top-K, identical construction for every arm."""

    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw, skipped = build_rank_market_data(test, None, SCORE_COLUMN)
    if not raw:
        raise ValueError("No test price data for {}".format(model_id))

    selected = select_top_k(test, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(raw, test, selected)
    results, trades = _run_engine(
        prepared,
        make_rank_strategy(MAX_POSITIONS, INITIAL_CAPITAL / MAX_POSITIONS, min_holding_days=horizon),
        INITIAL_CAPITAL,
        config_path,
        test["date"].min(),
        test["date"].max(),
    )
    stats = summarize_backtest("net", results, trades, INITIAL_CAPITAL)
    stats.pop("strategy", None)
    return {
        "model_id": model_id,
        "top_k": top_k,
        "window_start": str(pd.Timestamp(test["date"].min()).date()),
        "window_end": str(pd.Timestamp(test["date"].max()).date()),
        "skipped_symbols": len(skipped),
        **stats,
    }


def run(
    dataset_dir: Path,
    models: Sequence[str],
    top_k_grid: Sequence[float],
    horizon: int = 10,
    config_path: str = "config/settings.yaml",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    rows = []
    for model_id in models:
        for top_k in top_k_grid:
            row = backtest_test_split(model_id, dataset_dir, top_k, horizon, config_path)
            rows.append(row)

    frame = pd.DataFrame(rows)
    columns = [
        "model_id",
        "top_k",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe_zero_rate",
        "max_drawdown",
        "turnover_multiple",
        "filled_trade_count",
    ]
    frame = frame[[c for c in columns if c in frame.columns]]

    print("\n=== test-split backtest, identical construction, forced Top-K ===")
    print(frame.round(4).to_string(index=False))

    pivot = frame.pivot(index="top_k", columns="model_id", values="total_return")
    if len(models) == 2 and pivot.shape[1] == 2:
        left, right = pivot.columns[0], pivot.columns[1]
        print("\n--- matched Top-K: total return difference ({} - {}) ---".format(right, left))
        for top_k, row in pivot.iterrows():
            print("    top_k {:<6}  {} {:>8.2%}   {} {:>8.2%}   diff {:>+8.2%}".format(
                top_k, left, row[left], right, row[right], row[right] - row[left]))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "walkforward_topk_ab.csv", index=False, encoding="utf-8-sig")
    return frame


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--top-k-grid", default="0.05,0.10")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--output-dir", default=DATASET_DIR)
    args = parser.parse_args(argv)
    run(
        Path(args.dataset_dir),
        [m.strip() for m in args.models.split(",") if m.strip()],
        [float(v) for v in args.top_k_grid.split(",") if v.strip()],
        args.horizon,
        args.config_path,
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
