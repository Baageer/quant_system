"""How much of v1's published number came from the static (survivorship) universe?

Re-running the whole chain on the PIT-filtered dataset changes two things at once:
the model is refit on a different training population, *and* the test-window
candidate pool changes.  Only the second is the direct look-ahead in the reported
backtest, so the two are separated here by holding the model fixed:

* ``v1_static``  -- published v1: static pool everywhere            (control)
* ``v1_pit``     -- v1's *scores*, candidates restricted to real members
* ``pit_pit``    -- full PIT pipeline, refit on the PIT population    (control)

Then::

    test-window look-ahead  = v1_pit  - v1_static
    everything else         = pit_pit - v1_pit      (refit + hyperparameters)

The fourth cell -- the PIT model scored on the static universe -- is not
constructible without re-predicting, since the model's prediction file only
contains rows that were members.

Example::

    python -m research.market_regime.pit_universe_decomposition
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from research.market_regime.index_membership import membership_mask
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
DEFAULT_ARMS = "v1_static:v1:static,v1_pit:v1:pit,pit_pit:pit:pit"
MODEL_IDS = {"v1": "hs300_excess_v1_outcome", "pit": "hs300_pit_outcome"}


def backtest_arm(
    model_id: str,
    apply_pit: bool,
    pool: str,
    dataset_dir: Path,
    top_k: float,
    horizon: int,
    config_path: str,
    membership_dir: Optional[Path] = None,
) -> Dict[str, float]:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    rows_before = len(test)
    if apply_pit:
        test = test[membership_mask(test, pool, membership_dir).to_numpy()].reset_index(drop=True)
        if test.empty:
            raise ValueError("PIT filter removed every test row for {}".format(model_id))

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
        "pit": bool(apply_pit),
        "top_k": top_k,
        "candidate_rows": rows_before,
        "candidates_kept": len(test),
        "skipped_symbols": len(skipped),
        **stats,
    }


def run(
    arms: Sequence[str],
    dataset_dir: Path,
    top_k_grid: Sequence[float],
    horizon: int = 10,
    pool: str = "hs300",
    config_path: str = "config/settings.yaml",
    membership_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    rows = []
    for spec in arms:
        label, model_key, universe = spec.split(":")
        for top_k in top_k_grid:
            row = backtest_arm(MODEL_IDS[model_key], universe == "pit", pool, dataset_dir,
                               top_k, horizon, config_path, membership_dir)
            row["arm"] = label
            rows.append(row)

    frame = pd.DataFrame(rows)
    columns = ["arm", "model_id", "pit", "top_k", "candidates_kept", "total_return", "annual_return",
               "annual_volatility", "sharpe_zero_rate", "max_drawdown", "turnover_multiple",
               "filled_trade_count"]
    frame = frame[[c for c in columns if c in frame.columns]]

    print("\n=== test-split backtest, identical construction, universe switched on/off ===")
    display = frame.copy()
    for column in ("total_return", "annual_return", "annual_volatility", "max_drawdown"):
        display[column] = display[column].map(lambda v: "{:>8.2%}".format(v))
    for column in ("sharpe_zero_rate", "turnover_multiple"):
        display[column] = display[column].map(lambda v: "{:>8.4f}".format(v))
    print(display.to_string(index=False))

    print("\n--- decomposition of the published -> PIT change ---")
    for top_k in top_k_grid:
        sub = frame[frame["top_k"] == top_k].set_index("arm")
        if not {"v1_static", "v1_pit", "pit_pit"}.issubset(sub.index):
            continue
        base = sub.loc["v1_static", "total_return"]
        universe = sub.loc["v1_pit", "total_return"] - base
        refit = sub.loc["pit_pit", "total_return"] - sub.loc["v1_pit", "total_return"]
        print("    top_k {:<6}  published {:>8.2%} | test-window universe {:>+8.2%} | refit+hyperparams {:>+8.2%} | total {:>+8.2%}".format(
            top_k, base, universe, refit, sub.loc["pit_pit", "total_return"] - base))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "pit_universe_decomposition.csv", index=False, encoding="utf-8-sig")
    return frame


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=DEFAULT_ARMS)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--top-k-grid", default="0.05,0.10")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--membership-dir", default=None)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    args = parser.parse_args(argv)
    run(
        [a.strip() for a in args.arms.split(",") if a.strip()],
        Path(args.dataset_dir),
        [float(v) for v in args.top_k_grid.split(",") if v.strip()],
        args.horizon,
        args.pool,
        args.config_path,
        Path(args.membership_dir) if args.membership_dir else None,
        Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
