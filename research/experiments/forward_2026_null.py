"""The like-for-like null for the 2026 forward window, per pool.

The read-out rule pre-registered for this window is the strategy's **percentile
inside its own same-window random distribution**, so the null has to be rebuilt
on 2026 rather than carried over from the published window -- a different
universe, a different volatility regime and a different length all move it.

Reusing ``random_portfolio_baseline`` verbatim is what makes this like-for-like:
identical cross-section, identical price frames, identical execution chain,
only ``entry_score`` permuted.  Two things are supplied from outside:

* ``price_history_dirs`` -- the merged historical + 2026 price trees;
* ``top_k`` -- read from each arm's published ranking manifest, never typed in.
  The plan assumed 0.1 for both pools; the manifests say hs300 froze 0.05.  A
  null computed at the wrong basket width would be a different experiment, and
  the discrepancy would be invisible.

Example::

    python -m research.experiments.forward_2026_null --pool hs300 --seeds 200
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from research.experiments.forward_2026_backtest import (
    ARMS,
    FORWARD_ROOT,
    POOLS,
    published_execution,
)
from research.experiments.random_portfolio_baseline import run_random_portfolio_baseline
from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR
from research.market_regime.output_paths import DATASET_DIR

HORIZON = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), default="hs300")
    parser.add_argument("--arm", choices=sorted(ARMS), default="frozen")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--forward-root", default=str(FORWARD_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    forward_root = Path(args.forward_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    staging = forward_root / "dataset_2026"

    model_id = POOLS[args.pool]["model_id"]
    arm_model_id = model_id + ARMS[args.arm]
    execution = published_execution(dataset_dir, model_id)

    if not (staging / "trade_outcome_test_predictions_{}.csv".format(arm_model_id)).exists():
        raise SystemExit(
            "{} not staged; run forward_2026_backtest first".format(staging / arm_model_id))

    print("{} arm={}  top_k={} (from {})\n".format(
        arm_model_id, args.arm, execution["top_k"], Path(execution["source"]).name))

    summary = run_random_portfolio_baseline(
        arm_model_id,
        staging,
        output_dir,
        seeds=args.seeds,
        seed=args.seed,
        top_k=execution["top_k"],
        horizon=HORIZON,
        initial_capital=execution["initial_capital"],
        max_positions=execution["max_positions"],
        config_path=args.config_path,
        min_holding_days=execution["min_holding_days"],
        tag="forward2026_{}_{}".format(args.pool, args.arm),
        split="test",
        price_history_dirs=[str(DEFAULT_PRICE_HISTORY_DIR), str(forward_root / "price_history")],
    )

    payload = {
        "pool": args.pool,
        "arm": args.arm,
        "model_id": arm_model_id,
        "top_k_source": execution["source"],
        "top_k": execution["top_k"],
        "min_holding_days": execution["min_holding_days"],
        "initial_capital": execution["initial_capital"],
        "max_positions": execution["max_positions"],
        "seeds": args.seeds,
        "seed": args.seed,
        "rows": summary["split_rows"],
        "symbols": summary["split_symbols"],
        "price_symbols": summary["price_symbols"],
        "price_history_dirs": summary["price_history_dirs"],
        "random": summary["random"],
        "actual": summary["actual"],
    }
    (forward_root / "null_manifest_forward2026_{}_{}.json".format(args.pool, args.arm)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
