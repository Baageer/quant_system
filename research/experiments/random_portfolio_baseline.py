"""Is the Top-K backtest any better than picking 20 names at random?

The 中证1000 rerun put ``zz1000_excess_v1_outcome`` at +27.71% net against a
pool whose equal-weight buy-and-hold is +30.78% -- i.e. indistinguishable from
"buy the whole pool".  But the backtest holds **20** names while the pool is
**~950**, so that comparison is not like-for-like either.  Adding features until
the strategy clears +35% would leave the same question open: real alpha, or just
more beta from a pool that already drifts up?

This script closes the gap by holding everything fixed except the score.  It
takes the *identical* test cross-section, the *identical* price frames and the
*identical* execution chain (``select_top_k`` -> ``apply_rank_selection`` ->
``make_rank_strategy`` -> engine), and replaces ``entry_score`` with a uniform
random permutation.  Each draw therefore picks a random ``top_k`` slice of the
day's candidates and a random 20 of those -- same candidate count, same weekly
rebalance, same 10-day minimum hold, same costs.  Doing that N times gives the
distribution of a *score-free* portfolio on that pool.

The strategy's percentile inside that distribution is the answer to "does the
model add anything at 20 names?" -- and unlike the pool comparison it is
directly interpretable.

**The HS300 arm is a method control, not a second result.**  HS300's strategy
must land far out in the tail of its own random distribution; if it does not,
the procedure has no power and the 中证1000 number cannot be read at all.

Implementation note: ``make_rank_strategy`` breaks ties among same-day entry
candidates by ``rank_score``, so permuting ``entry_score`` is not enough -- the
``rank_score`` column inside the price frames (joined once by
``build_rank_market_data`` from the *real* scores) has to be rewritten too.
``_score_slots`` precomputes where each symbol's score rows sit inside its price
frame so that rewrite is an array assignment rather than a re-join.

Example::

    python -m research.experiments.random_portfolio_baseline --arm zz1000 --seeds 200
    python -m research.experiments.random_portfolio_baseline --arm zz1000 --split validation --tag zz1000_pit_val

``--model-id`` is only needed to point at a retired arm (the static-pool ids are
still in the dataset directory and still runnable); it overrides ``--arm``.

``--top-k`` defaults to the width the model was **published** at, read from its
ranking manifest, rather than to a fixed 0.1.  The two sides of the comparison
are only like-for-like at the same basket width, and the widths are not the same
across pools: measured 2026-09-18, ``hs300_pit_outcome`` published 0.05 against
``zz1000_pit_outcome``'s 0.10.  A hard-coded default therefore silently widened
the HS300 arm, and the mismatch would have read as alpha or as decay rather than
as an error.  Deriving it makes like-for-like structural instead of contingent
on remembering to pass the flag.

.. note::
    On ``--split validation`` the *actual* arm's published number is a max over
    the Top-K grid (that is how ``trade_outcome_ranking_evaluation`` picks
    Top-K), while each random draw is a single fixed width -- i.e. the comparison
    is tilted **in the model's favour**.  Measured 2026-09-16: for
    ``hs300_excess_v1_outcome``, ``zz1000_excess_v1_outcome`` and
    ``zz1000_pit_outcome`` the validation argmax is the ``top_k=0.1`` row, so at
    their published widths the two sides are like-for-like and nothing has to be
    discounted.  Re-check this if the model or pool changes.

    The single-window percentile is *not* the whole story: the zz1000 static arm
    scores 0/200 on validation and 41.5% on test, while the PIT arm is in the
    tail on both.  Compare signs across splits, and see
    ``validation_null_phase_check.py`` for the phase caveat.

    The published summaries carry a ``top_k_source`` field saying whether the
    width came from the manifest or from ``--top-k``; check it before quoting a
    number, since a null drawn at an explicit width is only comparable to a
    published arm at that same width.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)


#: Pool -> the model that pool is *currently* published under.  These were the
#: retired static-pool ids (``*_excess_v1_outcome``) long after the PIT pools
#: replaced them, so the default arm drew a null distribution for a model nobody
#: was quoting.  Keep this in step with ``forward_2026_backtest.POOLS``; the two
#: are duplicate copies of the same fact, which is how the drift happened.
ARMS = {
    "hs300": "hs300_pit_outcome",
    "zz1000": "zz1000_pit_outcome",
    "zz500": "zz500_pit_outcome",
}

REPORT_METRICS = (
    "total_return",
    "annual_return",
    "sharpe_zero_rate",
    "max_drawdown",
    "turnover_multiple",
    "filled_buy_count",
)


def _score_slots(scores: pd.DataFrame, raw_market_data: Dict[str, pd.DataFrame]):
    """Locate every score row inside its symbol's price frame.

    Returns ``symbol -> (frame_positions, score_positions)``: the two aligned
    arrays needed to scatter any permutation of ``scores[SCORE_COLUMN]`` back
    into ``rank_score`` without re-joining.
    """

    slots = {}
    by_symbol = {symbol: group for symbol, group in scores.groupby("symbol", sort=False)}
    for symbol, frame in raw_market_data.items():
        group = by_symbol.get(symbol)
        if group is None:
            continue
        frame_positions = frame.index.get_indexer(pd.DatetimeIndex(group["date"]))
        if (frame_positions < 0).any():
            raise ValueError("Score dates outside the price frame for {}".format(symbol))
        slots[symbol] = (frame_positions, group.index.to_numpy())
    return slots


def _write_rank_score(prepared: Dict[str, pd.DataFrame], values: np.ndarray, slots) -> None:
    """Overwrite ``rank_score`` from a permuted score vector, matching the join."""

    for symbol, frame in prepared.items():
        column = np.zeros(len(frame), dtype=float)
        if symbol in slots:
            frame_positions, score_positions = slots[symbol]
            column[frame_positions] = values[score_positions]
        frame["rank_score"] = column


def carry_forward_marks(price_of, all_dates) -> pd.DataFrame:
    """Valuation marks: last valid close carried forward through missing bars.

    ``load_local_price_history`` drops rows whose close is NaN, so a halted name
    has *no row* on those days rather than a NaN row.  Reindexing to the union
    calendar and forward-filling restores the mark.  The ledgers in this package
    must use this for their ``stock_value`` leg: dropping the name instead
    deletes that position from the book for the day, which is a one-day phantom
    loss the size of the name's weight.  ``backtest/engine.py`` marks the same
    way (see ``_build_optimized_data_structures``).
    """

    return pd.DataFrame(price_of).reindex(pd.DatetimeIndex(all_dates)).ffill()


def _run_one(prepared, context: Dict[str, object]) -> Dict[str, object]:
    """Backtest one selection through the production execution chain."""

    results, trades = _run_engine(
        prepared,
        make_rank_strategy(
            context["max_positions"], context["target_notional"], min_holding_days=context["horizon"]
        ),
        context["initial_capital"],
        context["config_path"],
        context["start_date"],
        context["end_date"],
    )
    return summarize_backtest("random", results, trades, context["initial_capital"])


def run_random_portfolio_baseline(
    model_id: str,
    dataset_dir: Path,
    output_dir: Path,
    seeds: int = 200,
    seed: int = 20260915,
    top_k: Optional[float] = None,
    horizon: int = 10,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    min_holding_days: Optional[int] = None,
    tag: str = "",
    split: str = "test",
    verbose: bool = True,
    price_history_dirs: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Score-free distribution for one arm, plus where the real strategy lands.

    ``top_k`` defaults to the width the model was *published* at, read from its
    ranking manifest -- the two sides are only like-for-like at the same width,
    and a narrower or wider random draw would land as apparent alpha or apparent
    decay rather than as an error.  Pass it only to deviate deliberately.
    """

    explicit_top_k = top_k
    top_k = published_top_k(dataset_dir, model_id) if top_k is None else float(top_k)
    scores = load_outcome_scores(dataset_dir, model_id, split, horizon)
    raw_market_data, skipped = build_rank_market_data(
        scores, price_history_dir, SCORE_COLUMN, price_history_dirs=price_history_dirs)
    if not raw_market_data:
        raise ValueError("No {} price data for {}".format(split, model_id))
    slots = _score_slots(scores, raw_market_data)
    context: Dict[str, object] = {
        "max_positions": max_positions,
        "target_notional": initial_capital / max_positions,
        "horizon": horizon if min_holding_days is None else min_holding_days,
        "initial_capital": initial_capital,
        "config_path": config_path,
        "start_date": scores["date"].min(),
        "end_date": scores["date"].max(),
    }

    actual_selected = select_top_k(scores, top_k, SCORE_COLUMN)
    actual = _run_one(apply_rank_selection(raw_market_data, scores, actual_selected), context)

    values = np.asarray(scores[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)
    probe = scores.copy()
    rows: List[Dict[str, object]] = []
    started = time.time()
    for draw in range(seeds):
        shuffled = rng.permutation(values)
        probe[SCORE_COLUMN] = shuffled
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, shuffled, slots)
        performance = _run_one(prepared, context)
        performance["draw"] = draw
        performance["selected_count"] = int(np.asarray(selected, dtype=bool).sum())
        if verbose and (draw % 10 == 0 or draw == seeds - 1):
            elapsed = time.time() - started
            print(
                "  draw {:>4}/{}  return={:>8.2%}  selected={}  [{:.0f}s]".format(
                    draw + 1, seeds, performance["total_return"], performance["selected_count"], elapsed
                ),
                flush=True,
            )
        rows.append(performance)

    draws = pd.DataFrame(rows)
    summary: Dict[str, object] = {
        "model_id": model_id,
        "seeds": int(seeds),
        "seed": int(seed),
        "split": split,
        "top_k": float(top_k),
        "top_k_source": (
            "argument" if explicit_top_k is not None
            else "published manifest for {}".format(model_id)
        ),
        "horizon": int(horizon),
        "min_holding_days": int(context["horizon"]),
        "max_positions": int(max_positions),
        "split_rows": int(len(scores)),
        "split_symbols": int(scores["symbol"].nunique()),
        "price_symbols": int(len(raw_market_data)),
        "price_history_dirs": list(price_history_dirs) if price_history_dirs else [price_history_dir],
        "skipped_prices": skipped,
        "actual": {key: actual[key] for key in REPORT_METRICS},
        "actual_selected_count": int(np.asarray(actual_selected, dtype=bool).sum()),
        "random": {},
    }
    for metric in REPORT_METRICS:
        drawn = draws[metric].astype(float)
        actual_value = float(actual[metric])
        summary["random"][metric] = {
            "mean": float(drawn.mean()),
            "std": float(drawn.std(ddof=1)) if len(drawn) > 1 else 0.0,
            "min": float(drawn.min()),
            "max": float(drawn.max()),
            "median": float(drawn.median()),
            "percentile_of_actual": float((drawn <= actual_value).mean()),
            "empirical_p_value": float((1 + int((drawn >= actual_value).sum())) / (1 + len(drawn))),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "random_portfolio_{}{}".format(model_id, "_{}".format(tag) if tag else "")
    draws.to_csv(output_dir / (prefix + "_draws.csv"), index=False)
    (output_dir / (prefix + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _print_report(model_id, summary)
    return summary


def _print_report(model_id: str, summary: Dict[str, object]) -> None:
    print("\n{}  ({} random draws of {} names, top_k={})".format(
        model_id, summary["seeds"], summary["max_positions"], summary["top_k"]))
    print("  {:<22} {:>10} {:>12} {:>12} {:>10} {:>8}".format(
        "metric", "actual", "random mean", "random sd", "pctile", "p-val"))
    for metric, stats in summary["random"].items():
        print("  {:<22} {:>10.4f} {:>12.4f} {:>12.4f} {:>9.1%} {:>8.4f}".format(
            metric,
            float(summary["actual"][metric]),
            stats["mean"],
            stats["std"],
            stats["percentile_of_actual"],
            stats["empirical_p_value"],
        ))


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="zz1000")
    parser.add_argument("--model-id", default=None, help="Override the arm's model id.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--top-k", type=float, default=None,
                        help="Defaults to the width --model-id was published at, read from its "
                             "ranking manifest, so both sides of the comparison hold the same one.")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--min-holding-days", type=int, default=None,
                        help="Defaults to --horizon. Set very high to hold the draw as a static basket.")
    parser.add_argument("--tag", default="", help="Suffix for the output file names.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test",
                        help="Split to draw from. validation makes the comparison optimistic "
                             "for the model (see the module docstring).")
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--price-history-dirs", default=None,
                        help="Comma-separated roots, concatenated at load time (forward windows).")
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    dirs = [d.strip() for d in args.price_history_dirs.split(",") if d.strip()] \
        if args.price_history_dirs else None
    run_random_portfolio_baseline(
        args.model_id or ARMS[args.arm],
        Path(args.dataset_dir),
        Path(args.output_dir),
        seeds=args.seeds,
        seed=args.seed,
        top_k=args.top_k,
        horizon=args.horizon,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        price_history_dir=args.price_history_dir,
        config_path=args.config_path,
        min_holding_days=args.min_holding_days,
        tag=args.tag,
        split=args.split,
        price_history_dirs=dirs,
    )


if __name__ == "__main__":
    main()
