"""E2: which head carries the ranking -- expected excess, expected downside, or both?

The published score is ``entry_score = pred_excess - 0.25 * max(-pred_mae, 0)``.
Measured rank IC says the two heads are not peers: ``mae_model`` ranks the forward
outcome about an order of magnitude better than ``excess_model`` does, and stably
across validation and test (HS300 0.20/0.23 vs 0.009/0.039; ZZ1000 0.21/0.24 vs
0.068/0.087).  So the question is not idle -- the combine may be doing the work,
or one head may be carrying the other.

No retraining and no re-inference: the published prediction files already carry
the *raw* head outputs (``predicted_excess_return_10d`` / ``predicted_mae_10d``),
so the three arms are three ways of reading columns that are already on disk.
``A_full`` is taken from the disk ``entry_score`` verbatim rather than
recombined, so the incumbent arm is bit-identical to the published one and any
difference between arms is the arm.

**Read the verdict in engine net return, never in
``selected_excess_over_universe``.**  The latter is "basket realised excess minus
pool realised excess" and therefore favours the excess head *by construction* --
judging on it would kill ``A_mae`` by definition, which is the thing under test.

**The arms must be signed the way the heads enter the score, and the downside
head is the trap.**  ``entry_score`` subtracts ``max(-pred_mae, 0)``, so the
downside head *raises* the score the *shallower* the predicted drawdown; an arm
built as ``-predicted_mae`` sorts the predicted-worst names to the top and is an
inverted head, not an ablated one.  That mistake has been made once here: it
produced a ``A_mae`` leg whose basket had deeper realised downside than the
universe it was drawn from (HS300 -5.65% vs -3.82%; ZZ1000 -6.77% vs -4.57%) and
a ZZ1000 test net return of -68.8%, all of which was the sign and none of which
was the head.  ``test_arm_orientation_matches_the_score_convention`` now pins the
orientation against the published formula rather than against a literal.

Two cost decisions, both stated rather than assumed:

* **Basket width is frozen at each pool's published value** and never re-picked
  per arm.  Re-selecting the width per arm would confound "changed the score"
  with "changed the construction parameter".
* **The random null is shared across the three arms.**  ``rng.permutation``
  shuffles *positions*, so with a fixed seed the recovered index permutation is
  identical whatever the values are (verified, not assumed); a permutation
  destroys the score's content, so the null depends only on
  ``(universe, top_k, seed)`` and is the same for every arm.  That makes the
  already-published 200-draw nulls reusable instead of 1200 fresh engine runs.

Pre-registered criteria: ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md``
§3 E2.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import _score_slots
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import (
    fold_boundaries,
    select_configuration,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_model import target_columns
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    published_top_k,
)
# The engine plumbing lives in a shared module now that E1 needs the same chain.
# Re-exported under the names this module has always used, so the tests that pin
# them keep pinning the same callables.
from research.experiments.ranking_ablation_harness import (  # noqa: F401
    backtest_arm,
    basket_downside,
    fold_excess_table,
    load_null_total_returns,
)


ARMS = ("A_full", "A_excess", "A_mae")
BASELINE_ARM = "A_full"
HORIZON = 10
#: Monthly validation sub-windows for the fold gate.  Validation spans ~30
#: months, so this yields ~30 folds.
FOLD_STEP_MONTHS = 1

#: ``null_tag`` names the published 200-draw test-split null for each pool.  The
#: tags are not uniform across pools (``pit`` vs ``zz1000_pit``) and each pool
#: also has same-width nulls for *other* windows, so the tag is only a pointer:
#: :func:`load_null_total_returns` re-verifies the width and the row count
#: against the cross-section being evaluated, which is what makes the null
#: actually the same universe.  HS300's predates the A1 halted-name valuation
#: fix, which moved maxDD/Sharpe but left returns untouched -- only
#: ``total_return`` is read from it here.
POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome", "null_tag": "pit"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "null_tag": "zz1000_pit"},
}


def load_ablation_scores(
    dataset_dir: Path, model_id: str, split: str, horizon: int = HORIZON
) -> pd.DataFrame:
    """One column per arm, plus the labels, from the published prediction file.

    Rows are dropped if **any** arm is unusable, so the three arms are scored on
    an identical row set -- otherwise the per-symbol slot map used to rewrite
    ``rank_score`` would be right for one arm and off by a row for another.
    """

    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    excess_column, mae_column = target_columns(horizon)
    predicted_excess = "predicted_excess_return_{}d".format(horizon)
    predicted_mae = "predicted_mae_{}d".format(horizon)
    required = {
        "date", "symbol", SCORE_COLUMN, excess_column, mae_column,
        predicted_excess, predicted_mae, "risk_penalty",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} missing columns: {}".format(path.name, sorted(missing)))

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    for column in (SCORE_COLUMN, excess_column, mae_column, predicted_excess, predicted_mae):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    penalties = frame["risk_penalty"].dropna().unique()
    if len(penalties) != 1:
        raise ValueError("expected one risk_penalty, found {}".format(list(penalties)))
    risk_penalty = float(penalties[0])

    # ``A_full`` is the disk column, not a recombination of the other two.  The
    # raw columns round-trip through CSV, so recomputing would differ by ~1e-16
    # and the incumbent arm would stop being bit-identical to the published one.
    frame["A_full"] = frame[SCORE_COLUMN]
    frame["A_excess"] = frame[predicted_excess]
    # ``A_mae`` is the downside head's *contribution to the published score*, up
    # to the positive factor ``risk_penalty``, and it is signed the way that
    # contribution is.  Reading it as ``-predicted_mae`` looks natural ("MAE is a
    # loss, negate it") and is backwards: ``predicted_mae_10d`` is stored as a
    # return and is negative for essentially every row, and the score *subtracts*
    # ``max(-pred_mae, 0)``, so the head makes the score *higher* the *shallower*
    # the predicted drawdown.  Ordering by ``-predicted_mae`` therefore sorts the
    # predicted-worst-downside names to the top of the basket.  ``-max(-pred, 0)``
    # is the contribution term itself (``max`` pins the rows with a positive
    # predicted MAE to the flat 0 the score gives them).
    frame["A_mae"] = -np.maximum(-frame[predicted_mae], 0.0)

    frame = frame.dropna(subset=list(ARMS) + [excess_column, mae_column])
    frame.attrs["risk_penalty"] = risk_penalty
    frame.attrs["model_id"] = model_id
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def evaluate_pool(
    pool: str,
    dataset_dir: Path,
    output_dir: Path,
    horizon: int = HORIZON,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    price_history_dirs: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Three arms, two splits, one width, one shared null."""

    model_id = POOLS[pool]["model_id"]
    top_k = published_top_k(dataset_dir, model_id)
    print("\n{}  model={}  top_k={} (published)".format(pool, model_id, top_k))

    validation = load_ablation_scores(dataset_dir, model_id, "validation", horizon)
    test = load_ablation_scores(dataset_dir, model_id, "test", horizon)
    risk_penalty = float(test.attrs["risk_penalty"])
    print("  risk_penalty={}  validation rows={}  test rows={}".format(
        risk_penalty, len(validation), len(test)))

    # Resolved before any engine run: a missing or mismatched null is a setup
    # error, and discovering it after twelve backtests wastes the whole run.
    null_returns = load_null_total_returns(
        dataset_dir, model_id, POOLS[pool]["null_tag"], top_k, len(test))
    print("  null: {} draws, mean {:+.2%} (verified top_k and row count)".format(
        len(null_returns), null_returns.mean()))

    validation_raw, validation_skipped = build_rank_market_data(
        validation, price_history_dir, SCORE_COLUMN, price_history_dirs=price_history_dirs)
    test_raw, test_skipped = build_rank_market_data(
        test, price_history_dir, SCORE_COLUMN, price_history_dirs=price_history_dirs)
    if not validation_raw or not test_raw:
        raise ValueError("missing price data for {}".format(pool))
    validation_slots = _score_slots(validation, validation_raw)
    test_slots = _score_slots(test, test_raw)

    def run(frame, raw, slots):
        return {
            arm: backtest_arm(
                frame, raw, slots, arm,
                frame["date"].min(), frame["date"].max(),
                top_k, horizon, initial_capital, max_positions, config_path,
            )
            for arm in ARMS
        }

    print("  running validation arms...")
    validation_runs = run(validation, validation_raw, validation_slots)
    print("  running test arms...")
    test_runs = run(test, test_raw, test_slots)

    rows: List[Dict[str, object]] = []
    for split, runs in (("validation", validation_runs), ("test", test_runs)):
        for arm in ARMS:
            performance = runs[arm]["performance"]
            rows.append({
                "split": split,
                "arm": arm,
                **{"{}_{}".format(split, key): value
                   for key, value in performance.items() if key != "strategy"},
                **{"{}_{}".format(split, key): value
                   for key, value in basket_downside(
                       validation if split == "validation" else test,
                       runs[arm]["selected"], horizon).items()},
            })
    metrics = pd.DataFrame(rows)

    def cell(split, arm, column):
        return float(metrics.loc[(metrics["split"] == split) & (metrics["arm"] == arm), column].iloc[0])

    print("\n  net total return by arm:")
    for arm in ARMS:
        print("    {:<10} validation={:>8.2%}  test={:>8.2%}".format(
            arm, cell("validation", arm, "validation_total_return"),
            cell("test", arm, "test_total_return")))

    folds = fold_boundaries(
        validation["date"].min(), validation["date"].max(), FOLD_STEP_MONTHS)
    fold_excess = fold_excess_table(
        {arm: validation_runs[arm]["equity"] for arm in ARMS}, folds, initial_capital,
        baseline_arm=BASELINE_ARM)
    grid = pd.DataFrame([
        {
            "arm": arm,
            "validation_total_return": cell("validation", arm, "validation_total_return"),
            # Validation turnover against validation turnover: the gate compares
            # an arm's validation costliness with the baseline's on the same
            # window.  Reading the test column here would compare two windows.
            "validation_turnover_multiple": cell(
                "validation", arm, "validation_turnover_multiple"),
        }
        for arm in ARMS
    ])
    outcome = select_configuration(
        grid,
        baseline_key={"arm": BASELINE_ARM},
        key_columns=("arm",),
        return_column="validation_total_return",
        turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess,
    )
    print("\n  selection rule: {}\n    {}".format(
        "NO CHANGE" if outcome.row is None else outcome.row["arm"], outcome.reason))

    percentiles = {
        arm: float((null_returns <= cell("test", arm, "test_total_return")).mean())
        for arm in ARMS
    }
    print("  test percentile vs the shared {}-draw null (mean {:+.2%}):".format(
        len(null_returns), null_returns.mean()))
    for arm in ARMS:
        print("    {:<10} {:>7.1%}".format(arm, percentiles[arm]))

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "two_head_ablation_{}".format(pool)
    metrics.to_csv(output_dir / (prefix + "_metrics.csv"), index=False)
    # Merge on ``arm``, never by position: ``fold_excess_table`` returns arms in
    # sorted order while the grid is built in ARMS order, so assigning the fold
    # columns positionally would silently pair each arm with another one's folds.
    fold_columns = ["fold_{}".format(index) for index in range(len(folds))]
    grid.merge(fold_excess[["arm"] + fold_columns], on="arm", how="left").to_csv(
        output_dir / (prefix + "_validation_grid.csv"), index=False)
    manifest = {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "top_k_source": "published manifest (frozen across arms)",
        "risk_penalty": risk_penalty,
        "baseline_arm": BASELINE_ARM,
        "selection_rule": {
            "row": None if outcome.row is None else str(outcome.row["arm"]),
            "reason": outcome.reason,
            "gates": dict(outcome.gates),
        },
        "test_percentile_vs_shared_null": percentiles,
        "null_draws": int(len(null_returns)),
        "null_mean_total_return": float(null_returns.mean()),
        "execution": {
            "horizon": int(horizon),
            "max_positions": max_positions,
            "initial_capital": initial_capital,
            "validation_skipped_prices": validation_skipped,
            "test_skipped_prices": test_skipped,
        },
    }
    (output_dir / (prefix + "_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), default=None,
                        help="Default: both pools.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--price-history-dirs", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    dirs = [d.strip() for d in args.price_history_dirs.split(",") if d.strip()] \
        if args.price_history_dirs else None
    pools = [args.pool] if args.pool else sorted(POOLS)
    for pool in pools:
        evaluate_pool(
            pool, Path(args.dataset_dir), Path(args.output_dir),
            horizon=args.horizon, initial_capital=args.initial_capital,
            max_positions=args.max_positions, price_history_dir=args.price_history_dir,
            config_path=args.config_path, price_history_dirs=dirs,
        )


if __name__ == "__main__":
    main()
