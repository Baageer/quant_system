"""E9 step 1: push the re-represented scores through the **unmodified** production chain.

Step 0 (``cross_sectional_representation_screen``) refit the two-head Ridge on a
per-date re-representation of the existing features and read the resulting
``entry_score`` on validation.  It closed the axis outright for 中证1000 (all four
variants failed, two of them got *worse*) and advanced three candidates for
HS300.  This module is the expensive half: backtest the survivors with the
production execution chain and judge them with the same four gates E6 used.

**What is deliberately unchanged.**  ``top_k`` is the published value
(``published_top_k``), the cap is the production 20, the strategy is
``make_rank_strategy`` struck on the ISO week, the engine is
``probability_trade_evaluation._run_engine``, and ``alpha`` / ``risk_penalty`` /
the feature list all come from the model manifest.  The representation is the
only thing that moves -- otherwise step 0's screen would not be screening the
thing this module tests.

**Two things this module is not allowed to do**, both written into the
pre-registration:

* it may not re-tune the variant on the test window -- validation selects, test
  is read once;
* it may not treat a validation win as a result.  E1 (GBDT: wins validation,
  loses test by ~17pp in both pools) and E6 (same effective width, 23.96pp apart)
  are both exactly this failure mode, and E9 has one of its own on the record:
  the two pools disagreed at step 0.

**The fold gate is the reason step 1 runs only after P0-4.**  ``select_configuration``'s
fourth gate asks for every monthly fold to be won, which no arm in this repo has
ever cleared (E6: 36/36 cells failed, best 16/29).  ``--min-fold-agreement``
therefore carries the calibrated theta; the module also reports the raw fold win
count so the answer can be re-read at any threshold without re-running.

Example::

    python -m research.experiments.cross_sectional_representation_backtest --pools hs300
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.cross_sectional_representation_screen import (
    fit_arm,
    load_model_settings,
    score_arm,
)
from research.experiments.ranking_ablation_harness import (
    backtest_arm,
    fold_excess_table,
    load_null_total_returns,
)
from research.experiments.random_portfolio_baseline import _score_slots
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import (
    fold_boundaries,
    phase_excess_table,
    select_configuration,
)
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_model import load_outcome_split
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)


HORIZON = 10
FOLD_STEP_MONTHS = 1
WEEKDAYS = (0, 1, 2, 3, 4)
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
CONFIGURED_MIN_FOLD_AGREEMENT = 1.0
#: E6 calibrated 0.05 as the measured degeneracy cut for the turnover gate; E9
#: keeps the same construction layer, so it inherits the same floor.
CONFIGURED_TURNOVER_FLOOR = 0.05
#: E1's bar, restated here so the number in the report and the number in the code
#: cannot drift apart.
MIN_EXCESS_IMPROVEMENT = 0.02
MIN_PERCENTILE = 0.95

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome", "null_tag": "pit"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "null_tag": "zz1000_pit"},
}


def attach_variant_scores(
    stored: pd.DataFrame,
    panel: pd.DataFrame,
    arms: Mapping[str, Mapping[str, object]],
    risk_penalty: float,
) -> pd.DataFrame:
    """Add one score column per variant, after proving the rows line up.

    The published predictions file and this module's own panel are two different
    reads of the same window.  If they disagree on a single row, every column
    added here would be a shifted copy of the real one and the backtest would be
    silently scoring the wrong names -- so the alignment is asserted, not assumed.
    """

    frame = stored.copy()
    if len(panel) != len(frame):
        raise ValueError("panel rows {} != stored rows {}".format(len(panel), len(frame)))
    same_date = (pd.to_datetime(panel["date"]).to_numpy() == pd.to_datetime(frame["date"]).to_numpy()).all()
    same_symbol = (np.asarray(panel["symbol"]) == np.asarray(frame["symbol"])).all()
    if not (same_date and same_symbol):
        raise ValueError("panel rows are not in the same order as the stored predictions")
    for name, arm in arms.items():
        frame[name] = score_arm(arm, panel, risk_penalty)["entry_score"]
    # The raw column must be the published score, or the baseline every gate
    # differences against is not the incumbent.
    drift = float(np.abs(np.asarray(frame["raw"], dtype=float)
                         - np.asarray(frame[SCORE_COLUMN], dtype=float)).max())
    if drift > 1e-9:
        raise ValueError("raw arm does not reproduce the published {}: max drift {:g}".format(SCORE_COLUMN, drift))
    return frame


def phase_mean_equity(equity_by_weekday: Mapping[int, pd.Series]) -> pd.Series:
    """Average the ISO-weekday family into one equity curve (E6's convention)."""

    if not equity_by_weekday:
        raise ValueError("no weekday equity to average")
    return pd.concat(list(equity_by_weekday.values()), axis=1).mean(axis=1).sort_index()


def run_variants(
    pool: str,
    variants: Sequence[str],
    dataset_dir: Path,
    output_dir: Path,
    min_fold_agreement: float = CONFIGURED_MIN_FOLD_AGREEMENT,
    turnover_floor: Optional[float] = None,
    price_history_dir: Optional[str] = None,
    price_history_dirs: Optional[Sequence[str]] = None,
    config_path: str = "config/settings.yaml",
    weekdays: Sequence[int] = WEEKDAYS,
) -> Dict[str, object]:
    spec = POOLS[pool]
    model_id = spec["model_id"]
    settings = load_model_settings(dataset_dir, model_id)
    features, alpha, risk_penalty = settings["feature_columns"], settings["alpha"], settings["risk_penalty"]
    dataset_id = settings["dataset_id"]
    top_k = published_top_k(dataset_dir, model_id)
    print("\n{}  model={}  features={}  alpha={:g}  risk_penalty={:g}  top_k={} (published)  cap={}".format(
        pool, model_id, len(features), alpha, risk_penalty, top_k, MAX_POSITIONS))

    train = load_outcome_split(dataset_dir, dataset_id, "train", features, HORIZON)
    panels = {"validation": load_outcome_split(dataset_dir, dataset_id, "validation", features, HORIZON),
              "test": load_outcome_split(dataset_dir, dataset_id, "test", features, HORIZON)}
    arms = {name: fit_arm(train, features, name, alpha) for name in ("raw", *variants)}

    joined, raw_market_data, slots = {}, {}, {}
    for split in ("validation", "test"):
        stored = load_outcome_scores(dataset_dir, model_id, split, HORIZON)
        joined[split] = attach_variant_scores(stored, panels[split], arms, risk_penalty)
        raw, skipped = build_rank_market_data(joined[split], price_history_dir, SCORE_COLUMN,
                                              price_history_dirs=price_history_dirs)
        if not raw:
            raise ValueError("missing price data for {} {}".format(pool, split))
        raw_market_data[split] = raw
        slots[split] = _score_slots(joined[split], raw)
        print("  {}: {} rows, {} dates, {} symbols  (skipped prices: {})".format(
            split, len(joined[split]), joined[split]["date"].nunique(),
            joined[split]["symbol"].nunique(), skipped if skipped else "none"))

    names = ["raw", *variants]
    print("\n  running {} arms x {} weekdays on validation ({} runs)...".format(
        len(names), len(weekdays), len(names) * len(weekdays)))
    started = time.time()
    rows, equity = [], {}
    for name in names:
        for weekday in weekdays:
            run = backtest_arm(
                joined["validation"], raw_market_data["validation"], slots["validation"], name,
                joined["validation"]["date"].min(), joined["validation"]["date"].max(),
                top_k, HORIZON, INITIAL_CAPITAL, MAX_POSITIONS, config_path,
                weekday=weekday, label=name)
            equity[(name, weekday)] = run["equity"]
            rows.append({"weekday": int(weekday), "arm": name,
                         **{key: value for key, value in run["performance"].items() if key != "strategy"}})
        print("    {:<10} {:.0f}s".format(name, time.time() - started), flush=True)
    validation_raw = pd.DataFrame(rows)
    validation = validation_raw.groupby("arm", as_index=False).mean(numeric_only=True)

    print("\n  running {} test arms (production ISO week)...".format(len(names)))
    test_rows = []
    for name in names:
        run = backtest_arm(
            joined["test"], raw_market_data["test"], slots["test"], name,
            joined["test"]["date"].min(), joined["test"]["date"].max(),
            top_k, HORIZON, INITIAL_CAPITAL, MAX_POSITIONS, config_path, label=name)
        test_rows.append({"arm": name,
                          **{key: value for key, value in run["performance"].items() if key != "strategy"}})
    test_metrics = pd.DataFrame(test_rows).set_index("arm")

    # --- gates ---------------------------------------------------------------
    folds = fold_boundaries(joined["validation"]["date"].min(), joined["validation"]["date"].max(),
                            FOLD_STEP_MONTHS)
    mean_equity = {name: phase_mean_equity({weekday: equity[(name, weekday)] for weekday in weekdays})
                   for name in names}
    fold_excess = fold_excess_table(mean_equity, folds, INITIAL_CAPITAL, baseline_arm="raw")
    phase_excess = phase_excess_table(
        validation_raw, validation_raw[validation_raw["arm"] == "raw"], key_columns=("arm",))

    grid = validation.loc[:, ["arm", "total_return", "turnover_multiple", "filled_trade_count"]].rename(
        columns={"total_return": "validation_total_return", "turnover_multiple": "validation_turnover_multiple"})
    fold_wins = fold_win_counts(fold_excess)
    print("\n  fold agreement ({} monthly folds; gate threshold {:g}):".format(len(folds), min_fold_agreement))
    for name in names:
        print("    {:<10} {}/{} folds won  ({:.1%})".format(name, fold_wins[name], len(folds),
                                                            fold_wins[name] / float(len(folds))))

    floor = CONFIGURED_TURNOVER_FLOOR if turnover_floor is None else float(turnover_floor)
    print("  turnover ratio vs baseline: {}".format(
        {row["arm"]: round(float(row["validation_turnover_multiple"])
                           / float(grid.loc[grid["arm"] == "raw", "validation_turnover_multiple"].iloc[0]), 3)
         for _, row in grid.iterrows()}))
    print("  turnover floor in force: {:.2f}".format(floor))

    outcome = select_configuration(
        grid, baseline_key={"arm": "raw"}, key_columns=("arm",),
        return_column="validation_total_return", turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess, phase_excess=phase_excess,
        turnover_floor_frac=floor, min_fold_agreement=float(min_fold_agreement))
    print("\n  selection rule: {}\n    {}".format(
        "NO CHANGE" if outcome.row is None else outcome.row["arm"], outcome.reason))

    # --- primary criterion: like-for-like random percentile -------------------
    published = load_null_total_returns(dataset_dir, model_id, spec["null_tag"], top_k,
                                        len(joined["test"]), max_positions=MAX_POSITIONS)
    baseline_return = float(test_metrics.loc["raw", "total_return"])
    baseline_percentile = float((published <= baseline_return).mean())
    print("\n  random null: {} published draws, mean {:+.2%}".format(len(published), published.mean()))
    print("  raw  test {:+.2%}  percentile {:.1%}".format(baseline_return, baseline_percentile))
    for name in names[1:]:
        value = float(test_metrics.loc[name, "total_return"])
        print("  {:<10} test {:+.2%}  percentile {:.1%}  vs raw {:+.2%}".format(
            name, value, float((published <= value).mean()), value - baseline_return))

    return {
        # The gate inputs are returned and written to disk, not just consumed:
        # the fold decision is a function of these and of P0-4's theta, so saving
        # them lets the answer be re-read at any threshold without paying for the
        # backtests again.
        "fold_excess": fold_excess, "phase_excess": phase_excess, "grid": grid,
        "pool": pool, "model_id": model_id, "top_k": top_k, "max_positions": MAX_POSITIONS,
        "horizon": HORIZON, "min_fold_agreement": float(min_fold_agreement), "turnover_floor": floor,
        "fold_count": len(folds), "fold_wins": fold_wins,
        "validation": validation, "test_metrics": test_metrics,
        "baseline_test_return": baseline_return, "baseline_percentile": baseline_percentile,
        "null_mean": float(published.mean()), "null_draws": int(len(published)),
        "selection": {"arm": None if outcome.row is None else str(outcome.row["arm"]),
                      "reason": outcome.reason},
        "criterion_improvement": {
            name: float(test_metrics.loc[name, "total_return"]) - baseline_return for name in names[1:]},
        "criterion_min_excess": MIN_EXCESS_IMPROVEMENT,
        "criterion_min_percentile": MIN_PERCENTILE,
        "test_percentiles": {name: float((published <= float(test_metrics.loc[name, "total_return"])).mean())
                             for name in names},
    }


def fold_columns(table: pd.DataFrame) -> list:
    """The fold columns, counted by **the same rule the gate counts them by**.

    ``select_configuration`` uses a plain ``startswith(fold_prefix)``; any
    stricter helper here would report a different denominator than the one the
    gate divided by, which is exactly the kind of two-definitions-of-one-number
    drift that makes a fold count unreadable.
    """

    return sorted(column for column in table.columns if column.startswith("fold_"))


def fold_win_counts(fold_excess: pd.DataFrame) -> Dict[str, int]:
    """Folds won, per arm, over exactly the columns the gate divides by."""

    columns = fold_columns(fold_excess)
    if not columns:
        raise ValueError("fold_excess has no fold_* columns to count")
    if "arm" not in fold_excess.columns:
        raise ValueError("fold_excess has no 'arm' column")
    return {str(arm): int((block[columns] > 0).sum(axis=1).iloc[0])
            for arm, block in fold_excess.groupby("arm", sort=False)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--pools", nargs="+", default=["hs300"], choices=sorted(POOLS))
    parser.add_argument("--variants", nargs="+", default=["demean", "zscore"])
    parser.add_argument("--min-fold-agreement", type=float, default=CONFIGURED_MIN_FOLD_AGREEMENT,
                        help="P0-4's calibrated theta; the configured value is 1.0")
    parser.add_argument("--turnover-floor", type=float, default=None)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    for pool in args.pools:
        result = run_variants(pool, args.variants, Path(args.dataset_dir), output_dir,
                              min_fold_agreement=args.min_fold_agreement,
                              turnover_floor=args.turnover_floor,
                              price_history_dir=args.price_history_dir,
                              config_path=args.config_path)
        result["validation"].to_csv(output_dir / "cross_sectional_backtest_{}_validation.csv".format(pool),
                                    index=False)
        result["test_metrics"].reset_index().to_csv(
            output_dir / "cross_sectional_backtest_{}_test.csv".format(pool), index=False)
        result["fold_excess"].to_csv(output_dir / "cross_sectional_backtest_{}_fold_excess.csv".format(pool),
                                     index=False)
        result["phase_excess"].to_csv(output_dir / "cross_sectional_backtest_{}_phase_excess.csv".format(pool),
                                      index=False)
        result["grid"].to_csv(output_dir / "cross_sectional_backtest_{}_gate_grid.csv".format(pool),
                              index=False)
        payload = {key: value for key, value in result.items()
                   if key not in ("validation", "test_metrics", "fold_excess", "phase_excess", "grid")}
        (output_dir / "cross_sectional_backtest_{}_summary.json".format(pool)).write_text(
            json.dumps(payload, indent=2, default=str))
        print("\n  written: {}".format(output_dir / "cross_sectional_backtest_{}_summary.json".format(pool)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
