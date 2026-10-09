"""E6: basket width (``top_k``) and position cap (``max_positions``).

Pre-registration: ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md`` §3 E6.
Two alleged defects in the construction layer:

1. ``TOP_K_GRID = (0.005, ..., 0.10)`` while ZZ1000 selected **0.10, the grid's
   upper bound** -- so the optimum might lie outside the grid;
2. ``max_positions`` has never been in any grid -- fixed at 20 everywhere.

**The trap the pre-registration names**: raising the position count is not alpha.
So the primary criterion is the config's percentile against a **like-for-like
random null**, never its absolute net return, and never its excess over an index.

Two structural facts, measured before the grid was run, reshape the experiment.
The first is the trap that makes this experiment non-obvious; the second is a
measurement artefact that has to be documented because it contaminates a gate.

* **The axes are coupled, but not redundant.** The engine holds at most
  ``max_positions`` names, filled from the ``select_top_k`` basket in descending
  ``rank_score``, so for a fixed cap every ``top_k`` whose basket reaches the cap
  has the **same entry candidate pool** and the same effective width
  ``min(ceil(n * top_k), max_positions)``.  On HS300's test window the
  cross-section is ~268 names, so at ``max_positions=20`` the cells
  ``top_k = 0.10 .. 0.30`` all have width 20; on ZZ1000 (~643 names) even
  ``0.05`` does.  That looks like it makes most of the extended grid redundant.
  **It does not**, and the first draft of this module asserted that it did until
  the consistency check below falsified it: the *exit* rule also reads basket
  membership (``hold_trade = eod_candidate.shift(1)``, so a held name is kept
  while it stays in the basket), and a wider basket retains held names longer,
  which occupies slots and blocks new entries.  Measured: ``tk0.10_mp20`` and
  ``tk0.30_mp20`` are width-identical on HS300's test window and return
  +50.68% / +55.71% on 1421 / 1135 trades.  So ``top_k`` stays live even where
  the cap binds -- through *holding period*, not through basket breadth.  The
  experiment therefore reports the width-vector grouping **and** how many
  grouped cells actually diverge, instead of assuming the grouping.
* **The cap is coupled to exposure.** ``target_notional = initial_capital /
  max_positions``, so raising the cap above the basket width does not hold more
  names -- it holds the same names at a *smaller* size.  And ``turnover_multiple``
  (``trade_value / initial_capital``) scales with that notional, so the turnover
  gate is partly reading exposure, not churn.  The pre-registration says to
  document the coupling; it turns out to matter, so the gate outcome is reported
  both at the calibrated floor and at a zero floor, with ``filled_trade_count``
  (exposure-free) alongside ``turnover_multiple`` (exposure-scaled).

Pre-registered criteria: (1) primary -- the new config's like-for-like random
percentile is ``>=`` the incumbent's **and** ``>= 95%``; (2) the four gates with
the incumbent production config as baseline; (3) the turnover floor is calibrated
separately and its derivation written down; (4) if nothing clears, the rule
returns ``None`` and the answer is **do not change**.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.experiments.ranking_ablation_harness import (
    backtest_arm,
    fold_excess_table,
    load_null_total_returns,
)
from research.experiments.random_portfolio_baseline import (
    _score_slots,
    run_random_portfolio_baseline,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import (
    fold_boundaries,
    phase_excess_table,
    select_configuration,
)
from research.market_regime.tradability_ranking_evaluation import (
    build_rank_market_data,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)


HORIZON = 10
#: Monthly validation sub-windows for the fold gate (validation spans ~30 months).
FOLD_STEP_MONTHS = 1
#: The ISO-weekday family the phase gate is paired over: 0=Mon .. 4=Fri.
WEEKDAYS = (0, 1, 2, 3, 4)

#: The published ``TOP_K_GRID`` plus the pre-registered extension.  The first six
#: values are ``tradability_ranking_evaluation.TOP_K_GRID`` unchanged, so the
#: incumbent cell is on this grid and the four gates have a real baseline.
TOP_K_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30)
#: Never gridded before.  20 is the production value and the baseline's cap.
MAX_POSITIONS_GRID = (10, 20, 30, 40)
BASELINE_MAX_POSITIONS = 20

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome", "null_tag": "pit"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "null_tag": "zz1000_pit"},
}


def cell_name(top_k: float, max_positions: int) -> str:
    """A grid cell's label.  Also the ``arm`` column, so the gates key off it."""

    return "tk{:g}_mp{}".format(float(top_k), int(max_positions))


def date_sizes(scores: pd.DataFrame) -> pd.Series:
    """The cross-section size of each row's date -- the ``n`` in ``ceil(n*top_k)``.

    Computed once per split and threaded through: ``select_top_k`` calls exactly
    this groupby, and re-deriving it per grid cell would put the same 36
    groupbys over half a million rows in the path of the identity check.
    """

    return scores.groupby("date", sort=False)[SCORE_COLUMN].transform("size")


def effective_widths(
    scores: pd.DataFrame, top_k: float, max_positions: int,
    counts: Optional[pd.Series] = None,
) -> pd.Series:
    """Per-date ``min(ceil(n * top_k), max_positions)`` -- the names actually held.

    This is the quantity the two grid axes both feed into, and it is what makes
    most of the grid redundant: the engine fills at most ``max_positions`` slots
    from the basket, so the *names* depend only on this number.
    """

    if not 0 < top_k <= 1:
        raise ValueError("top_k must be in (0, 1]")
    if counts is None:
        counts = date_sizes(scores)
    basket = np.ceil(counts.astype(float) * float(top_k))
    return pd.Series(
        np.minimum(basket, float(max_positions)), index=scores.index, name="width")


def width_groups(
    scores: pd.DataFrame,
    top_k_grid: Sequence[float],
    max_positions_grid: Sequence[int],
) -> Dict[Tuple[float, int], List[Tuple[float, int]]]:
    """Grid cells that hold the same names, every date, *at the same size*.

    The key is the ``effective_widths`` vector **and** the cap, because the cap is
    ``target_notional = initial_capital / max_positions``: two cells with the same
    width vector but different caps hold the same names at different sizes, which
    is an exposure difference and not a redundant configuration.  Grouping them
    would also make the print below claim the grid is smaller than it is.

    Cells in a group are **not** thereby the same strategy, and this function does
    not claim they are: the exit rule reads basket membership as well, so a wider
    basket holds on to its names longer (see the module docstring).  The grouping
    is a hypothesis about redundancy that :func:`evaluate_pool` then tests against
    the engine's own returns rather than assuming.
    """

    counts = date_sizes(scores)
    keys = {}
    for top_k in top_k_grid:
        for cap in max_positions_grid:
            cell = (float(top_k), int(cap))
            keys[cell] = (tuple(
                effective_widths(scores, top_k, cap, counts=counts).tolist()), cell[1])
    groups: Dict[tuple, List[Tuple[float, int]]] = {}
    for cell, key in keys.items():
        groups.setdefault(key, []).append(cell)
    return {cell: sorted(groups[key]) for cell, key in keys.items()}


def calibrate_turnover_floor(
    grid: pd.DataFrame,
    baseline_name: str,
    candidates: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50),
) -> Dict[str, object]:
    """Pick the floor the way ``probability_trade_evaluation`` picked 0.05.

    That precedent's justification is a *method*, not a number: two thresholds
    are equivalent when they exclude the same set, and when they are, you take
    the **smaller** -- "a measured degeneracy cut, not a tuned one", i.e. do not
    keep a restriction that is not doing any work.  This does the same thing
    mechanically: group the candidate floors by the set they exclude and take the
    smallest floor of the largest group (the plateau).

    **The returns are not consulted.**  Only the turnover ratio enters, so this
    cannot be steered towards a configuration that happens to look good.
    """

    if baseline_name not in set(grid["arm"]):
        raise ValueError("baseline_name={!r} is not on the grid".format(baseline_name))
    baseline_turnover = float(
        grid.loc[grid["arm"] == baseline_name, "validation_turnover_multiple"].iloc[0])
    if not baseline_turnover > 0:
        raise ValueError("baseline turnover is not positive; a ratio is meaningless")
    ratios = grid["validation_turnover_multiple"].astype(float) / baseline_turnover
    excluded = {}
    for floor in candidates:
        excluded[float(floor)] = tuple(sorted(
            grid.loc[ratios < float(floor), "arm"].astype(str)))
    plateau: Dict[tuple, List[float]] = {}
    for floor, names in excluded.items():
        plateau.setdefault(names, []).append(floor)
    widest = max(plateau.values(), key=len)
    widest_key = [key for key, floors in plateau.items() if floors is widest][0]
    return {
        "floor": float(min(widest)),
        "plateau": sorted(float(item) for item in widest),
        "excluded_at_floor": list(widest_key),
        "excluded_per_floor": {str(floor): list(names) for floor, names in excluded.items()},
        "baseline_turnover_multiple": baseline_turnover,
        "turnover_ratio": {str(arm): float(value) for arm, value in zip(grid["arm"], ratios)},
    }


def width_mechanism(
    scores: pd.DataFrame, top_k_grid: Sequence[float], max_positions_grid: Sequence[int]
) -> pd.DataFrame:
    """The side product: what each axis actually controls, per pool and split."""

    counts = date_sizes(scores)
    rows = []
    for top_k in top_k_grid:
        basket = np.ceil(counts.astype(float) * float(top_k))
        for cap in max_positions_grid:
            widths = effective_widths(scores, top_k, cap, counts=counts)
            rows.append({
                "arm": cell_name(top_k, cap),
                "top_k": float(top_k),
                "max_positions": int(cap),
                "median_basket": float(np.median(basket)),
                "median_effective_width": float(widths.median()),
                "cap_binding_share": float((basket > cap).mean()),
                "dates": int(scores["date"].nunique()),
            })
    return pd.DataFrame(rows)


def _aggregate_phases(validation_raw: pd.DataFrame) -> pd.DataFrame:
    """Weekday-average the numeric columns, keeping the phase extremes."""

    averaged = [
        column for column in validation_raw.columns
        if column not in ("arm", "weekday")
        and pd.api.types.is_numeric_dtype(validation_raw[column])
    ]
    grid = validation_raw.groupby("arm", as_index=False)[averaged].mean()
    extremes = validation_raw.groupby("arm")["total_return"].agg(["min", "max"]).rename(
        columns={"min": "phase_min", "max": "phase_max"}).reset_index()
    # Merged on ``arm`` rather than assigned positionally: the groupby result does
    # not share the frame's RangeIndex, and assigning it puts every value at NaN.
    return grid.merge(extremes, on="arm", how="left")


def evaluate_pool(
    pool: str,
    dataset_dir: Path,
    output_dir: Path,
    horizon: int = HORIZON,
    initial_capital: float = 500000.0,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    price_history_dirs: Optional[Sequence[str]] = None,
    weekdays: Sequence[int] = WEEKDAYS,
    seeds: int = 200,
    seed: int = 20260915,
    turnover_floor: Optional[float] = None,
    draw_nulls: bool = True,
) -> Dict[str, object]:
    spec = POOLS[pool]
    model_id = spec["model_id"]
    baseline_top_k = published_top_k(dataset_dir, model_id)
    baseline_arm = cell_name(baseline_top_k, BASELINE_MAX_POSITIONS)
    print("\n{}  model={}  baseline top_k={} (published) max_positions={}".format(
        pool, model_id, baseline_top_k, BASELINE_MAX_POSITIONS))

    # The published scores, read through the same loader the random-null runner
    # uses -- a private loader here could drop a different set of rows and the
    # row-count leg of the same-cross-section check would then fail for a reason
    # that has nothing to do with the null.
    scores = {
        split: load_outcome_scores(dataset_dir, model_id, split, horizon)
        for split in ("validation", "test")
    }
    for split, frame in scores.items():
        print("  {}: {} rows, {} dates, {} symbols".format(
            split, len(frame), frame["date"].nunique(), frame["symbol"].nunique()))

    # --- side product: which axis is binding, per pool -----------------------
    mechanism = {}
    for split, frame in scores.items():
        table = width_mechanism(frame, TOP_K_GRID, MAX_POSITIONS_GRID)
        table.insert(0, "split", split)
        mechanism[split] = table
    mechanism = pd.concat(mechanism.values(), ignore_index=True)
    print("\n  effective width min(ceil(n*top_k), max_positions), {} split:".format("validation"))
    for _, row in mechanism[mechanism["split"] == "validation"].iterrows():
        if row["top_k"] in (0.03, 0.05, 0.10, 0.30):
            print("    {:>13}  basket={:>6.0f}  width={:>5.0f}  cap binds {:>5.0%}".format(
                row["arm"], row["median_basket"], row["median_effective_width"],
                row["cap_binding_share"]))

    # --- which cells the width algebra says are redundant --------------------
    # Per split, never shared: the cross-section sizes differ (HS300 is ~217
    # names on validation and ~268 on test), so a cap that collapses the whole
    # top_k range in one window does not necessarily collapse it in the other.
    groups = {
        split: width_groups(frame, TOP_K_GRID, MAX_POSITIONS_GRID)
        for split, frame in scores.items()
    }
    for split in scores:
        families = {tuple(item) for item in groups[split].values()}
        print("  {}: {} cells -> {} distinct constructions ({} groups hold >1 cell)".format(
            split, len(groups[split]), len(families),
            sum(1 for family in families if len(family) > 1)))

    # --- the grid ------------------------------------------------------------
    raw, slots = {}, {}
    for split in ("validation", "test"):
        raw[split], skipped = build_rank_market_data(
            scores[split], price_history_dir, SCORE_COLUMN,
            price_history_dirs=price_history_dirs)
        if not raw[split]:
            raise ValueError("missing price data for {} {}".format(pool, split))
        slots[split] = _score_slots(scores[split], raw[split])
        print("  {} skipped prices: {}".format(split, skipped if skipped else "none"))

    cells = [(float(top_k), int(cap))
             for cap in MAX_POSITIONS_GRID for top_k in TOP_K_GRID]
    print("\n  running {} cells x {} weekdays on validation ({} runs)...".format(
        len(cells), len(weekdays), len(cells) * len(weekdays)))
    started = time.time()
    validation_rows, weekday_equity = [], {}
    for index, (top_k, cap) in enumerate(cells):
        name = cell_name(top_k, cap)
        for weekday in weekdays:
            run = backtest_arm(
                scores["validation"], raw["validation"], slots["validation"],
                SCORE_COLUMN, scores["validation"]["date"].min(),
                scores["validation"]["date"].max(), top_k, horizon, initial_capital,
                cap, config_path, weekday=weekday, label=name)
            weekday_equity[(name, weekday)] = run["equity"]
            validation_rows.append({
                "weekday": int(weekday), "arm": name,
                "top_k": top_k, "max_positions": cap,
                **{key: value for key, value in run["performance"].items()
                   if key != "strategy"},
            })
        print("    [{:>2}/{}] {:<13} {:.0f}s".format(
            index + 1, len(cells), name, time.time() - started), flush=True)
    validation_raw = pd.DataFrame(validation_rows)
    validation = _aggregate_phases(validation_raw)

    print("\n  running {} test cells (production ISO week)...".format(len(cells)))
    test_rows = []
    for top_k, cap in cells:
        name = cell_name(top_k, cap)
        run = backtest_arm(
            scores["test"], raw["test"], slots["test"], SCORE_COLUMN,
            scores["test"]["date"].min(), scores["test"]["date"].max(),
            top_k, horizon, initial_capital, cap, config_path, label=name)
        test_rows.append({
            "arm": name, "top_k": top_k, "max_positions": cap,
            **{key: value for key, value in run["performance"].items()
               if key != "strategy"},
        })
    test_metrics = pd.DataFrame(test_rows)

    # Does same-width actually mean same portfolio?  Each split is tested against
    # *its own* grouping -- validation's widths say nothing about test's -- and
    # the answer is reported rather than asserted, because the first draft of
    # this module assumed "yes" and the engine said "no".
    redundancy = []
    for split, frame in (("test", test_metrics), ("validation", validation)):
        check = frame.set_index("arm")["total_return"].astype(float)
        for family in sorted({tuple(item) for item in groups[split].values()}):
            members = [cell_name(*cell) for cell in family]
            if len(members) < 2:
                continue
            values = check[members]
            redundancy.append({
                "split": split,
                "group": "+".join(members),
                "cells": len(members),
                "min_total_return": float(values.min()),
                "max_total_return": float(values.max()),
                "spread": float(values.max() - values.min()),
                "identical": bool(values.round(12).nunique() == 1),
            })
    redundancy = pd.DataFrame(redundancy)
    diverged = redundancy[~redundancy["identical"]]
    print("\n  same-width groups: {} of {} still diverge in outcome (largest spread "
          "{:.2%}) -- the exit rule reads basket membership".format(
              len(diverged), len(redundancy), float(redundancy["spread"].max())))
    if len(diverged):
        for _, row in diverged.sort_values("spread", ascending=False).head(6).iterrows():
            print("    {:<10} {} cells  {:+.2%} .. {:+.2%}  (spread {:.2%})".format(
                row["split"], int(row["cells"]), row["min_total_return"],
                row["max_total_return"], row["spread"]))

    # --- gates ---------------------------------------------------------------
    weekday_mean_equity = {
        name: pd.concat(
            [weekday_equity[(name, weekday)] for weekday in weekdays], axis=1
        ).mean(axis=1).sort_index()
        for top_k, cap in cells for name in [cell_name(top_k, cap)]
    }
    folds = fold_boundaries(
        scores["validation"]["date"].min(), scores["validation"]["date"].max(),
        FOLD_STEP_MONTHS)
    fold_excess = fold_excess_table(
        weekday_mean_equity, folds, initial_capital, baseline_arm=baseline_arm)
    phase_excess = phase_excess_table(
        validation_raw, validation_raw[validation_raw["arm"] == baseline_arm],
        key_columns=("arm",))

    grid = validation.loc[:, [
        "arm", "top_k", "max_positions", "total_return", "turnover_multiple",
        "filled_trade_count", "max_drawdown"]].rename(columns={
            "total_return": "validation_total_return",
            "turnover_multiple": "validation_turnover_multiple"})

    calibration = calibrate_turnover_floor(grid, baseline_arm)
    floor = float(calibration["floor"] if turnover_floor is None else turnover_floor)
    print("\n  turnover ratio vs baseline (measured; returns not consulted):")
    for arm, ratio in sorted(calibration["turnover_ratio"].items(), key=lambda kv: kv[1]):
        traded = int(grid.loc[grid["arm"] == arm, "filled_trade_count"].iloc[0])
        print("    {:<13} ratio={:>6.3f}  trades={:>4}".format(arm, ratio, traded))
    print("  calibrated floor {:.2f} (plateau {} exclude the same set)".format(
        calibration["floor"], calibration["plateau"]))
    if turnover_floor is not None:
        print("  floor OVERRIDDEN to {:.2f} by the caller".format(floor))

    def select(floor_value: float):
        return select_configuration(
            grid,
            baseline_key={"arm": baseline_arm},
            key_columns=("arm",),
            return_column="validation_total_return",
            turnover_column="validation_turnover_multiple",
            fold_excess=fold_excess,
            phase_excess=phase_excess,
            turnover_floor_frac=floor_value,
        )

    outcome = select(floor)
    free = select(0.0)
    print("\n  selection rule (floor={:.2f}): {}\n    {}".format(
        floor, "NO CHANGE" if outcome.row is None else outcome.row["arm"], outcome.reason))
    print("  same, floor=0.00 (shows whether the turnover gate decided it): {}\n    {}".format(
        "NO CHANGE" if free.row is None else free.row["arm"], free.reason))

    forced_through = (
        free.row is not None
        and (outcome.row is None or str(free.row["arm"]) != str(outcome.row["arm"])))
    print("  turnover gate is decision-relevant (floor=0 would change the winner): {}".format(
        forced_through))

    # --- primary criterion: like-for-like random percentile -------------------
    published = load_null_total_returns(
        dataset_dir, model_id, spec["null_tag"], baseline_top_k,
        len(scores["test"]), max_positions=BASELINE_MAX_POSITIONS)
    baseline_return = float(
        test_metrics.loc[test_metrics["arm"] == baseline_arm, "total_return"].iloc[0])
    baseline_percentile = float((published <= baseline_return).mean())
    print("\n  baseline {} test={:+.2%}  percentile {:.1%} vs {} published draws (mean {:+.2%})".format(
        baseline_arm, baseline_return, baseline_percentile, len(published),
        published.mean()))

    chosen = None if outcome.row is None else str(outcome.row["arm"])
    null_result = None
    criterion_1 = False
    if chosen is None:
        print("  no candidate clears the gates -> the rule says DO NOT CHANGE; "
              "no new null is drawn (pre-registration: nulls only for survivors)")
    elif not draw_nulls:
        print("  --no-nulls: skipping the primary criterion")
    else:
        top_k, cap = (float(outcome.row["top_k"]), int(outcome.row["max_positions"]))
        tag = "e6_tk{:g}_mp{}".format(top_k, cap)
        print("\n  drawing {} random draws for the selected cell {} (top_k={}, max_positions={})...".format(
            seeds, chosen, top_k, cap))
        summary = run_random_portfolio_baseline(
            model_id, dataset_dir, output_dir, seeds=seeds, seed=seed, top_k=top_k,
            horizon=horizon, initial_capital=initial_capital, max_positions=cap,
            price_history_dir=price_history_dir, config_path=config_path,
            tag=tag, split="test")
        # The null runner re-runs the *real* arm itself.  If that disagrees with
        # this module's own run of the same cell, one of the two chains is not
        # the production one and the percentile would be comparing windows.
        runner_actual = float(summary["actual"]["total_return"])
        if abs(runner_actual - float(
                test_metrics.loc[test_metrics["arm"] == chosen, "total_return"].iloc[0])) > 1e-9:
            raise ValueError(
                "null runner's actual ({:+.6f}) differs from this module's run of the "
                "same cell".format(runner_actual))
        drawn = load_null_total_returns(
            dataset_dir, model_id, tag, top_k, len(scores["test"]), max_positions=cap)
        percentile = float((drawn <= runner_actual).mean())
        null_result = {
            "arm": chosen, "top_k": top_k, "max_positions": cap, "tag": tag,
            "draws": int(len(drawn)), "null_mean": float(drawn.mean()),
            "actual_total_return": runner_actual, "percentile": percentile,
            "baseline_percentile": baseline_percentile,
        }
        criterion_1 = percentile >= baseline_percentile and percentile >= 0.95
        print("  selected cell: percentile {:.1%} vs baseline {:.1%}; null mean {:+.2%}".format(
            percentile, baseline_percentile, drawn.mean()))
        print("  criterion 1 (percentile >= baseline AND >= 95%): {}".format(
            "PASS" if criterion_1 else "FAIL"))

    criteria = {
        "criterion_1_null_percentile": criterion_1,
        "criterion_2_four_gates": outcome.row is not None,
        "criterion_4_no_change_when_nothing_clears": outcome.row is None,
    }
    if outcome.row is None:
        verdict = "no change (no candidate cleared the gates)"
    elif criterion_1:
        verdict = "adopt {}".format(chosen)
    else:
        verdict = "undecidable: {} cleared the gates but failed the primary criterion".format(chosen)
    print("\n  VERDICT: {}".format(verdict))

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "basket_width_{}".format(pool)
    validation_raw.to_csv(output_dir / (prefix + "_validation_phases.csv"), index=False)
    test_metrics.to_csv(output_dir / (prefix + "_test_metrics.csv"), index=False)
    mechanism.to_csv(output_dir / (prefix + "_width_mechanism.csv"), index=False)
    redundancy.to_csv(output_dir / (prefix + "_same_width_groups.csv"), index=False)
    fold_columns = ["fold_{}".format(index) for index in range(len(folds))]
    grid.merge(phase_excess[["arm", "phase_excess_min", "phases"]], on="arm", how="left").merge(
        fold_excess[["arm"] + fold_columns], on="arm", how="left").to_csv(
        output_dir / (prefix + "_validation_grid.csv"), index=False)

    report = {
        "pool": pool,
        "model_id": model_id,
        "baseline_arm": baseline_arm,
        "baseline_top_k": float(baseline_top_k),
        "baseline_max_positions": BASELINE_MAX_POSITIONS,
        "top_k_grid": [float(item) for item in TOP_K_GRID],
        "max_positions_grid": [int(item) for item in MAX_POSITIONS_GRID],
        "width_groups": {
            split: {
                "cells": len(groups[split]),
                "distinct_constructions": len({tuple(v) for v in groups[split].values()}),
                "multi_cell_groups": sorted({
                    "+".join(cell_name(*cell) for cell in family)
                    for family in {tuple(v) for v in groups[split].values()}
                    if len(family) > 1}),
            }
            for split in ("validation", "test")
        },
        "same_width_groups": {
            "groups": int(len(redundancy)),
            "diverged": int(len(diverged)),
            "largest_spread": float(redundancy["spread"].max()) if len(redundancy) else 0.0,
        },
        "turnover_calibration": calibration,
        "turnover_floor_used": floor,
        "turnover_gate_is_decision_relevant": bool(forced_through),
        "selection_rule": {
            "row": chosen,
            "reason": outcome.reason,
            "gates": dict(outcome.gates),
            "reason_without_turnover_gate": free.reason,
            "row_without_turnover_gate": None if free.row is None else str(free.row["arm"]),
        },
        "baseline_test_total_return": baseline_return,
        "baseline_percentile": baseline_percentile,
        "null": null_result,
        "criteria": criteria,
        "verdict": verdict,
        "execution": {
            "horizon": int(horizon),
            "initial_capital": float(initial_capital),
            "weekdays": [int(day) for day in weekdays],
            "test_schedule": "production ISO week (weekday=None)",
        },
    }
    (output_dir / (prefix + "_manifest.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), default=None, help="Default: both pools.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--weekdays", default=",".join(str(day) for day in WEEKDAYS),
                        help="ISO weekdays for the phase family; one value runs one schedule.")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--turnover-floor", type=float, default=None,
                        help="Override the calibrated floor (the calibration is still reported).")
    parser.add_argument("--no-nulls", action="store_true",
                        help="Skip drawing the survivor's random null (the primary criterion).")
    args = parser.parse_args(argv)
    weekdays = tuple(int(item) for item in args.weekdays.split(",") if item.strip())
    pools = [args.pool] if args.pool else sorted(POOLS)
    for pool in pools:
        evaluate_pool(
            pool, Path(args.dataset_dir), Path(args.output_dir),
            horizon=args.horizon, initial_capital=args.initial_capital,
            price_history_dir=args.price_history_dir, config_path=args.config,
            weekdays=weekdays, seeds=args.seeds, seed=args.seed,
            turnover_floor=args.turnover_floor, draw_nulls=not args.no_nulls)


if __name__ == "__main__":
    main()
