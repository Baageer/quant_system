"""E5b: cap the basket's exposure to any one industry, and see whether it pays.

E5's first action (step 0) did **not** close the line: every cell had a maximum
daily industry weight at or above the pre-registered 20%.  But step 0's corrected
reading split the two pools, and this run acts on that split:

* **HS300: the tilt is real.**  Same-width median percentile 93.3% (validation) /
  98.5% (test), above the 90th in all three width bands and both windows.
* **ZZ1000: no stable tilt.**  Validation sits at 59.5% -- level with a random
  basket of the same size -- and test at 79.5%.  The windows disagree.

So E5b runs **HS300 only**.  ZZ1000 is recorded as **not motivated** rather than
run-and-ignored: a candidate whose whole premise is absent in one window cannot
be adjudicated by adding a second pool, and running it anyway would put a number
in the report that only invites being read as confirmation.

Pre-registration (written before any number below was produced)
---------------------------------------------------------------

* **Arm.**  The published HS300 scores, the published ``top_k``, the published
  engine -- everything identical to the incumbent except the basket.  The
  candidate is therefore exactly "the same model, forced to spread across
  industries".  Any difference is attributable to the cap and nothing else.
* **Cap rule, fixed here.**  ``MAX_INDUSTRY_SHARE = 0.30``.  For a date whose
  published basket holds ``n`` names, the quota is ``quota = max(floor(0.30 * n),
  1)``.  Walk the day's universe in descending score order and accept a name iff
  its industry currently holds fewer than ``quota`` names, stopping at ``n``
  accepted or the universe exhausted.  ``floor`` and not ``ceil``: with n = 13,
  ceil(3.9) = 4 would be 30.8% of the book, i.e. the cap would not bind at all
  where it was supposed to.  The ``max(..., 1)`` keeps a small basket from
  emptying an industry entirely, which is a different intervention than a cap.
* **Backfill, and why.**  Walking the *universe* rather than the selected set is
  the backfill: continuing down the ranked list past rank ``n`` is exactly
  "replace the excluded name with the next eligible one".  The basket is
  therefore held at its published **width**, on purpose.  E6 measured the
  construction axis at 56-71pp of test spread against ~17pp for the model axis,
  so letting the cap shrink the book would confound the industry effect with an
  exposure effect and no reading of the result could separate them.  When the
  universe cannot fill the basket, the shortfall is **counted and reported** --
  a cap that silently shrinks the book is not the arm described above.
* **Criteria** (from the plans doc, E5 criterion 1, operationalised):
  1. **four gates** via ``select_configuration`` against the uncapped arm, at the
     same settings E6 and E9 used (configured fold agreement, turnover floor
     0.05) so the three are comparable;
  2. **random percentile >= 95%** on test, against the *published* same-width
     null -- ``top_k``, row count and ``max_positions`` all match the capped arm,
     so the published draws are the right null and no new ones are drawn;
  3. **improvement >= +2pp** on test versus the uncapped arm.
* **Declared in advance: criterion 1 is expected to be unreachable.**  The fold
  gate's default is 1.0 and P0-4 measured that the configured value is satisfied
  by 0.0000% of the null, so "29 folds all won" is closed in code.  P0-4's
  calibrated theta (18/29) was deliberately **not** wired in -- criterion 4 of
  that pre-registration makes it a separate decision.  This run does not make
  it.  Criterion 1 is therefore reported with its gate counts and its failure is
  **not** a finding about the cap.
* **Sensitivity, firewalled.**  20% and 40% are run and reported for shape only.
  **Adoption reads 30% and nothing else**; the other two may not be adopted in
  this run under any outcome.  Without that firewall a three-point sweep is just
  a search for the setting that happens to win.
* **Must-report, or the arm may not be testing what it claims.**  The cap binds
  at *selection*, but the engine holds each name for at least ``HORIZON`` days
  and rebalances weekly, so the **held** book is a blend of many capped baskets
  and may still exceed 30% in one industry.  So the realised held concentration
  is measured with the step-0 machinery and reported for the capped arm beside
  the baseline.  **If the held book does not move, this arm is not a test of
  industry neutralisation and the report says so** rather than reading the
  equity curve as one.

What this module cannot say: it does not touch the labels, so it cannot speak to
E5a, and it inherits step 0's static-snapshot industry labels (the pre-registered
option (a)) -- a forward look in the *labelling*, reported as an approximation.

Example::

    python -m research.experiments.industry_cap_ab
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.cross_sectional_representation_backtest import (
    FOLD_STEP_MONTHS,
    WEEKDAYS,
    phase_mean_equity,
)
from research.experiments.industry_concentration import (
    INDUSTRY_SOURCE,
    industry_of,
    load_industry_map,
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
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)
from research.experiments.cross_sectional_representation_backtest import (
    fold_win_counts,
)

HORIZON = 10
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
#: The pre-registered primary.  Reported sensitivity lives in ``SENSITIVITY``.
MAX_INDUSTRY_SHARE = 0.30
#: Reported for shape only; adoption reads :data:`MAX_INDUSTRY_SHARE` alone.
SENSITIVITY = (0.20, 0.40)
#: E6 measured 0.05 as the degeneracy cut for the turnover gate; E5b sits on the
#: same construction layer as E6 and E9, so it inherits the floor.
CONFIGURED_TURNOVER_FLOOR = 0.05
CONFIGURED_MIN_FOLD_AGREEMENT = 1.0
MIN_EXCESS_IMPROVEMENT = 0.02
MIN_PERCENTILE = 0.95
#: The industry permutation null for the *held* book, from step 0.
PERMUTATIONS = 200
SEED = 20260920

POOL = "hs300"
MODEL_ID = "hs300_pit_outcome"
NULL_TAG = "pit"


def capped_selection(
    selected: pd.Series,
    scores: pd.DataFrame,
    mapping: Mapping[str, str],
    share: float,
    score_column: str = SCORE_COLUMN,
) -> tuple:
    """The pre-registered rule: greedy quota pass in score order, basket width held.

    Returns ``(capped, diagnostics)``.  The diagnostics are not decoration -- the
    rule can fail to fill a basket when a day's universe is too concentrated to
    supply ``n`` names under the quota, and an arm that quietly runs thinner than
    the incumbent is a different arm.  The shortfall is counted so the report can
    say whether that happened.
    """

    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    frame = scores.loc[:, ["date", "symbol", score_column]].copy()
    frame["_selected"] = np.asarray(selected, dtype=bool)
    # Rank order over the *whole* universe, not just the selected set: continuing
    # past rank n is the backfill, and it is what keeps the width constant.
    frame["_rank"] = frame.groupby("date", sort=False)[score_column].rank(
        method="first", ascending=False)

    rows, shortfalls, quotas, depths = [], {}, {}, {}
    for date, block in frame.groupby("date", sort=True):
        block = block.sort_values("_rank", kind="mergesort")
        target = int(block["_selected"].sum())
        quota = max(int(np.floor(share * target)), 1)
        quotas[date] = quota
        used: Dict[str, int] = {}
        chosen, reached = [], 0
        for position, symbol in enumerate(block["symbol"].to_numpy()):
            if len(chosen) >= target:
                break
            name = industry_of(symbol, mapping)
            if used.get(name, 0) >= quota:
                continue
            used[name] = used.get(name, 0) + 1
            chosen.append(symbol)
            reached = position + 1
        shortfall = target - len(chosen)
        if shortfall:
            shortfalls[date] = shortfall
        # How deep into the ranking the backfill had to reach, relative to the
        # published width.  1.0 means every kept name was already in the basket;
        # 4.0 means the day's top names were one industry and the cap filled the
        # book from four baskets deep.  Reported because a cap that reaches very
        # far down is no longer "the same model, spread out" -- and nothing else
        # in the output would show it.
        if target:
            depths[date] = reached / float(target)
        rows.append((block, set(chosen)))

    keep = np.zeros(len(frame), dtype=bool)
    position = {index: i for i, index in enumerate(frame.index)}
    for block, chosen in rows:
        for index, symbol in zip(block.index, block["symbol"].to_numpy()):
            if symbol in chosen:
                keep[position[index]] = True
    diagnostics = {
        "target_names": int(sum(int(b["_selected"].sum()) for b, _ in rows)),
        "kept_names": int(keep.sum()),
        "days_short_of_target": len(shortfalls),
        "worst_shortfall": int(max(shortfalls.values())) if shortfalls else 0,
        "quota_median": float(np.median(list(quotas.values()))),
        "quota_min": int(min(quotas.values())),
        "quota_max": int(max(quotas.values())),
        "backfill_depth_median": float(np.median(list(depths.values()))) if depths else 1.0,
        "backfill_depth_p95": float(np.percentile(list(depths.values()), 95)) if depths else 1.0,
        "backfill_depth_max": float(max(depths.values())) if depths else 1.0,
        "share": float(share),
    }
    return pd.Series(keep, index=frame.index), diagnostics


def make_filter(scores: pd.DataFrame, mapping: Mapping[str, str], share: float):
    """Close over the frame so ``backtest_arm`` sees a plain ``Series -> Series``."""

    state: Dict[str, object] = {}

    def filtered(selected: pd.Series) -> pd.Series:
        capped, diagnostics = capped_selection(selected, scores, mapping, share)
        state["diagnostics"] = diagnostics
        return capped

    filtered.state = state  # type: ignore[attr-defined]
    return filtered


def baseline_selection_filter(scores: pd.DataFrame, mapping: Mapping[str, str]):
    """The uncapped arm run through the same seam, so both arms touch one path.

    Deliberately not ``selection_filter=None``: routing the baseline through the
    identical code path means the seam itself cannot be the difference.  A no-op
    filter is not free here -- it re-orders the frame -- so if it changed the
    equity curve at all, that would be the finding.
    """

    state: Dict[str, object] = {}

    def passthrough(selected: pd.Series) -> pd.Series:
        state["diagnostics"] = {
            "share": 1.0,
            "target_names": int(np.asarray(selected, dtype=bool).sum()),
            "kept_names": int(np.asarray(selected, dtype=bool).sum()),
            "days_short_of_target": 0, "worst_shortfall": 0,
            "quota_median": float("nan"), "quota_min": 0, "quota_max": 0,
            "backfill_depth_median": 1.0, "backfill_depth_p95": 1.0, "backfill_depth_max": 1.0,
        }
        return selected

    passthrough.state = state  # type: ignore[attr-defined]
    return passthrough


def _performance(run_arm: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in run_arm["performance"].items() if key != "strategy"}


def held_concentration(results, market_data, mapping, top_k, scores, draws):
    """Step 0's machinery, aimed at the capped arm's realised book.

    Reusing the step-0 functions rather than re-deriving anything is the point:
    the width-matched null and the bracket were argued out there, and a second
    implementation here would be a second chance to get the width wrong -- which
    is the exact mistake step 0 had to repair twice.
    """

    from research.experiments.industry_concentration import (
        close_panel_from,
        concentration_bounds,
        concentration_metrics,
        held_percentile_vs_baskets,
        permuted_basket_weights,
        portfolio_industry_weights,
    )

    panel = close_panel_from(market_data)
    held, exposure, n_hold = portfolio_industry_weights(results, panel, mapping)
    metrics = concentration_metrics(held, n_hold)
    bounds = concentration_bounds(held)
    baskets = permuted_basket_weights(scores, mapping, top_k, held_counts=n_hold, draws=draws)
    percentile = held_percentile_vs_baskets(metrics, concentration_metrics(baskets))
    summary = {
        "held_max_industry_weight_median": float(metrics["max_industry_weight"].median()),
        "held_max_industry_weight_p95": float(metrics["max_industry_weight"].quantile(0.95)),
        "held_max_industry_weight_max": float(metrics["max_industry_weight"].max()),
        "held_max_industry_weight_lower": float(bounds["lower"].max()),
        "held_basket_percentile_median": float(percentile.median()),
        "held_share_days_above_p95": float((percentile > 0.95).mean()),
        "held_effective_industries_median": float(metrics["effective_industries"].median()),
        "held_n_hold_median": float(n_hold.median()),
        "held_exposure_mean": float(exposure.mean()),
    }
    return summary, metrics


def run(
    dataset_dir: Path,
    output_dir: Path,
    industry_path: Path,
    draws: int = PERMUTATIONS,
    config_path: str = "config/settings.yaml",
    weekdays: Sequence[int] = WEEKDAYS,
) -> Dict[str, object]:
    mapping = load_industry_map(industry_path)
    top_k = published_top_k(dataset_dir, MODEL_ID)
    print("{}  model={}  top_k={} (published)  cap={:.0%}  sensitivity={}".format(
        POOL, MODEL_ID, top_k, MAX_INDUSTRY_SHARE,
        ", ".join("{:.0%}".format(value) for value in SENSITIVITY)))

    stored, market, slots = {}, {}, {}
    for split in ("validation", "test"):
        stored[split] = load_outcome_scores(dataset_dir, MODEL_ID, split, HORIZON)
        raw, skipped = build_rank_market_data(stored[split], None, SCORE_COLUMN)
        if not raw:
            raise ValueError("missing price data for {} {}".format(POOL, split))
        market[split] = raw
        slots[split] = _score_slots(stored[split], raw)
        print("  {}: {} rows, {} dates, {} symbols (skipped: {})".format(
            split, len(stored[split]), stored[split]["date"].nunique(),
            stored[split]["symbol"].nunique(), skipped if skipped else "none"))

    # Arm names are the labels the tables carry.  Every arm shares one score
    # column because E5b does not touch the scores at all -- only the basket.
    arms = {"uncapped": None, "cap30": MAX_INDUSTRY_SHARE}
    for share in SENSITIVITY:
        arms["cap{:02d}".format(int(share * 100))] = share
    filters = {
        name: (baseline_selection_filter(stored["validation"], mapping) if share is None
               else make_filter(stored["validation"], mapping, share))
        for name, share in arms.items()
    }

    print("\n  validation: {} arms x {} weekdays = {} runs".format(
        len(arms), len(weekdays), len(arms) * len(weekdays)))
    rows, equity = [], {}
    started = time.time()
    for name in arms:
        for weekday in weekdays:
            run_arm = backtest_arm(
                stored["validation"], market["validation"], slots["validation"], SCORE_COLUMN,
                stored["validation"]["date"].min(), stored["validation"]["date"].max(),
                top_k, HORIZON, INITIAL_CAPITAL, MAX_POSITIONS, config_path,
                weekday=weekday, label=name, selection_filter=filters[name])
            equity[(name, weekday)] = run_arm["equity"]
            rows.append({"weekday": int(weekday), "arm": name, **_performance(run_arm)})
        print("    {:<10} {:.0f}s".format(name, time.time() - started), flush=True)
    validation_raw = pd.DataFrame(rows)
    validation = validation_raw.groupby("arm", as_index=False).mean(numeric_only=True)

    print("\n  test: {} arms on the production ISO week".format(len(arms)))
    test_rows, test_results = [], {}
    for name in arms:
        # A *fresh* filter for test: the diagnostics state must not carry the
        # validation window's basket counts into the test readout.
        arm_filter = (baseline_selection_filter(stored["test"], mapping) if arms[name] is None
                      else make_filter(stored["test"], mapping, arms[name]))
        run_arm = backtest_arm(
            stored["test"], market["test"], slots["test"], SCORE_COLUMN,
            stored["test"]["date"].min(), stored["test"]["date"].max(),
            top_k, HORIZON, INITIAL_CAPITAL, MAX_POSITIONS, config_path,
            label=name, selection_filter=arm_filter)
        test_results[name] = {"results": run_arm["results"],
                              "diagnostics": arm_filter.state["diagnostics"]}
        test_rows.append({"arm": name, **_performance(run_arm)})
    test_metrics = pd.DataFrame(test_rows).set_index("arm")

    print("\n  selection diagnostics (test window):")
    for name in arms:
        diag = test_results[name]["diagnostics"]
        print("    {:<10} kept {}/{} names | days short {} (worst -{}) | quota med {:.0f} | "
              "backfill depth med {:.2f} p95 {:.2f} max {:.2f}".format(
                  name, diag["kept_names"], diag["target_names"], diag["days_short_of_target"],
                  diag["worst_shortfall"], diag["quota_median"],
                  diag["backfill_depth_median"], diag["backfill_depth_p95"],
                  diag["backfill_depth_max"]))

    # --- gates ---------------------------------------------------------------
    folds = fold_boundaries(stored["validation"]["date"].min(),
                            stored["validation"]["date"].max(), FOLD_STEP_MONTHS)
    mean_equity = {name: phase_mean_equity({weekday: equity[(name, weekday)] for weekday in weekdays})
                   for name in arms}
    fold_excess = fold_excess_table(mean_equity, folds, INITIAL_CAPITAL, baseline_arm="uncapped")
    phase_excess = phase_excess_table(
        validation_raw, validation_raw[validation_raw["arm"] == "uncapped"], key_columns=("arm",))
    grid = validation.loc[:, ["arm", "total_return", "turnover_multiple", "filled_trade_count"]].rename(
        columns={"total_return": "validation_total_return",
                 "turnover_multiple": "validation_turnover_multiple"})
    fold_wins = fold_win_counts(fold_excess)
    print("\n  fold agreement ({} monthly folds, gate threshold {:g}):".format(
        len(folds), CONFIGURED_MIN_FOLD_AGREEMENT))
    for name in arms:
        print("    {:<10} {}/{} folds won ({:.1%})".format(
            name, fold_wins[name], len(folds), fold_wins[name] / float(len(folds))))

    outcome = select_configuration(
        grid, baseline_key={"arm": "uncapped"}, key_columns=("arm",),
        return_column="validation_total_return", turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess, phase_excess=phase_excess,
        turnover_floor_frac=CONFIGURED_TURNOVER_FLOOR,
        min_fold_agreement=float(CONFIGURED_MIN_FOLD_AGREEMENT))
    print("\n  selection rule: {}\n    {}".format(
        "NO CHANGE" if outcome.row is None else outcome.row["arm"], outcome.reason))

    # --- criteria 2 and 3 ----------------------------------------------------
    published = load_null_total_returns(dataset_dir, MODEL_ID, NULL_TAG, top_k,
                                        len(stored["test"]), max_positions=MAX_POSITIONS)
    baseline_return = float(test_metrics.loc["uncapped", "total_return"])
    print("\n  random null: {} published draws, mean {:+.2%}".format(
        len(published), published.mean()))
    print("  {:<10} test {:+.2%}  (baseline, uncapped)".format("uncapped", baseline_return))
    criteria = {}
    for name in arms:
        if name == "uncapped":
            continue
        value = float(test_metrics.loc[name, "total_return"])
        percentile = float((published <= value).mean())
        criteria[name] = {
            "test_return": value, "percentile": percentile,
            "improvement": value - baseline_return,
            "passes_percentile": bool(percentile >= MIN_PERCENTILE),
            "passes_improvement": bool(value - baseline_return >= MIN_EXCESS_IMPROVEMENT),
        }
        print("  {:<10} test {:+.2%}  percentile {:.1%}  vs baseline {:+.2%}".format(
            name, value, percentile, value - baseline_return))

    # --- the arm has to actually move the book -------------------------------
    print("\n  realised HELD concentration (the cap binds at selection, not at holding):")
    held, held_daily = {}, {}
    for name in ("uncapped", "cap30"):
        summary, metrics = held_concentration(
            test_results[name]["results"], market["test"], mapping, top_k, stored["test"], draws)
        held[name] = {**summary, "selection_diagnostics": test_results[name]["diagnostics"]}
        held_daily[name] = metrics
        print("    {:<10} held max med {:.1%} p95 {:.1%} max {:.1%}  same-width percentile med "
              "{:.1%}  names med {:.0f}".format(
                  name, summary["held_max_industry_weight_median"],
                  summary["held_max_industry_weight_p95"],
                  summary["held_max_industry_weight_max"],
                  summary["held_basket_percentile_median"], summary["held_n_hold_median"]))

    moved = (held["cap30"]["held_max_industry_weight_median"]
             < held["uncapped"]["held_max_industry_weight_median"])
    primary = criteria["cap30"]
    if not moved:
        verdict = ("cap did not move the held book -> this run is NOT a test of industry "
                   "neutralisation, whatever the equity curve says")
    elif primary["passes_percentile"] and primary["passes_improvement"]:
        verdict = ("primary criteria 2/3 pass; criterion 1 is the four gates and is reported "
                   "separately (fold gate closed at the configured default, not by this result)")
    else:
        verdict = "primary criteria 2/3 fail -> 不改"
    print("\n  E5b verdict: {}".format(verdict))

    return {
        "pool": POOL, "model_id": MODEL_ID, "top_k": top_k, "max_positions": MAX_POSITIONS,
        "horizon": HORIZON, "primary_share": MAX_INDUSTRY_SHARE, "sensitivity": list(SENSITIVITY),
        "min_fold_agreement": CONFIGURED_MIN_FOLD_AGREEMENT,
        "turnover_floor": CONFIGURED_TURNOVER_FLOOR,
        "criterion_min_improvement": MIN_EXCESS_IMPROVEMENT,
        "criterion_min_percentile": MIN_PERCENTILE,
        "fold_count": len(folds), "fold_wins": fold_wins,
        "selection": {"arm": None if outcome.row is None else str(outcome.row["arm"]),
                      "reason": outcome.reason},
        "baseline_test_return": baseline_return,
        "criteria": criteria, "held": held, "verdict": verdict,
        "zz1000": "not motivated (step 0: validation 59.5% vs test 79.5%, windows disagree)",
        "grid": grid, "fold_excess": fold_excess, "phase_excess": phase_excess,
        "validation": validation, "test_metrics": test_metrics,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--industry-path", default=INDUSTRY_SOURCE)
    parser.add_argument("--draws", type=int, default=PERMUTATIONS)
    parser.add_argument("--config-path", default="config/settings.yaml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run(Path(args.dataset_dir), output_dir, Path(args.industry_path),
                 draws=args.draws, config_path=args.config_path)
    prefix = "industry_cap_{}".format(POOL)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / ("{}_{}.csv".format(prefix, key)), index=(key == "test_metrics"))
    payload = {key: value for key, value in result.items() if not isinstance(value, pd.DataFrame)}
    (output_dir / (prefix + "_summary.json")).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n  wrote {}".format(output_dir / prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
