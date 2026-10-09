"""P0-4: how often does a **no-skill** arm clear the fold gate?

Pre-registration: ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md`` §2 P0-4.

The rule's fourth gate asks for ``fold_agreement >= min_fold_agreement``, default
**1.0**, over the 29 monthly validation folds -- i.e. "beat the baseline in every
one of 29 months".  E6 measured the consequence: 36 grid cells x 2 pools, **36/36
rows failed in both pools**, and the best candidate won only 16/29 = 55%.

**That threshold is not strict, it is unreachable, and the reason is arithmetic.**
A score-free arm is a random permutation of the same cross-section pushed through
the same execution chain, so with no skill its per-fold excess is a coin flip and
its win count is approximately ``Binomial(29, 0.5)`` -- whose 95th percentile is
**20/29 = 0.6897** (exactly: ``P(X>=20)=3.07%``, ``P(X>=19)=6.80%``).  Requiring
1.0 asks for an event whose probability under the null is **2**\\ :sup:`-29`
``~= 1.9e-9``.  A gate that no arm can pass is not filtering, and "the fold gate
rejected it" carries no information.

So this experiment measures the null distribution and prices the gate in the only
honest currency available: the **size** of the test.  The threshold rule is
pre-registered and reads **no candidate returns**:

    theta = the smallest k/29 such that P_null(X >= k) <= 0.05

That is the same *method* as the turnover floor's -- a measured cut, not a tuned
one: it measures a property of the **test**, not of the candidates, and the
output is independent of which arms happen to be under consideration.  Compare
``probability_trade_evaluation``'s 0.05, whose justification is that two
thresholds excluding the same set are equivalent so the smaller one is kept.

Three things this experiment is **not**:

* It is **not** a way to salvage E6.  The calibration is already known not to
  move E6: the best candidate's 16/29 sits below a null-derived 20/29.  E6's grid
  also carries test readings, so re-scanning it under a new theta would be
  tuning on a read window.  The grid is read here for **description** only, and
  the pre-registration fixes that reading as not-a-selection-input.
* It does **not** change any default.  Producing theta is this experiment's whole
  output; adopting it is a separate decision.
* It does **not** touch the fold **granularity**.  ``FOLD_STEP_MONTHS`` and the
  29-fold count are held fixed -- moving the threshold and the statistic at the
  same time would leave no way to tell which one did the work.

The statistic is measured exactly as the gate consumes it: on the weekday-family
**averaged** equity curve (``basket_width_grid``'s convention), so each draw costs
one engine run per phase.  Per-phase agreements fall out for free and are
reported alongside, since how much the phase average matters is itself a result.

Example::

    python -m research.experiments.fold_gate_calibration --seeds 5 --pool hs300   # smoke
    python -m research.experiments.fold_gate_calibration                          # both pools, N=200
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.ranking_ablation_harness import backtest_arm
from research.experiments.random_portfolio_baseline import _score_slots
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import fold_boundaries, fold_returns_from_equity
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)


HORIZON = 10
#: Held fixed by the pre-registration: only the threshold moves in this experiment.
FOLD_STEP_MONTHS = 1
#: The phase family the statistic is averaged over -- E6's convention, kept.
WEEKDAYS = (0, 1, 2, 3, 4)
#: The gate's own default, quoted in every report so the gap is never implicit.
CONFIGURED_MIN_FOLD_AGREEMENT = 1.0
#: Pre-registered size of the gate.
ALPHA = 0.05
#: Production cap; the baseline cell is ``(published top_k, 20)`` as everywhere else.
BASELINE_MAX_POSITIONS = 20

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome"},
    "zz1000": {"model_id": "zz1000_pit_outcome"},
}


def fold_agreement(
    fold_returns: Sequence[float], baseline_fold_returns: Sequence[float]
) -> Dict[str, float]:
    """Fraction of folds the arm beat the baseline on -- the gate's statistic.

    A fold whose return is unmeasurable counts as a **loss**, which is what the
    gate does (``(table[fold_columns] > 0).sum()`` is False for NaN).  The count
    is returned rather than swallowed: silently scoring an arm on 26 folds while
    calling it "out of 29" is the kind of drift the rest of this package pins.
    """

    arm_values = np.asarray(fold_returns, dtype=float)
    base_values = np.asarray(baseline_fold_returns, dtype=float)
    if arm_values.shape != base_values.shape:
        raise ValueError(
            "fold vectors are not the same length: arm={}, baseline={}".format(
                arm_values.shape, base_values.shape))
    if not len(arm_values):
        raise ValueError("no folds to score against")
    finite = np.isfinite(arm_values) & np.isfinite(base_values)
    wins = int(np.sum(finite & (arm_values > base_values)))
    return {
        "wins": float(wins),
        "folds": float(len(arm_values)),
        "unscored_folds": float(len(arm_values) - int(finite.sum())),
        "agreement": float(wins) / float(len(arm_values)),
    }


def _win_counts(agreements: np.ndarray, folds: int) -> np.ndarray:
    """Turn agreements back into integer win counts, refusing a mismatched ``folds``.

    ``agreement = wins / folds`` is only invertible when the caller's ``folds``
    is the one the agreements were computed on; passing 29 for a run made with 12
    would otherwise round to a plausible-looking integer vector.
    """

    counts = np.rint(np.asarray(agreements, dtype=float) * int(folds)).astype(int)
    residual = np.abs(np.asarray(agreements, dtype=float) * int(folds) - counts)
    if np.any(residual > 1e-9):
        raise ValueError(
            "agreements are not multiples of 1/{}; they were measured on a different "
            "fold count (worst residual {:.3g})".format(folds, float(residual.max())))
    if np.any(counts < 0) or np.any(counts > int(folds)):
        raise ValueError("win counts outside 0..{}".format(folds))
    return counts


def calibrate_fold_threshold(
    agreements: np.ndarray,
    folds: int,
    alpha: float = ALPHA,
    bootstrap: int = 2000,
    seed: int = 20260919,
) -> Dict[str, object]:
    """``theta = smallest k/folds whose null exceedance is at or below ``alpha``.

    Reads **only** the null agreements.  Nothing about any candidate enters, which
    is what lets the result be called measured rather than tuned -- there is a
    test pinning that (permuting a candidate column cannot move theta, because no
    candidate column is an argument).

    Two ends are reported and they are **not interchangeable**.  ``theta`` is the
    point estimate -- the rule applied to the observed draws.  The pre-registration
    (P0-4 criterion 2) adopts ``theta_bootstrap_upper`` instead, because N=200
    draws pin a discrete order statistic loosely and the failure this experiment
    exists to prevent is a threshold low enough to admit no-skill arms.  So
    **``theta`` is what the rule returned; ``theta_bootstrap_upper`` is the value
    the experiment is allowed to use.**  ``theta_bootstrap_p975`` sits between the
    two and is reported so the report can show the whole bootstrap without
    re-deriving it.  (A docstring here previously claimed ``theta`` *was* the
    upper bound; it was not, and the difference is one fold in both pools.)

    ``theta > 1`` means even ``k = folds`` exceeds ``alpha`` -- no-skill arms do
    win every fold often enough that the statistic cannot separate anyone.  That
    is a statement about the *statistic*, not the threshold, and it is reported
    as such rather than clipped to 1.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1), got {!r}".format(alpha))
    values = np.asarray(agreements, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("agreements must be a non-empty 1-D array")
    counts = _win_counts(values, folds)

    def rule(sample: np.ndarray) -> float:
        for k in range(0, int(folds) + 1):
            if float(np.mean(sample >= k)) <= float(alpha):
                return float(k) / float(folds)
        return float(int(folds) + 1) / float(folds)

    theta = rule(counts)
    rng = np.random.default_rng(seed)
    draws = [
        rule(counts[rng.integers(0, len(counts), len(counts))])
        for _ in range(int(bootstrap))
    ]
    # ``theta`` is a discrete order statistic, so the bootstrap spread around it
    # is the honest statement of how well N draws pin it down.  ``max`` is the
    # pre-registered upper end and is deliberately crude; the 97.5th percentile
    # is reported beside it so the report can show both without re-deriving them.
    histogram = {
        "{}/{}".format(k, int(folds)): int(np.sum(counts == k))
        for k in range(0, int(folds) + 1)
    }
    # Binomial(29, 0.5) is the no-skill reference the pre-registration names.  The
    # variance ratio is the second pre-registered output: it says how much of the
    # 29-fold spread is independent information rather than one bet repeated.
    binomial_variance = float(folds) * 0.25
    observed_variance = float(np.var(counts, ddof=1)) if len(counts) > 1 else 0.0
    return {
        "folds": int(folds),
        "alpha": float(alpha),
        "draws": int(len(counts)),
        "theta": float(theta),
        "theta_bootstrap_upper": float(max(draws)) if draws else float(theta),
        "theta_bootstrap_p975": float(np.percentile(draws, 97.5)) if draws else float(theta),
        "theta_bootstrap_median": float(np.median(draws)) if draws else float(theta),
        "theta_is_reachable": bool(theta <= 1.0),
        "null_pass_rate_at_theta": float(np.mean(counts >= int(round(theta * folds)))),
        "null_pass_rate_at_configured": float(np.mean(counts >= int(round(
            CONFIGURED_MIN_FOLD_AGREEMENT * folds)))),
        "null_mean_agreement": float(values.mean()),
        "null_sd_agreement": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "null_mean_wins": float(counts.mean()),
        "null_variance_ratio_vs_binomial": (
            observed_variance / binomial_variance if binomial_variance else float("nan")),
        "histogram": histogram,
    }


def _phase_mean_equity(equity_by_weekday: Mapping[int, pd.Series]) -> pd.Series:
    """Average the phase family's equity curves -- how the gate's folds are read."""

    return pd.concat(list(equity_by_weekday.values()), axis=1).mean(axis=1).sort_index()


def _run_phases(
    scores: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    slots: Dict[str, tuple],
    column: str,
    top_k: float,
    horizon: int,
    initial_capital: float,
    config_path: str,
    weekdays: Sequence[int],
    label: str,
) -> Dict[int, pd.Series]:
    """One engine run per phase, reusing the shared harness rather than re-wiring it."""

    start, end = scores["date"].min(), scores["date"].max()
    return {
        int(weekday): backtest_arm(
            scores, raw_market_data, slots, column, start, end, top_k, horizon,
            initial_capital, BASELINE_MAX_POSITIONS, config_path,
            weekday=int(weekday), label=label)["equity"]
        for weekday in weekdays
    }


def read_grid_agreements(dataset_dir: Path, pool: str, folds: int) -> Optional[np.ndarray]:
    """E6's recorded per-cell agreements, for **description** only.

    The pre-registration allows exactly one use of these: reporting where the
    already-measured grid sits relative to the null.  They may not be turned into
    a selection -- the grid carries test readings, so choosing a cell off them is
    tuning on a read window.  Returns ``None`` when E6 has not been run here.
    """

    path = dataset_dir / "basket_width_{}_validation_grid.csv".format(pool)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [column for column in frame.columns if column.startswith("fold_")]
    if len(columns) != int(folds):
        raise ValueError(
            "{} has {} fold columns but this run uses {}".format(
                path.name, len(columns), folds))
    return (frame[columns] > 0).sum(axis=1).to_numpy(dtype=float) / float(folds)


def run_pool(
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
    bootstrap: int = 2000,
    verbose: bool = True,
) -> Dict[str, object]:
    """Null distribution of ``fold_agreement`` for one pool, plus the calibrated theta."""

    model_id = POOLS[pool]["model_id"]
    top_k = published_top_k(dataset_dir, model_id)
    print("\n{}  model={}  baseline=(tk{:g}, mp{}) published width  phases={}".format(
        pool, model_id, top_k, BASELINE_MAX_POSITIONS, list(weekdays)))

    scores = load_outcome_scores(dataset_dir, model_id, "validation", horizon)
    raw_market_data, skipped = build_rank_market_data(
        scores, price_history_dir, SCORE_COLUMN, price_history_dirs=price_history_dirs)
    if not raw_market_data:
        raise ValueError("no validation price data for {}".format(model_id))
    slots = _score_slots(scores, raw_market_data)
    print("  validation: {} rows, {} dates, {} symbols  ({} skipped prices)".format(
        len(scores), scores["date"].nunique(), scores["symbol"].nunique(),
        skipped if skipped else "none"))

    folds = fold_boundaries(
        scores["date"].min(), scores["date"].max(), FOLD_STEP_MONTHS)
    # The baseline is the production cell, measured on the same phase family --
    # the gate only ever compares arms within one phase convention.
    baseline_equity = _run_phases(
        scores, raw_market_data, slots, SCORE_COLUMN, top_k, horizon,
        initial_capital, config_path, weekdays, "baseline")
    baseline_fold = {
        "mean": fold_returns_from_equity(
            _phase_mean_equity(baseline_equity), folds, initial_capital),
        **{
            int(weekday): fold_returns_from_equity(series, folds, initial_capital)
            for weekday, series in baseline_equity.items()
        },
    }
    selfcheck = fold_agreement(baseline_fold["mean"], baseline_fold["mean"])
    if selfcheck["wins"] != 0.0:
        raise ValueError(
            "the baseline does not agree with itself on {} folds".format(selfcheck["wins"]))
    print("  {} monthly folds from {} to {}   baseline self-agreement 0/{} (as designed)".format(
        len(folds), scores["date"].min().date(), scores["date"].max().date(), len(folds)))

    probe = scores.copy()
    values = np.asarray(scores[SCORE_COLUMN], dtype=float)
    permuted_column = "_p0_4_permuted"
    probe[permuted_column] = values
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    started = time.time()
    for draw in range(int(seeds)):
        shuffled = rng.permutation(values)
        probe[permuted_column] = shuffled
        equity = _run_phases(
            probe, raw_market_data, slots, permuted_column, top_k, horizon,
            initial_capital, config_path, weekdays, "null")
        averaged = fold_agreement(
            fold_returns_from_equity(_phase_mean_equity(equity), folds, initial_capital),
            baseline_fold["mean"])
        row = {"draw": int(draw), **averaged}
        for weekday, series in equity.items():
            row["agreement_weekday_{}".format(weekday)] = fold_agreement(
                fold_returns_from_equity(series, folds, initial_capital),
                baseline_fold[int(weekday)])["agreement"]
        rows.append(row)
        if verbose and (draw % 5 == 0 or draw == int(seeds) - 1):
            print("  draw {:>4}/{:<4} agreement={:.3f} ({:.0f}/{:.0f})  [{:.0f}s]".format(
                draw + 1, int(seeds), row["agreement"], row["wins"], row["folds"],
                time.time() - started), flush=True)

    draws = pd.DataFrame(rows)
    calibration = calibrate_fold_threshold(
        draws["agreement"].to_numpy(dtype=float), len(folds), bootstrap=bootstrap, seed=seed)

    # Descriptive only -- see ``read_grid_agreements`` and the module docstring.
    grid_agreements = read_grid_agreements(dataset_dir, pool, len(folds))
    grid_report = None
    if grid_agreements is not None:
        theta = float(calibration["theta"])
        grid_report = {
            "cells": int(len(grid_agreements)),
            "max_agreement": float(grid_agreements.max()),
            "cells_at_or_above_theta": int(np.sum(grid_agreements >= theta)),
            "cells_at_or_above_configured": int(
                np.sum(grid_agreements >= CONFIGURED_MIN_FOLD_AGREEMENT)),
            "null_share_below_grid_max": float(
                np.mean(draws["agreement"].to_numpy(dtype=float) < grid_agreements.max())),
        }
        print("\n  [description only] E6's {} cells: max agreement {:.3f}; "
              "{:.1%} of null draws fall below it".format(
                  grid_report["cells"], grid_report["max_agreement"],
                  grid_report["null_share_below_grid_max"]))
        print("  [description only] cells clearing theta {}: {}; clearing the configured "
              "{}: {}".format(theta, grid_report["cells_at_or_above_theta"],
                              CONFIGURED_MIN_FOLD_AGREEMENT,
                              grid_report["cells_at_or_above_configured"]))

    summary: Dict[str, object] = {
        "pool": pool,
        "model_id": model_id,
        "split": "validation",
        "top_k": float(top_k),
        "max_positions": BASELINE_MAX_POSITIONS,
        "horizon": int(horizon),
        "fold_step_months": FOLD_STEP_MONTHS,
        "fold_count": int(len(folds)),
        "weekdays": [int(day) for day in weekdays],
        "seeds": int(seeds),
        "seed": int(seed),
        "alpha": float(ALPHA),
        "configured_min_fold_agreement": float(CONFIGURED_MIN_FOLD_AGREEMENT),
        "validation_rows": int(len(scores)),
        "skipped_prices": skipped,
        "calibration": calibration,
        "per_phase_agreement": {
            "weekday_{}".format(int(day)): {
                "mean": float(draws["agreement_weekday_{}".format(int(day))].mean()),
                "theta": float(calibrate_fold_threshold(
                    draws["agreement_weekday_{}".format(int(day))].to_numpy(dtype=float),
                    len(folds), bootstrap=0, seed=seed)["theta"]),
            }
            for day in weekdays
        },
        "grid_description_only": grid_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "fold_gate_calibration_{}".format(pool)
    draws.to_csv(output_dir / (prefix + "_draws.csv"), index=False)
    (output_dir / (prefix + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _print_report(pool, calibration, summary["per_phase_agreement"])
    return summary


def _print_report(pool: str, calibration: Mapping[str, object], per_phase) -> None:
    print("\n{}  null fold_agreement over {} draws x {} folds".format(
        pool, calibration["draws"], calibration["folds"]))
    print("  mean {:.3f}  sd {:.3f}  (Binomial sd would be {:.3f}; variance ratio {:.2f})".format(
        calibration["null_mean_agreement"], calibration["null_sd_agreement"],
        (0.25 / float(calibration["folds"])) ** 0.5,
        calibration["null_variance_ratio_vs_binomial"]))
    print("  pass rate at the CONFIGURED threshold {:.1f}: {:.4%}   (null probability "
          "2^-{} = {:.2e})".format(
              CONFIGURED_MIN_FOLD_AGREEMENT, calibration["null_pass_rate_at_configured"],
              calibration["folds"], 2.0 ** -float(calibration["folds"])))
    print("  theta = {:.4f} ({}), bootstrap upper {:.4f}".format(
        calibration["theta"],
        "reachable" if calibration["theta_is_reachable"] else "UNREACHABLE",
        calibration["theta_bootstrap_upper"]))
    print("  null pass rate at theta: {:.3%} (target alpha {:.0%})".format(
        calibration["null_pass_rate_at_theta"], calibration["alpha"]))
    print("  per-phase theta (single schedule instead of the average): {}".format(
        ", ".join("{}={:.3f}".format(name.replace("weekday_", "wd"), value["theta"])
                  for name, value in per_phase.items())))


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS) + ["both"], default="both")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--weekdays", default=",".join(str(day) for day in WEEKDAYS),
                        help="ISO weekdays for the phase family; one value runs one schedule.")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--price-history-dirs", default=None,
                        help="Comma-separated roots, concatenated at load time.")
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    dirs = [item.strip() for item in args.price_history_dirs.split(",") if item.strip()] \
        if args.price_history_dirs else None
    weekdays = tuple(int(item) for item in args.weekdays.split(",") if item.strip())
    pools = sorted(POOLS) if args.pool == "both" else [args.pool]
    summaries = [
        run_pool(
            pool, Path(args.dataset_dir), Path(args.output_dir),
            horizon=args.horizon, initial_capital=args.initial_capital,
            price_history_dir=args.price_history_dir, config_path=args.config_path,
            price_history_dirs=dirs, weekdays=weekdays, seeds=args.seeds,
            seed=args.seed, bootstrap=args.bootstrap)
        for pool in pools
    ]
    if len(summaries) > 1:
        print("\nSUMMARY  theta per pool: {}".format(", ".join(
            "{}={:.4f}".format(item["pool"], item["calibration"]["theta"])
            for item in summaries)))
    return summaries


if __name__ == "__main__":
    main()
