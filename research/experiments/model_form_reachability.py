"""E1: is *linearity* the bottleneck?  HistGB vs the published Ridge.

The whole model side of this project is ``sklearn.linear_model`` Ridge over 28/30
technical features -- 58/62 trainable parameters.  Nothing non-linear has ever
been tried.  This experiment holds everything else fixed and swaps only the
estimator, so the answer means something either way: if HistGB does not beat the
Ridge, linearity is not the constraint and E3/E4 (also model-side) are likely
dead too; if it beats it clearly, that is the one big lever left.

**What is held fixed** (all four are the point of the experiment, not incidental):

* features -- the published ``feature_columns``, read from the frozen pickle;
* split -- the published train/validation/test, via ``load_outcome_split``;
* preprocessing -- the **published** ``preprocessor``, not a refit one.  Reusing
  it removes a degree of freedom rather than adding one;
* the baseline -- the **published model objects themselves**, so ``ridge`` is
  literally the production arm and the comparison is paired by construction.

The anchor check below (published model -> disk ``entry_score``) is therefore not
just a sanity check on the plumbing: if it fails, the split/feature assembly here
is not the one the published numbers came from and nothing else in this file can
be read.

Pre-registered criteria: ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md``
§3 E1.  Four, all of which must hold to call linearity a bottleneck:

1. effect size -- test ``selected_excess_over_universe`` beats the Ridge by
   **>= +2.0pp** (the noise band is +-3~6pp, so anything smaller is unreadable);
2. same sign in both windows -- ``excess_return_rank_ic`` improves on validation
   **and** test;
3. engine -- the HistGB arm clears the four gates (baseline = the Ridge arm);
4. the HistGB arm's same-width random percentile is **>=95%**.

If only part holds, the pre-registered answer is "undecidable": do not adopt and
-- the part that matters -- **do not tune**.  HistGB has many hyperparameters and
"it lost, let me try again" is how this experiment would never end.  The
hyperparameters below are frozen as written.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.experiments.random_portfolio_baseline import _score_slots
from research.experiments.ranking_ablation_harness import (
    backtest_arm,
    basket_downside,
    fold_excess_table,
    load_null_total_returns,
)
from research.market_regime.logistic_model import transform_features
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import (
    fold_boundaries,
    phase_excess_table,
    select_configuration,
)
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_model import (
    evaluate_outcome_predictions,
    load_outcome_split,
)
from research.market_regime.trade_outcome_ranking_evaluation import published_top_k


ARMS = ("ridge", "histgb")
BASELINE_ARM = "ridge"
HORIZON = 10
FOLD_STEP_MONTHS = 1
#: The ISO-weekday family the phase gate is paired over: 0=Mon .. 4=Fri.
WEEKDAYS = (0, 1, 2, 3, 4)

#: **Frozen as pre-registered -- not tuned.**  Shallow trees and strong
#: regularisation because the training set is 2-3 x 10^5 rows with a known very
#: low signal-to-noise ratio, and because a tree model that wins only after
#: tuning would not answer the question this experiment asks.
HISTGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=500,
    l2_regularization=10.0,
    early_stopping=False,
    random_state=20260918,
)

#: ``dataset_id``/``trend_model_id`` are read off the published manifests; they
#: are what makes ``load_outcome_split`` return the same rows the published model
#: was fitted on.  ``null_tag`` names that pool's published 200-draw null.
POOLS = {
    "hs300": {
        "model_id": "hs300_pit_outcome",
        "dataset_id": "excess_hs300_pit_v1",
        "trend_model_id": "hs300_logistic_v2",
        "null_tag": "pit",
    },
    "zz1000": {
        "model_id": "zz1000_pit_outcome",
        "dataset_id": "excess_zz1000_pit_v1",
        "trend_model_id": "zz1000_logistic_v1",
        "null_tag": "zz1000_pit",
    },
}


def load_published_model(dataset_dir: Path, model_id: str) -> Dict[str, object]:
    """The frozen pickle: feature columns, preprocessor, and both Ridge heads."""

    path = dataset_dir / "trade_outcome_model_{}.pkl".format(model_id)
    if not path.exists():
        raise ValueError("no published model for model_id={!r} at {}".format(model_id, path))
    with path.open("rb") as handle:
        model = pickle.load(handle)
    missing = {"feature_columns", "preprocessor", "excess_model", "mae_model"}.difference(model)
    if missing:
        raise ValueError("{} missing {}".format(path.name, sorted(missing)))
    return model


def anchor_check(
    model: Mapping[str, object],
    test_samples: pd.DataFrame,
    horizon: int,
    risk_penalty: float,
    dataset_dir: Path,
    model_id: str,
    tolerance: float = 1e-12,
) -> float:
    """Re-score the test split with the published model and diff the disk score.

    This is §1.4's anchor: it is the only check that the cross-section assembled
    here is byte-for-byte the one the published numbers came from.  A failure is
    not a warning -- every other number in this file is then about a different
    universe.
    """

    # Both sides are normalised here rather than assumed: the prediction has to be
    # taken in the same row order the diff is done in, and a caller handing in a
    # raw CSV read (string dates, unsorted) should get "keys differ" only when the
    # rows genuinely differ, not whenever the frame was not pre-sorted.
    samples = test_samples.copy()
    samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
    samples = samples.sort_values(["date", "symbol"]).reset_index(drop=True)

    features = transform_features(samples, model["preprocessor"])
    predicted_excess = np.asarray(model["excess_model"].predict(features), dtype=float)
    predicted_mae = np.asarray(model["mae_model"].predict(features), dtype=float)
    recomputed = predicted_excess - float(risk_penalty) * np.maximum(-predicted_mae, 0.0)

    disk = pd.read_csv(
        dataset_dir / "trade_outcome_test_predictions_{}.csv".format(model_id),
        dtype={"symbol": str}, low_memory=False)
    disk["date"] = pd.to_datetime(disk["date"], errors="coerce")
    disk = disk.sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(disk) != len(samples):
        raise ValueError(
            "anchor row count differs: disk {} vs loaded {} -- the split loader is not "
            "reading the same cross-section".format(len(disk), len(samples)))
    keys_match = (
        disk["date"].equals(samples["date"])
        and disk["symbol"].astype(str).equals(samples["symbol"].astype(str))
    )
    if not keys_match:
        raise ValueError("anchor (date, symbol) keys differ from the disk file")
    delta = float(np.max(np.abs(recomputed - np.asarray(disk["entry_score"], dtype=float))))
    if not delta <= tolerance:
        raise ValueError("anchor FAILED: max |entry_score - recomputed| = {:.3e} > {:.1e}".format(
            delta, tolerance))
    return delta


def _fit_histgb(
    features: np.ndarray, targets: Sequence[np.ndarray]
) -> Tuple[HistGradientBoostingRegressor, ...]:
    return tuple(
        HistGradientBoostingRegressor(**HISTGB_PARAMS).fit(features, target) for target in targets
    )


def build_arm_scores(
    train_samples: pd.DataFrame,
    split_samples: pd.DataFrame,
    model: Mapping[str, object],
    horizon: int,
    risk_penalty: float,
) -> pd.DataFrame:
    """One column per arm for ``split_samples``, plus the labels.

    Both arms are scored on the same rows with the same published preprocessor;
    the only difference between the columns is the estimator that produced them.
    """

    excess_column, mae_column = "future_excess_return_{}d".format(horizon), "future_mae_{}d".format(horizon)
    train_features = transform_features(train_samples, model["preprocessor"])
    split_features = transform_features(split_samples, model["preprocessor"])
    train_excess = np.asarray(pd.to_numeric(train_samples[excess_column]), dtype=float)
    train_mae = np.asarray(pd.to_numeric(train_samples[mae_column]), dtype=float)

    histgb_excess, histgb_mae = _fit_histgb(train_features, (train_excess, train_mae))

    frame = split_samples.loc[:, ["date", "symbol", excess_column, mae_column]].copy()
    for name, excess_model, mae_model in (
        ("ridge", model["excess_model"], model["mae_model"]),
        ("histgb", histgb_excess, histgb_mae),
    ):
        predicted_excess = np.asarray(excess_model.predict(split_features), dtype=float)
        predicted_mae = np.asarray(mae_model.predict(split_features), dtype=float)
        frame["{}_predicted_excess".format(name)] = predicted_excess
        frame["{}_predicted_mae".format(name)] = predicted_mae
        frame[name] = predicted_excess - float(risk_penalty) * np.maximum(-predicted_mae, 0.0)
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def label_metrics(
    samples: pd.DataFrame,
    scored: pd.DataFrame,
    horizon: int,
    risk_penalty: float,
    top_k: float,
    split: str,
) -> pd.DataFrame:
    """Criteria 1 and 2, on the label cross-section ``evaluate_outcome_predictions``
    already defines -- identical code path to the one that produced the published
    numbers, so the comparison with the Ridge is like-for-like."""

    rows = []
    for arm in ARMS:
        metrics = evaluate_outcome_predictions(
            samples,
            np.asarray(scored["{}_predicted_excess".format(arm)], dtype=float),
            np.asarray(scored["{}_predicted_mae".format(arm)], dtype=float),
            horizon, risk_penalty, top_k, split, arm,
        )
        rows.append({"split": split, "arm": arm, **metrics})
    return pd.DataFrame(rows)


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
    weekdays: Sequence[int] = WEEKDAYS,
) -> Dict[str, object]:
    spec = POOLS[pool]
    model_id = spec["model_id"]
    top_k = published_top_k(dataset_dir, model_id)
    model = load_published_model(dataset_dir, model_id)
    feature_columns = list(model["feature_columns"])
    print("\n{}  model={}  top_k={} (published)  features={}".format(
        pool, model_id, top_k, len(feature_columns)))

    samples = {
        split: load_outcome_split(dataset_dir, spec["dataset_id"], split, feature_columns, horizon)
        for split in ("train", "validation", "test")
    }
    print("  rows: train={} validation={} test={}".format(
        *(len(samples[split]) for split in ("train", "validation", "test"))))

    manifest = json.loads(
        (dataset_dir / "trade_outcome_manifest_{}.json".format(model_id)).read_text(encoding="utf-8"))
    risk_penalty = float(manifest["selected_risk_penalty"])
    print("  risk_penalty={} (published)".format(risk_penalty))

    delta = anchor_check(model, samples["test"], horizon, risk_penalty, dataset_dir, model_id)
    print("  anchor: published model reproduces disk entry_score, max |delta| = {:.2e}".format(delta))

    scored, metrics = {}, []
    for split in ("validation", "test"):
        scored[split] = build_arm_scores(
            samples["train"], samples[split], model, horizon, risk_penalty)
        metrics.append(label_metrics(
            samples[split], scored[split], horizon, risk_penalty, top_k, split))
    metrics = pd.concat(metrics, ignore_index=True)

    def metric(split, arm, column):
        return float(metrics.loc[(metrics["split"] == split) & (metrics["arm"] == arm), column].iloc[0])

    print("\n  label cross-section ({:.0%} Top-K, published risk_penalty):".format(top_k))
    for split in ("validation", "test"):
        for arm in ARMS:
            print("    {:<7} {:<10} excess/universe={:+.4%}  excess_rank_ic={:+.4f}".format(
                split, arm, metric(split, arm, "selected_excess_over_universe"),
                metric(split, arm, "excess_return_rank_ic")))

    # Criterion 1 and 2, evaluated here and reported rather than branched on --
    # the pre-registered rule is "all four or undecidable", so a partial pass is
    # a finding, not a licence to keep going.
    effect = {
        "test_excess_over_universe_gain_pp": 100.0 * (
            metric("test", "histgb", "selected_excess_over_universe")
            - metric("test", "ridge", "selected_excess_over_universe")),
        "validation_excess_over_universe_gain_pp": 100.0 * (
            metric("validation", "histgb", "selected_excess_over_universe")
            - metric("validation", "ridge", "selected_excess_over_universe")),
        "validation_rank_ic_gain": (
            metric("validation", "histgb", "excess_return_rank_ic")
            - metric("validation", "ridge", "excess_return_rank_ic")),
        "test_rank_ic_gain": (
            metric("test", "histgb", "excess_return_rank_ic")
            - metric("test", "ridge", "excess_return_rank_ic")),
    }
    criterion_1 = effect["test_excess_over_universe_gain_pp"] >= 2.0
    criterion_2 = effect["validation_rank_ic_gain"] > 0 and effect["test_rank_ic_gain"] > 0
    print("\n  criterion 1 (test excess/universe gain >= +2.0pp): {:.2f}pp -> {}".format(
        effect["test_excess_over_universe_gain_pp"], "PASS" if criterion_1 else "FAIL"))
    print("  criterion 2 (rank IC up on BOTH windows): val {:+.4f}, test {:+.4f} -> {}".format(
        effect["validation_rank_ic_gain"], effect["test_rank_ic_gain"],
        "PASS" if criterion_2 else "FAIL"))

    null_returns = load_null_total_returns(
        dataset_dir, model_id, spec["null_tag"], top_k, len(scored["test"]))
    print("  null: {} draws, mean {:+.2%} (verified top_k and row count)".format(
        len(null_returns), null_returns.mean()))

    raw = {}
    slots = {}
    for split in ("validation", "test"):
        raw[split], skipped = build_rank_market_data(
            scored[split], price_history_dir, ARMS[0], price_history_dirs=price_history_dirs)
        if not raw[split]:
            raise ValueError("missing price data for {} {}".format(pool, split))
        slots[split] = _score_slots(scored[split], raw[split])
        print("  {} skipped prices: {}".format(split, skipped if skipped else "none"))

    # Validation runs over the whole weekday family, both arms, so the phase gate
    # is paired: comparing a candidate's worst schedule against the baseline's
    # average is what makes an unpaired phase gate reject arms that win every
    # day.  ``weekday=None`` on test is the production ISO week.
    print("\n  running validation over {} weekdays x {} arms...".format(len(weekdays), len(ARMS)))
    validation_rows, validation_equity = [], {}
    for weekday in weekdays:
        for arm in ARMS:
            run = backtest_arm(
                scored["validation"], raw["validation"], slots["validation"], arm,
                scored["validation"]["date"].min(), scored["validation"]["date"].max(),
                top_k, horizon, initial_capital, max_positions, config_path, weekday=weekday)
            validation_equity[(arm, weekday)] = run["equity"]
            validation_rows.append({
                "weekday": int(weekday), "arm": arm,
                **{key: value for key, value in run["performance"].items() if key != "strategy"},
            })
    validation_raw = pd.DataFrame(validation_rows)
    # Phase-average the *numeric* columns only: ``summarize_backtest`` carries the
    # window bounds as strings, and a blanket mean raises on them.
    averaged_columns = [
        column for column in validation_raw.columns
        if column not in ("arm", "weekday")
        and pd.api.types.is_numeric_dtype(validation_raw[column])
    ]
    validation = validation_raw.groupby("arm", as_index=False)[averaged_columns].mean()
    # Merged on ``arm``: assigning the groupby result positionally puts it against
    # a RangeIndex and every value silently becomes NaN.
    validation = validation.merge(
        validation_raw.groupby("arm")["total_return"].agg(["min", "max"]).rename(
            columns={"min": "phase_min", "max": "phase_max"}).reset_index(),
        on="arm", how="left")

    print("  running test arms (production ISO week)...")
    test_runs = {
        arm: backtest_arm(
            scored["test"], raw["test"], slots["test"], arm,
            scored["test"]["date"].min(), scored["test"]["date"].max(),
            top_k, horizon, initial_capital, max_positions, config_path)
        for arm in ARMS
    }

    rows: List[Dict[str, object]] = []
    for arm in ARMS:
        performance = test_runs[arm]["performance"]
        rows.append({
            "split": "test", "arm": arm,
            **{"test_{}".format(key): value for key, value in performance.items() if key != "strategy"},
            **{"test_{}".format(key): value for key, value in basket_downside(
                scored["test"], test_runs[arm]["selected"], horizon).items()},
        })
    test_metrics = pd.DataFrame(rows)

    print("\n  net total return:")
    for arm in ARMS:
        value = float(validation.loc[validation["arm"] == arm, "total_return"].iloc[0])
        test_value = float(test_metrics.loc[test_metrics["arm"] == arm, "test_total_return"].iloc[0])
        print("    {:<7} validation(weekday-mean)={:>8.2%} (min {:>7.2%}, max {:>7.2%})  test={:>8.2%}".format(
            arm, value,
            float(validation.loc[validation["arm"] == arm, "phase_min"].iloc[0]),
            float(validation.loc[validation["arm"] == arm, "phase_max"].iloc[0]),
            test_value))

    folds = fold_boundaries(
        scored["validation"]["date"].min(), scored["validation"]["date"].max(), FOLD_STEP_MONTHS)
    # Each arm's equity is averaged over weekdays *first*, then folded, so the
    # fold gate asks about the schedule family rather than about one strike.
    weekday_mean_equity = {
        arm: pd.concat(
            [validation_equity[(arm, weekday)] for weekday in weekdays], axis=1
        ).mean(axis=1).sort_index()
        for arm in ARMS
    }
    fold_excess = fold_excess_table(
        weekday_mean_equity, folds, initial_capital, baseline_arm=BASELINE_ARM)
    phase_excess = phase_excess_table(
        validation_raw, validation_raw[validation_raw["arm"] == BASELINE_ARM],
        key_columns=("arm",))
    grid = validation.loc[:, ["arm", "total_return", "turnover_multiple"]].rename(
        columns={"total_return": "validation_total_return",
                 "turnover_multiple": "validation_turnover_multiple"})
    outcome = select_configuration(
        grid,
        baseline_key={"arm": BASELINE_ARM},
        key_columns=("arm",),
        return_column="validation_total_return",
        turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess,
        phase_excess=phase_excess,
    )
    criterion_3 = outcome.row is not None and str(outcome.row["arm"]) == "histgb"
    print("\n  selection rule: {}\n    {}".format(
        "NO CHANGE" if outcome.row is None else outcome.row["arm"], outcome.reason))
    print("  criterion 3 (histgb clears the four gates): {}".format(
        "PASS" if criterion_3 else "FAIL"))

    percentiles = {
        arm: float((null_returns <= float(
            test_metrics.loc[test_metrics["arm"] == arm, "test_total_return"].iloc[0])).mean())
        for arm in ARMS
    }
    print("  test percentile vs the shared {}-draw null (mean {:+.2%}):".format(
        len(null_returns), null_returns.mean()))
    for arm in ARMS:
        print("    {:<7} {:>7.1%}".format(arm, percentiles[arm]))
    criterion_4 = percentiles["histgb"] >= 0.95
    print("  criterion 4 (histgb percentile >= 95%): {}".format(
        "PASS" if criterion_4 else "FAIL"))

    criteria = {
        "criterion_1_effect_size": criterion_1,
        "criterion_2_cross_window_rank_ic": criterion_2,
        "criterion_3_four_gates": criterion_3,
        "criterion_4_null_percentile": criterion_4,
    }
    verdict = "linearity is the bottleneck" if all(criteria.values()) else "undecidable"
    print("\n  VERDICT: {} ({} of 4 criteria)".format(
        verdict, sum(criteria.values())))

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "model_form_reachability_{}".format(pool)
    metrics.to_csv(output_dir / (prefix + "_label_metrics.csv"), index=False)
    test_metrics.to_csv(output_dir / (prefix + "_test_metrics.csv"), index=False)
    validation_raw.to_csv(output_dir / (prefix + "_validation_phases.csv"), index=False)
    fold_columns = ["fold_{}".format(index) for index in range(len(folds))]
    grid.merge(phase_excess[["arm", "phase_excess_min", "phases"]], on="arm", how="left").merge(
        fold_excess[["arm"] + fold_columns], on="arm", how="left").to_csv(
        output_dir / (prefix + "_validation_grid.csv"), index=False)

    report = {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "risk_penalty": risk_penalty,
        "baseline_arm": BASELINE_ARM,
        "histgb_params": HISTGB_PARAMS,
        "anchor_max_abs_delta": delta,
        "effect": effect,
        "criteria": criteria,
        "verdict": verdict,
        "test_percentile_vs_shared_null": percentiles,
        "null_draws": int(len(null_returns)),
        "null_mean_total_return": float(null_returns.mean()),
        "selection_rule": {
            "row": None if outcome.row is None else str(outcome.row["arm"]),
            "reason": outcome.reason,
            "gates": dict(outcome.gates),
        },
        "execution": {
            "horizon": int(horizon),
            "max_positions": max_positions,
            "initial_capital": initial_capital,
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
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--weekdays", default=",".join(str(day) for day in WEEKDAYS),
                        help="ISO weekdays for the phase family; a single value runs one schedule.")
    args = parser.parse_args(argv)
    weekdays = tuple(int(item) for item in args.weekdays.split(",") if item.strip())
    pools = [args.pool] if args.pool else sorted(POOLS)
    for pool in pools:
        evaluate_pool(
            pool, Path(args.dataset_dir), Path(args.output_dir),
            horizon=args.horizon, initial_capital=args.initial_capital,
            max_positions=args.max_positions, price_history_dir=args.price_history_dir,
            config_path=args.config, weekdays=weekdays,
        )


if __name__ == "__main__":
    main()
