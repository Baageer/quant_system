"""E9 step 0, **revised**: a model-level screen that replaces a tautological one.

The pre-registered step 0 (``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md`` section 3)
proposed screening the re-represented features with the same ladder
``reversal_feature_study`` / ``fundamental_feature_study`` use -- daily Rank IC,
ICIR, win rate, quantile monotonicity -- and closing the whole axis if nothing
survived.  **That screen cannot see the representation at all.**  Every statistic
in the ladder is a *within-date rank* statistic, and cross-sectional demeaning,
z-scoring and ranking are all strictly increasing within-date transforms, so they
leave every daily rank -- and therefore the mean IC, the ICIR, the win rate, the
quantile spreads and the monotonicity -- **bit-identical**.  Measured on the
HS300 pit validation split (28 features, 125,274 rows, 579 dates): max |delta|
over all 28 features for both transforms was exactly ``0.0`` on every statistic,
including the per-day observation count.  Screening there would have produced a
copy of the raw table that *looks* like evidence for the transform.

So the screen moves up one level, to where the representation can actually act:
the **score**.  The effect of a per-date representation is not per-feature -- it
is on how the features *combine*, because a linear model fitted on the pooled
panel sees a different design matrix when each feature is re-expressed within
date.  This module therefore refits the production two-head Ridge on the
re-represented panel and compares the resulting **entry_score** against the raw
arm on validation.

What this screen can and cannot do:

* it **can** close the axis cheaply -- if a variant leaves the top-k basket
  essentially unchanged, no backtest of that variant can differ;
* it **cannot** conclude that a variant works.  A validation IC gain is a
  hypothesis; only the pre-registered four gates plus the test window can
  support a claim.  ``screen_decision`` returns ``advances`` for *go/no-go on
  spending step 1's compute*, nothing more.

Variants (all applied per date, so only same-date information is used -- the
entry remains T+1):

======================  ==========================================================
``raw``                 production pipeline, no representation change (baseline)
``demean``              ``x - mean_t(x)``, then the production preprocessor
``demean_post``         production preprocessor, **then** ``z - mean_t(z)``
``zscore``              ``(x - mean_t(x)) / sd_t(x)``, then the production preprocessor
``rank``                ``rank_t(x) / n_t``, then the production preprocessor
======================  ==========================================================

``demean_post`` exists to separate two mechanisms that the pre-registered design
conflated.  Rescaling a column changes what Ridge's penalty does to it, and
``demean`` changes the column's scale as well as removing its between-date level;
``demean_post`` removes the between-date level **with the scale left exactly as
the production standardizer set it**.  If the two arms behave alike the mechanism
is the demeaning; if they diverge the pre-registered ``demean`` arm was partly a
re-parameterisation of the penalty.

Example::

    python -m research.experiments.cross_sectional_representation_screen
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.market_regime.logistic_model import fit_preprocessor, transform_features
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.trade_outcome_model import (
    decision_score,
    load_outcome_split,
    target_columns,
)


HORIZON = 10
TOP_K = 0.05
# Pre-registered screen thresholds.  Frozen before either pool was screened *by
# this script*; HS300's numbers were already visible from the exploratory probe
# that motivated the revision, so for HS300 these are post-hoc by construction.
# See the plans doc for the exact wording of that caveat.
MIN_IC_DELTA = 0.005
MIN_HALF_DELTA = 0.002
MIN_BASKET_DISAGREEMENT = 0.05

VARIANTS = ("raw", "demean", "demean_post", "zscore", "rank")
CROSS_SECTIONAL_VARIANTS = ("demean", "zscore", "rank")

POOLS = {
    "hs300": {"dataset_id": "excess_hs300_pit_v1", "model_id": "hs300_pit_outcome"},
    "zz1000": {"dataset_id": "excess_zz1000_pit_v1", "model_id": "zz1000_pit_outcome"},
}


def load_model_settings(dataset_dir: Path, model_id: str) -> Dict[str, object]:
    """Read the frozen feature list and hyper-parameters of a published model.

    Nothing here is re-selected: the point of the screen is to change the
    representation and nothing else, so alpha and risk_penalty come from the
    manifest that produced the published arm.
    """

    path = Path(dataset_dir) / "trade_outcome_manifest_{}.json".format(model_id)
    manifest = json.loads(path.read_text())
    return {
        "feature_columns": list(manifest["feature_columns"]),
        "alpha": float(manifest["selected_alpha"]),
        "risk_penalty": float(manifest["selected_risk_penalty"]),
        "top_k": float(manifest.get("top_k", TOP_K)),
        "dataset_id": manifest.get("dataset_id"),
    }


def apply_representation(
    samples: pd.DataFrame, feature_columns: Sequence[str], variant: str
) -> pd.DataFrame:
    """Re-express the raw features within each date, before the preprocessor.

    The transform is per-date by construction, so a row's features depend only on
    that date's cross-section -- information available at ``t``'s close, with the
    entry still at T+1.
    """

    columns = list(feature_columns)
    frame = samples.copy()
    if variant == "raw":
        return frame
    if variant not in CROSS_SECTIONAL_VARIANTS:
        raise ValueError("Unknown pre-preprocessor variant: {}".format(variant))
    grouped = frame.groupby("date", sort=False)[columns]
    if variant == "demean":
        frame[columns] = frame[columns] - grouped.transform("mean")
    elif variant == "zscore":
        scale = grouped.transform("std")
        frame[columns] = (frame[columns] - grouped.transform("mean")) / scale.where(scale > 0)
    else:
        frame[columns] = grouped.rank(pct=True)
    return frame


def demean_within_date(values: np.ndarray, dates) -> np.ndarray:
    """Subtract each date's cross-sectional mean from an already-built design matrix."""

    matrix = pd.DataFrame(np.asarray(values, dtype=float))
    key = pd.Series(pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy())
    means = matrix.groupby(key, sort=False).transform("mean")
    return (matrix - means).to_numpy(dtype=float)


def build_design(
    samples: pd.DataFrame, feature_columns: Sequence[str], preprocessor: Mapping[str, object], variant: str
) -> np.ndarray:
    """The exact matrix a variant hands to Ridge, for fitting or for scoring."""

    if variant == "demean_post":
        return demean_within_date(transform_features(samples, preprocessor), samples["date"].to_numpy())
    represented = apply_representation(samples, feature_columns, variant)
    return transform_features(represented, preprocessor)


def fit_arm(
    samples: pd.DataFrame, feature_columns: Sequence[str], variant: str, alpha: float, horizon: int = HORIZON
) -> Dict[str, object]:
    """Refit the production two-head Ridge on one representation of the panel."""

    if variant == "demean_post":
        preprocessor = fit_preprocessor(samples, feature_columns)
    else:
        preprocessor = fit_preprocessor(apply_representation(samples, feature_columns, variant), feature_columns)
    design = build_design(samples, feature_columns, preprocessor, variant)
    excess_column, mae_column = target_columns(horizon)
    excess = np.asarray(pd.to_numeric(samples[excess_column], errors="raise"), dtype=float)
    mae = np.asarray(pd.to_numeric(samples[mae_column], errors="raise"), dtype=float)
    return {
        "variant": variant,
        "feature_columns": list(feature_columns),
        "preprocessor": preprocessor,
        "excess_model": Ridge(alpha=float(alpha)).fit(design, excess),
        "mae_model": Ridge(alpha=float(alpha)).fit(design, mae),
    }


def score_arm(
    arm: Mapping[str, object], samples: pd.DataFrame, risk_penalty: float
) -> Dict[str, np.ndarray]:
    """Predicted excess, predicted MAE and the production rank score of one arm."""

    design = build_design(samples, arm["feature_columns"], arm["preprocessor"], arm["variant"])
    predicted_excess = np.asarray(arm["excess_model"].predict(design), dtype=float)
    predicted_mae = np.asarray(arm["mae_model"].predict(design), dtype=float)
    return {
        "predicted_excess": predicted_excess,
        "predicted_mae": predicted_mae,
        "entry_score": decision_score(predicted_excess, predicted_mae, float(risk_penalty)),
    }


def daily_ic_series(dates, actual: np.ndarray, predicted: np.ndarray) -> pd.Series:
    """Per-date cross-sectional Rank IC, indexed by date so arms can be paired.

    Days with fewer than two usable rows, or with either side constant, are
    dropped -- the same convention as ``feature_study_core.daily_rank_ics``, but
    keeping the index because the screen compares two arms **day by day**; two
    overlapping IC means are not independent samples.
    """

    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(),
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    }).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    usable = frame.groupby("date", sort=False)
    keep = (usable["actual"].transform("nunique") >= 2) & (usable["predicted"].transform("nunique") >= 2)
    frame = frame[keep]
    if frame.empty:
        return pd.Series(dtype=float)
    ics = frame.groupby("date", sort=False).apply(
        lambda group: float(group["actual"].rank().corr(group["predicted"].rank())), include_groups=False
    )
    return ics.dropna()


def paired_ic_delta(reference: pd.Series, candidate: pd.Series) -> Dict[str, float]:
    """Mean and naive t of the per-day IC difference on the days both arms cover.

    The t-statistic is deliberately naive: consecutive 10-day-horizon ICs overlap
    and are strongly autocorrelated, so it is an upper bound on significance.  It
    is reported next to the half-window split rather than used alone.
    """

    paired = pd.concat({"reference": reference, "candidate": candidate}, axis=1).dropna()
    if len(paired) < 2:
        return {"days": int(len(paired)), "mean_delta": np.nan, "t_stat": np.nan}
    delta = (paired["candidate"] - paired["reference"]).to_numpy(dtype=float)
    std = float(np.std(delta, ddof=1))
    return {
        "days": int(len(paired)),
        "mean_delta": float(np.mean(delta)),
        "t_stat": float(np.mean(delta) / (std / np.sqrt(len(delta)))) if std > 0 else np.nan,
    }


def half_window_deltas(reference: pd.Series, candidate: pd.Series) -> Dict[str, float]:
    """The same paired delta computed independently on each half of the window."""

    paired = pd.concat({"reference": reference, "candidate": candidate}, axis=1).dropna()
    if len(paired) < 4:
        return {"first_half_delta": np.nan, "second_half_delta": np.nan}
    cutoff = paired.index.to_series().sort_values().iloc[len(paired) // 2]
    return {
        "first_half_delta": float((paired.loc[paired.index < cutoff, "candidate"]
                                   - paired.loc[paired.index < cutoff, "reference"]).mean()),
        "second_half_delta": float((paired.loc[paired.index >= cutoff, "candidate"]
                                    - paired.loc[paired.index >= cutoff, "reference"]).mean()),
    }


def top_k_overlap(dates, reference: np.ndarray, candidate: np.ndarray, top_k: float = TOP_K) -> float:
    """Mean fraction of the daily top-k basket the two arms agree on.

    Overlap is what makes the backtest the same or different: the engine buys the
    top ``max_positions`` names of ``rank_score``, so an identical basket every
    day means an identical portfolio regardless of any IC difference.
    """

    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(),
        "reference": np.asarray(reference, dtype=float),
        "candidate": np.asarray(candidate, dtype=float),
    }).dropna()
    if frame.empty:
        return np.nan
    shares = []
    for _, group in frame.groupby("date", sort=False):
        size = max(1, int(np.ceil(len(group) * float(top_k))))
        left = set(group.nlargest(size, "reference").index)
        right = set(group.nlargest(size, "candidate").index)
        shares.append(len(left & right) / float(size))
    return float(np.mean(shares)) if shares else np.nan


def score_rank_correlation(dates, reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean per-date Spearman correlation between two arms' scores."""

    ics = daily_ic_series(dates, np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float))
    return float(ics.mean()) if len(ics) else np.nan


def screen_decision(
    mean_delta: float,
    first_half_delta: float,
    second_half_delta: float,
    overlap: float,
    min_delta: float = MIN_IC_DELTA,
    min_half_delta: float = MIN_HALF_DELTA,
    min_basket_disagreement: float = MIN_BASKET_DISAGREEMENT,
) -> Dict[str, object]:
    """The pre-registered go/no-go for spending step 1's compute on one variant."""

    checks = {
        "ic_delta": bool(np.isfinite(mean_delta) and mean_delta >= float(min_delta)),
        "both_halves": bool(
            np.isfinite(first_half_delta) and np.isfinite(second_half_delta)
            and first_half_delta >= float(min_half_delta) and second_half_delta >= float(min_half_delta)
        ),
        # A variant whose basket never moves cannot change the portfolio, so
        # "advance" would buy a guaranteed no-op.
        "basket_moves": bool(np.isfinite(overlap) and (1.0 - overlap) >= float(min_basket_disagreement)),
    }
    return {"advances": bool(all(checks.values())), "checks": checks}


def check_fidelity(
    arm: Mapping[str, object],
    samples: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    stored_path: Path,
    horizon: int = HORIZON,
) -> Dict[str, float]:
    """Compare the raw arm's test-split scores against the published predictions.

    This is the only check that proves the refit reproduces the production
    pipeline row-for-row; if it fails, every number the screen reports is about a
    different model than the one that is live.
    """

    excess_column, mae_column = target_columns(horizon)
    columns = ["date", "symbol", "predicted_excess_return_{}d".format(horizon),
               "predicted_mae_{}d".format(horizon), "entry_score"]
    stored = pd.read_csv(stored_path, usecols=columns, dtype={"symbol": str})
    stored["date"] = pd.to_datetime(stored["date"], errors="coerce")
    mine = pd.DataFrame({
        "date": pd.to_datetime(samples["date"].to_numpy()),
        "symbol": np.asarray(samples["symbol"]),
        "predicted_excess_return_{}d".format(horizon): scores["predicted_excess"],
        "predicted_mae_{}d".format(horizon): scores["predicted_mae"],
        "entry_score": scores["entry_score"],
    })
    merged = mine.merge(stored, on=["date", "symbol"], suffixes=("_mine", "_stored"), how="inner")
    result = {"overlap_rows": int(len(merged)), "scored_rows": int(len(mine)), "stored_rows": int(len(stored))}
    for column in ("predicted_excess_return_{}d".format(horizon), "predicted_mae_{}d".format(horizon), "entry_score"):
        difference = np.abs(merged[column + "_mine"].to_numpy(dtype=float)
                            - merged[column + "_stored"].to_numpy(dtype=float))
        result["max_abs_diff_" + column] = float(np.nanmax(difference)) if len(difference) else np.nan
    return result


def screen_pool(
    dataset_dir: Path,
    pool: str,
    model_id: str,
    dataset_id: str,
    fidelity: bool = True,
) -> Dict[str, object]:
    """Fit every variant for one pool and measure it against the raw arm."""

    settings = load_model_settings(dataset_dir, model_id)
    features = settings["feature_columns"]
    alpha = settings["alpha"]
    risk_penalty = settings["risk_penalty"]
    # The manifest is the authority on which panel this model was fitted from;
    # a dataset id derived from the pool's name could drift from it silently.
    dataset_id = settings["dataset_id"] or dataset_id

    train = load_outcome_split(dataset_dir, dataset_id, "train", features, HORIZON)
    validation = load_outcome_split(dataset_dir, dataset_id, "validation", features, HORIZON)
    excess_column, _ = target_columns(HORIZON)
    actual_validation = np.asarray(pd.to_numeric(validation[excess_column], errors="raise"), dtype=float)

    arms = {}
    rows = []
    for variant in VARIANTS:
        arm = fit_arm(train, features, variant, alpha)
        arms[variant] = arm
        train_scores = score_arm(arm, train, risk_penalty)
        validation_scores = score_arm(arm, validation, risk_penalty)
        arm["train_scores"] = train_scores
        arm["validation_scores"] = validation_scores
        rows.append({
            "pool": pool,
            "variant": variant,
            "train_ic": float(daily_ic_series(train["date"], np.asarray(
                pd.to_numeric(train[excess_column], errors="raise"), dtype=float),
                train_scores["entry_score"]).mean()),
            "validation_ic": float(daily_ic_series(validation["date"], actual_validation,
                                                   validation_scores["entry_score"]).mean()),
            "validation_ic_excess_head": float(daily_ic_series(validation["date"], actual_validation,
                                                               validation_scores["predicted_excess"]).mean()),
        })

    table = pd.DataFrame(rows).set_index("variant")
    reference_ic = daily_ic_series(validation["date"], actual_validation,
                                   arms["raw"]["validation_scores"]["entry_score"])
    summary = {"pool": pool, "model_id": model_id, "dataset_id": dataset_id,
               "feature_count": len(features), "alpha": alpha, "risk_penalty": risk_penalty,
               "rows": {"train": int(len(train)), "validation": int(len(validation))},
               "validation_dates": int(validation["date"].nunique()),
               "variants": {}}

    for variant in VARIANTS:
        if variant == "raw":
            continue
        candidate_ic = daily_ic_series(validation["date"], actual_validation,
                                       arms[variant]["validation_scores"]["entry_score"])
        paired = paired_ic_delta(reference_ic, candidate_ic)
        halves = half_window_deltas(reference_ic, candidate_ic)
        overlap = top_k_overlap(validation["date"], arms["raw"]["validation_scores"]["entry_score"],
                                arms[variant]["validation_scores"]["entry_score"], TOP_K)
        table.loc[variant, "ic_delta_vs_raw"] = paired["mean_delta"]
        table.loc[variant, "paired_t"] = paired["t_stat"]
        table.loc[variant, "first_half_delta"] = halves["first_half_delta"]
        table.loc[variant, "second_half_delta"] = halves["second_half_delta"]
        table.loc[variant, "top_k_overlap_vs_raw"] = overlap
        table.loc[variant, "score_rank_corr_vs_raw"] = score_rank_correlation(
            validation["date"], arms["raw"]["validation_scores"]["entry_score"],
            arms[variant]["validation_scores"]["entry_score"])
        decision = screen_decision(paired["mean_delta"], halves["first_half_delta"],
                                   halves["second_half_delta"], overlap)
        table.loc[variant, "advances"] = decision["advances"]
        summary["variants"][variant] = {
            "mean_ic_delta_vs_raw": paired["mean_delta"],
            "paired_t_stat": paired["t_stat"],
            "paired_days": paired["days"],
            **halves,
            "top_k_overlap_vs_raw": overlap,
            "score_rank_corr_vs_raw": table.loc[variant, "score_rank_corr_vs_raw"],
            "decision": decision,
            "preprocessor_scale_ratio": _scale_ratio(arms["raw"], arms[variant]),
        }

    fidelity_result = None
    if fidelity:
        test = load_outcome_split(dataset_dir, dataset_id, "test", features, HORIZON)
        test_scores = score_arm(arms["raw"], test, risk_penalty)
        stored = Path(dataset_dir) / "trade_outcome_test_predictions_{}.csv".format(model_id)
        fidelity_result = check_fidelity(arms["raw"], test, test_scores, stored) if stored.exists() else {
            "error": "stored predictions missing: {}".format(stored)
        }
        summary["fidelity"] = fidelity_result
        summary["test_rows"] = int(len(test))

    return {"table": table, "summary": summary}


def _scale_ratio(reference: Mapping[str, object], candidate: Mapping[str, object]) -> float:
    """How much the production standardizer's scale moved, arm against raw.

    ``demean`` rescales each column, and Ridge's penalty is scale-dependent, so
    the median ratio is the quickest read on how far the arm drifted from a pure
    re-representation into a re-parameterisation of alpha.
    """

    left = np.asarray(reference["preprocessor"]["scales"], dtype=float)
    right = np.asarray(candidate["preprocessor"]["scales"], dtype=float)
    ratio = right / left
    return float(np.median(ratio[np.isfinite(ratio)]))


def print_report(result: Mapping[str, object]) -> None:
    summary = result["summary"]
    table = result["table"]
    print("=" * 100)
    print("{}  model={}  features={}  alpha={:g}  risk_penalty={:g}  rows train/val={}/{}".format(
        summary["pool"], summary["model_id"], summary["feature_count"], summary["alpha"],
        summary["risk_penalty"], summary["rows"]["train"], summary["rows"]["validation"]))
    print(table.to_string(float_format=lambda value: "{:.4f}".format(value)))
    print("screen rule: |ic_delta| >= {:.3f} AND both halves >= {:.3f} AND basket disagreement >= {:.2f}".format(
        MIN_IC_DELTA, MIN_HALF_DELTA, MIN_BASKET_DISAGREEMENT))
    for variant, payload in summary["variants"].items():
        print("  {:<12s} advances={:<5s} checks={} scale_ratio={:.3f}".format(
            variant, str(payload["decision"]["advances"]), payload["decision"]["checks"],
            payload["preprocessor_scale_ratio"]))
    if summary.get("fidelity") is not None:
        print("  fidelity vs published test predictions: {}".format(json.dumps(summary["fidelity"], default=str)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--pools", nargs="+", default=sorted(POOLS), choices=sorted(POOLS))
    parser.add_argument("--skip-fidelity-check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pool in args.pools:
        settings = POOLS[pool]
        result = screen_pool(dataset_dir, pool, settings["model_id"], None,
                             fidelity=not args.skip_fidelity_check)
        print_report(result)
        result["table"].reset_index().to_csv(
            output_dir / "cross_sectional_screen_{}_variants.csv".format(pool), index=False)
        (output_dir / "cross_sectional_screen_{}_summary.json".format(pool)).write_text(
            json.dumps(result["summary"], indent=2, default=str))
        print("  written: {}".format(output_dir / "cross_sectional_screen_{}_summary.json".format(pool)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
