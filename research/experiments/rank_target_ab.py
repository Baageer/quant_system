"""E4: does a cross-sectional rank objective beat MSE on the same Ridge?

The production model fits two Ridge heads by MSE on the **raw** T+1 excess return
and the raw T+1 MAE, but nothing downstream ever consumes their level: the
strategy reads ``entry_score`` through ``select_top_k``, which only uses its
**order**.  E4 asks whether optimising the rank directly -- per-date percentile
mapped to a normal quantile -- trades better.

**The plan document's one-line design ("leave Ridge alone, swap only y") cannot
be run literally, and running it literally would answer a different question.**
``y``'s
units propagate into ``entry_score`` in two places, and both change silently:

* **The combine is reweighted into a mae-head ablation.**  The published score is
  ``pred_excess - rp * max(-pred_mae, 0)``, a sum of two quantities in *return*
  units.  Measured on the published HS300 test scores, the per-date cross-sectional
  standard deviations are 0.00425 (excess) and 0.00280 (the mae term), i.e. a
  70:30 variance split (ZZ1000: 77:23).  A rank target puts the excess head on a
  ~N(0, 1) scale while the mae head stays at 0.0028, so the same formula silently
  becomes a 99.99:0.01 split -- the mae head is deleted and the arm *is* E2's
  ``A_excess``, not E4.
* **The regularisation is weakened ~180x.**  ``alpha`` acts on ``y`` in its own
  units.  Train ``var(future_excess_return_10d)`` is 5.47e-3 (HS300) / 8.47e-3
  (ZZ1000) while the rank target's variance is ~0.99, so the published
  ``alpha = 100`` is equivalent to ~18,293 / ~11,813 on a unit-variance target.
  Fitting the rank target at ``alpha = 100`` would under-regularise it, and any
  loss would be attributable to the shrinkage rather than to the objective.

Both are repaired **before any result exists**, by holding units fixed:

* ``score = s_e * pred_excess - rp * max(-s_m * pred_mae, 0)`` where ``s`` is the
  ratio of that head's *train-window* mean per-date cross-sectional standard
  deviation (MSE head / transformed head).  ``s > 0`` is a scalar, so the head's
  own ordering is untouched and only its weight in the combine is restored.
  ``s == 1`` reduces to the published formula, so the MSE arm stays bit-identical
  to the published ``entry_score``.
* ``alpha_rank = alpha_mse / var(y_mse_train)``, the exact Ridge equivalence
  ``Ridge(a) on y == Ridge(a/c^2) on y/c`` applied to the target's scale.

**Provenance is pinned by an anchor, not assumed.**  ``retrain_anchor_check``
refits the MSE arm on the train split and diffs it against the disk
``entry_score``; it is the same gate E1 uses, extended from "the published model
reproduces the disk score" to "the *retraining path* does".  Measured max |delta|
= 1.0e-16 on both pools with Spearman 1.000000.  If it fails, every number below
is about a different cross-section and the run is void.

Arms:

===============  ==================  ==================  =============
arm              excess head target  mae head target     combine
===============  ==================  ==================  =============
``A_published``  (disk column)       (disk column)       (as published)
``B_mse``        MSE raw             MSE raw             s = 1 (control)
``C_rank_excess`` rank                MSE raw             s_m = 1
``D_rank_both``  rank                rank                both
===============  ==================  ==================  =============

``B_mse`` is the identity case of the scaling rule and must equal
``A_published`` to floating point; it is what proves the *retraining* introduces
nothing.  ``D_rank_both`` is needed because E2 measured the mae head's rank IC
(0.23) an order of magnitude above the excess head's (0.039) -- an arm that only
re-targets the excess head could leave the basket almost unchanged and answer
nothing.

Pre-registered criteria (``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md``
§3 E4), evaluated on the **paired weekday family** on both windows:

0. **Anchor (hard gate).**  Retrained ``B_mse`` vs disk ``entry_score``,
   max |delta| <= 1e-12.  Failure aborts; nothing else is read.
1. Rank IC up on both windows -- **necessary, not sufficient** (the plan's own
   wording), because the target is now exactly what the metric measures.
2. Test paired-phase-mean excess over ``B_mse`` >= **+2pp**.
3. ``phase_excess_min > 0`` across the five weekdays -- the B1 phase discipline.
4. Same sign on validation and test (E1's verdict turned on exactly this).
5. Null percentile >= 95% -- **declared in advance to be non-discriminating**:
   E5b measured the *baseline* arm itself sitting at the 100th percentile, so
   this criterion passes for any arm.  It is reported with the baseline's own
   percentile beside it so the reader can see that.

The four gates of ``select_configuration`` are **reported but are not criteria**
here (the experiment is a model-side change, and the fold gate is known closed at
its configured default).  Reporting them anyway keeps the omission explicit
rather than silent.

**No tuning.**  Alpha comes from the scale rule, ``risk_penalty`` from the
published manifest, ``top_k`` from ``published_top_k``; nothing is re-picked on
either window.  If the result is negative, re-selecting alpha is a *separate*
pre-registration.

Example::

    python -m research.experiments.rank_target_ab
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from research.experiments.cross_sectional_representation_backtest import (
    FOLD_STEP_MONTHS,
    WEEKDAYS,
    phase_mean_equity,
)
from research.experiments.model_form_reachability import (
    anchor_check,
    load_published_model,
)
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
    target_columns,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    published_top_k,
)

BASELINE_ARM = "B_mse"
#: The incumbent: the disk ``entry_score`` verbatim, never recomputed, so it stays
#: bit-identical to the published numbers every other document in this repo cites.
PUBLISHED_ARM = "A_published"
ARMS = (PUBLISHED_ARM, "B_mse", "C_rank_excess", "D_rank_both")
#: ``arm -> (excess head target, mae head target)``.  ``B_mse`` is ``("mse", "mse")``
#: and is the control; the published arm is not fitted at all.
ARM_TARGETS = {
    "B_mse": ("mse", "mse"),
    "C_rank_excess": ("rank", "mse"),
    "D_rank_both": ("rank", "rank"),
}
#: The re-targeted arms, i.e. the ones criteria 1-5 are evaluated for.
CANDIDATE_ARMS = ("C_rank_excess", "D_rank_both")

HORIZON = 10
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
MIN_EXCESS_IMPROVEMENT = 0.02
MIN_PERCENTILE = 0.95
ANCHOR_TOLERANCE = 1e-12
#: Alpha grid for the label-level sensitivity, as multiples of the scale-consistent
#: value.  Reported, never adopted: the pre-registered alpha is the scale-consistent
#: one alone.
ALPHA_MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)

#: ``dataset_id`` is read off the published manifests: it is what makes
#: ``load_outcome_split`` return the same rows the published model was fitted on.
#: ``null_tag`` names that pool's published 200-draw test null.
POOLS = {
    "hs300": {
        "model_id": "hs300_pit_outcome",
        "dataset_id": "excess_hs300_pit_v1",
        "null_tag": "pit",
    },
    "zz1000": {
        "model_id": "zz1000_pit_outcome",
        "dataset_id": "excess_zz1000_pit_v1",
        "null_tag": "zz1000_pit",
    },
}


def cross_sectional_normal_rank(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Per-date rank -> (0, 1) -> normal quantile.

    ``(rank - 0.5) / n`` is used rather than ``rank / n`` so the transform is
    finite at both ends: ``rank == n`` would map to ``ppf(1) == +inf`` and a
    single such row would make the whole Ridge fit non-finite.  ``method="average"``
    gives tied labels the same rank, which is what the underlying statistic means
    -- tied future outcomes carry no ordering information.

    Label-free: it reads only the training targets and their dates, so no
    information crosses from the evaluation windows into the fit.
    """

    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    size = frame.groupby("date")["value"].transform("size").to_numpy(dtype=float)
    rank = frame.groupby("date")["value"].rank(method="average").to_numpy(dtype=float)
    quantile = (rank - 0.5) / size
    if not np.isfinite(quantile).all():
        raise ValueError("rank transform produced non-finite quantiles")
    return np.asarray(norm.ppf(quantile), dtype=float)


def mean_cross_sectional_std(values: np.ndarray, dates: pd.Series) -> float:
    """Mean over dates of the within-date standard deviation of ``values``.

    This is the statistic that governs how much a head moves the *score*: the
    strategy compares names inside one date, so a head's influence is its
    within-date spread, not its pooled spread.
    """

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": np.asarray(values, dtype=float)})
    per_date = frame.groupby("date")["value"].std(ddof=1).dropna()
    if per_date.empty:
        raise ValueError("no date has a usable cross-sectional standard deviation")
    return float(per_date.mean())


def target_scale_ratio(base: np.ndarray, candidate: np.ndarray, dates: pd.Series) -> float:
    """``s`` such that ``s * candidate`` has ``base``'s within-date spread.

    ``s > 0`` is a positive scalar, so it cannot reorder the head's own
    predictions -- it only restores the head's weight in the combine after the
    target change moved it to a different unit.
    """

    base_std = mean_cross_sectional_std(base, dates)
    candidate_std = mean_cross_sectional_std(candidate, dates)
    if candidate_std <= 0:
        raise ValueError("candidate head has no cross-sectional spread to scale")
    return float(base_std / candidate_std)


def _fit_heads(
    train_samples: pd.DataFrame,
    preprocessor: Mapping[str, object],
    horizon: int,
    alpha_mse: float,
) -> Dict[str, object]:
    """Fit every (head, target) pair the arms need, plus each arm's scale factors.

    Both targets are built from the train split alone.  ``alpha_rank`` is the
    Ridge equivalence ``Ridge(a) on y == Ridge(a / c**2) on y / c`` evaluated at
    ``c = std(y_mse_train)``, which is what makes "only the target changed" true
    in standardised units rather than only in name.
    """

    excess_column, mae_column = target_columns(horizon)
    features = transform_features(train_samples, preprocessor)
    dates = train_samples["date"]
    raw_excess = np.asarray(pd.to_numeric(train_samples[excess_column], errors="raise"), dtype=float)
    raw_mae = np.asarray(pd.to_numeric(train_samples[mae_column], errors="raise"), dtype=float)

    var_excess = float(np.var(raw_excess))
    if var_excess <= 0:
        raise ValueError("excess target has no variance; the scale rule is undefined")
    var_mae = float(np.var(raw_mae))
    if var_mae <= 0:
        raise ValueError("mae target has no variance; the scale rule is undefined")

    targets = {
        ("excess", "mse"): (raw_excess, float(alpha_mse)),
        ("excess", "rank"): (cross_sectional_normal_rank(raw_excess, dates), float(alpha_mse) / var_excess),
        ("mae", "mse"): (raw_mae, float(alpha_mse)),
        ("mae", "rank"): (cross_sectional_normal_rank(raw_mae, dates), float(alpha_mse) / var_mae),
    }
    models = {
        key: Ridge(alpha=alpha).fit(features, target)
        for key, (target, alpha) in targets.items()
    }

    # ``s`` is measured on the train window only.  The MSE head is the reference
    # because that is the unit the published score -- and therefore the published
    # 70:30 split -- is expressed in.
    scale: Dict[str, Dict[str, float]] = {"excess": {}, "mae": {}}
    fitted = {
        (head, kind): np.asarray(models[(head, kind)].predict(features), dtype=float)
        for head in ("excess", "mae")
        for kind in ("mse", "rank")
    }
    for head in ("excess", "mae"):
        scale[head]["mse"] = 1.0
        scale[head]["rank"] = target_scale_ratio(fitted[(head, "mse")], fitted[(head, "rank")], dates)

    return {
        "models": models,
        "scale": scale,
        "alpha_mse": float(alpha_mse),
        "alpha_rank_excess": float(alpha_mse) / var_excess,
        "alpha_rank_mae": float(alpha_mse) / var_mae,
        "var_excess_train": var_excess,
        "var_mae_train": var_mae,
        "feature_count": int(features.shape[1]),
        "train_rows": int(len(train_samples)),
    }


def arm_predictions(heads: Mapping[str, object], features: np.ndarray, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(scaled excess, scaled mae)`` for one fitted arm.

    The scaling is applied *here*, before the combine, so the pair returned is
    what the score is actually built from -- passing raw head output into
    ``evaluate_outcome_predictions`` would recompute a *different* score
    internally and report label metrics for an arm that was never run.
    """

    if arm == PUBLISHED_ARM:
        raise ValueError("the published arm is read off disk, not predicted here")
    excess_kind, mae_kind = ARM_TARGETS[arm]
    scale = heads["scale"]
    excess = scale["excess"][excess_kind] * np.asarray(
        heads["models"][("excess", excess_kind)].predict(features), dtype=float)
    mae = scale["mae"][mae_kind] * np.asarray(
        heads["models"][("mae", mae_kind)].predict(features), dtype=float)
    return excess, mae


def combine(excess: np.ndarray, mae: np.ndarray, risk_penalty: float) -> np.ndarray:
    """The published combine formula, unchanged.

    ``max(-s * mae, 0) == s * max(-mae, 0)`` for ``s > 0``, but the scaling is
    kept outside the ``max`` so the arithmetic matches the published expression
    term for term when ``s == 1``.
    """

    return np.asarray(excess, dtype=float) - float(risk_penalty) * np.maximum(-np.asarray(mae, dtype=float), 0.0)


def paired_phase_excess(
    phase_rows: pd.DataFrame,
    baseline_arm: str,
    candidate_arm: str,
    value_column: str = "total_return",
) -> pd.Series:
    """Candidate minus baseline, **per weekday**, indexed by weekday.

    The pairing is the whole point and it is not the same as subtracting the two
    arms' family averages.  Those agree only on a balanced family; if a weekday is
    missing for one arm, the difference-of-means quietly compares a 4-day average
    against a 5-day one, and the phase-spread criterion is then computed on a
    number that is part of the answer.  Subtracting first also keeps each
    weekday's *own* weather in the comparison, which is what criterion 3 asks
    about.
    """

    frame = phase_rows.loc[phase_rows["arm"].isin((baseline_arm, candidate_arm)),
                           ["weekday", "arm", value_column]]
    wide = frame.pivot_table(index="weekday", columns="arm", values=value_column, aggfunc="first")
    for arm in (baseline_arm, candidate_arm):
        if arm not in wide.columns:
            raise ValueError("no phase rows for arm {!r}".format(arm))
    paired = (wide[candidate_arm] - wide[baseline_arm]).dropna().sort_index()
    if paired.empty:
        raise ValueError("no weekday is present for both arms")
    return paired


def attach_published_score(
    dataset_dir: Path,
    model_id: str,
    samples: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Attach the disk ``entry_score`` and the disk head outputs to ``samples``.

    Joined on ``(date, symbol)`` with ``validate="one_to_one"``: a duplicated key
    would fan the frame out and every downstream row count would be wrong while
    still looking plausible.  ``anchor_check`` has already proved the disk file
    and the loaded split are the same cross-section, so a missing row here is a
    genuine defect rather than a legitimate gap.
    """

    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    disk = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    disk["date"] = pd.to_datetime(disk["date"], errors="coerce")
    columns = {
        SCORE_COLUMN: PUBLISHED_ARM,
        "predicted_excess_return_10d": "{}_predicted_excess".format(PUBLISHED_ARM),
        "predicted_mae_10d": "{}_predicted_mae".format(PUBLISHED_ARM),
    }
    missing = set(columns).difference(disk.columns)
    if missing:
        raise ValueError("{} missing columns: {}".format(path.name, sorted(missing)))
    disk = disk.loc[:, ["date", "symbol"] + list(columns)].rename(columns=columns)
    merged = samples.merge(disk, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(samples):
        raise ValueError("published score join changed the row count: {} -> {}".format(
            len(samples), len(merged)))
    if merged[PUBLISHED_ARM].isnull().any():
        raise ValueError("published score join left {} rows unmatched".format(
            int(merged[PUBLISHED_ARM].isnull().sum())))
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_arm_scores(
    split_samples: pd.DataFrame,
    preprocessor: Mapping[str, object],
    heads: Mapping[str, object],
    horizon: int,
    risk_penalty: float,
) -> pd.DataFrame:
    """One score column per arm plus that arm's **scaled** head outputs.

    Every arm is scored on the same rows with the same preprocessor; the only
    differences are the target each head was fitted to and the scale factor that
    restores its weight.  The head columns are carried beside the score so
    ``evaluate_outcome_predictions`` recomputes the arm's own score internally
    instead of the published one.
    """

    excess_column, mae_column = target_columns(horizon)
    features = transform_features(split_samples, preprocessor)
    frame = split_samples.loc[:, ["date", "symbol", excess_column, mae_column]].copy()
    for arm in ARMS:
        if arm == PUBLISHED_ARM:
            continue
        excess, mae = arm_predictions(heads, features, arm)
        frame[arm] = combine(excess, mae, risk_penalty)
        frame["{}_predicted_excess".format(arm)] = excess
        frame["{}_predicted_mae".format(arm)] = mae
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    # ``evaluate_outcome_predictions`` is handed this frame positionally, so a
    # reordering here would silently score one row's prediction against another
    # row's label.  The loader sorts by (date, symbol) and so does this function,
    # so the check is cheap and the failure mode it guards is not.
    source = split_samples.loc[:, ["date", "symbol"]].sort_values(["date", "symbol"]).reset_index(drop=True)
    if not frame.loc[:, ["date", "symbol"]].equals(source):
        raise ValueError("arm scores are not row-aligned with the samples they were built from")
    return frame


def retrain_anchor_check(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    model: Mapping[str, object],
    alpha_mse: float,
    risk_penalty: float,
    dataset_dir: Path,
    model_id: str,
    tolerance: float = ANCHOR_TOLERANCE,
) -> float:
    """Refit the MSE arm and diff it against the disk ``entry_score``.

    E1's ``anchor_check`` proves the *published pickle* reproduces the disk score.
    This proves the **retraining path** does, which is the stronger statement E4
    needs: every candidate arm is produced by fitting on the train split here, so
    if the MSE arm of that same path did not land on the published score, a
    difference between arms could be the training pipeline rather than the target.
    """

    excess_column, mae_column = target_columns(HORIZON)
    features = transform_features(train_samples, model["preprocessor"])
    train_excess = np.asarray(pd.to_numeric(train_samples[excess_column], errors="raise"), dtype=float)
    train_mae = np.asarray(pd.to_numeric(train_samples[mae_column], errors="raise"), dtype=float)
    excess_model = Ridge(alpha=float(alpha_mse)).fit(features, train_excess)
    mae_model = Ridge(alpha=float(alpha_mse)).fit(features, train_mae)

    samples = test_samples.copy()
    samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
    samples = samples.sort_values(["date", "symbol"]).reset_index(drop=True)
    test_features = transform_features(samples, model["preprocessor"])
    recomputed = combine(
        np.asarray(excess_model.predict(test_features), dtype=float),
        np.asarray(mae_model.predict(test_features), dtype=float),
        risk_penalty,
    )

    disk = pd.read_csv(
        dataset_dir / "trade_outcome_test_predictions_{}.csv".format(model_id),
        dtype={"symbol": str}, low_memory=False)
    disk["date"] = pd.to_datetime(disk["date"], errors="coerce")
    disk = disk.sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(disk) != len(samples):
        raise ValueError(
            "retrain anchor row count differs: disk {} vs loaded {} -- the split loader is "
            "not reading the same cross-section".format(len(disk), len(samples)))
    # Length alone would accept a same-sized cross-section of *different* names,
    # and the positional diff below would then report a mismatch that looks like a
    # training defect.  Name the real cause instead.
    if not disk.loc[:, ["date", "symbol"]].equals(samples.loc[:, ["date", "symbol"]]):
        raise ValueError("retrain anchor keys differ: the disk file is not the same cross-section")
    delta = float(np.max(np.abs(recomputed - np.asarray(disk[SCORE_COLUMN], dtype=float))))
    if not delta <= tolerance:
        raise ValueError(
            "retrain anchor FAILED: max |entry_score - retrained| = {:.3e} > {:.1e}; the "
            "training path does not reproduce the published model, so no arm here is "
            "comparable to it".format(delta, tolerance))
    return delta


def score_composition(
    scored: pd.DataFrame,
    risk_penalty: float,
    arms: Sequence[str] = ARMS,
) -> pd.DataFrame:
    """Each arm's within-date variance split between the two heads.

    This is the **direct evidence that defect 1 is repaired**.  Read literally,
    the plan's design would put ~99.99% of the score's cross-sectional variance
    on the excess head and the arm would be a mae-head ablation wearing E4's name.
    The published arm's split is ~70:30 and every candidate has to land near it.
    """

    dates = scored["date"]
    rows = []
    for arm in arms:
        excess = np.asarray(scored["{}_predicted_excess".format(arm)], dtype=float)
        mae = np.asarray(scored["{}_predicted_mae".format(arm)], dtype=float)
        var_excess = mean_cross_sectional_std(excess, dates) ** 2
        var_mae = mean_cross_sectional_std(float(risk_penalty) * np.maximum(-mae, 0.0), dates) ** 2
        total = var_excess + var_mae
        rows.append({
            "arm": arm,
            "mean_xsec_std_excess": float(np.sqrt(var_excess)),
            "mean_xsec_std_mae_term": float(np.sqrt(var_mae)),
            "excess_variance_share": float(var_excess / total) if total > 0 else np.nan,
            "mae_variance_share": float(var_mae / total) if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def alpha_sensitivity(
    train_samples: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    preprocessor: Mapping[str, object],
    horizon: int,
    risk_penalty: float,
    top_k: float,
    alpha_reference: float,
) -> pd.DataFrame:
    """Label-level rank IC across an alpha grid -- **reported, never adopted**.

    The pre-registered alpha is the scale-consistent one and nothing else.  This
    exists so a reader can see whether a label-level conclusion is an artefact of
    the shrinkage choice, without the engine ever running on a re-tuned arm.
    """

    excess_column, mae_column = target_columns(horizon)
    features = transform_features(train_samples, preprocessor)
    dates = train_samples["date"]
    raw_excess = np.asarray(pd.to_numeric(train_samples[excess_column], errors="raise"), dtype=float)
    rank_target = cross_sectional_normal_rank(raw_excess, dates)

    rows = []
    for multiplier in ALPHA_MULTIPLIERS:
        alpha = float(alpha_reference) * float(multiplier)
        excess_model = Ridge(alpha=alpha).fit(features, rank_target)
        for split, samples in splits.items():
            split_features = transform_features(samples, preprocessor)
            predicted = np.asarray(excess_model.predict(split_features), dtype=float)
            metrics = evaluate_outcome_predictions(
                samples, predicted, np.zeros(len(samples)), horizon, risk_penalty, top_k,
                split, "rank_excess_alpha_scan",
            )
            rows.append({
                "alpha": alpha,
                "alpha_multiplier": float(multiplier),
                "split": split,
                "excess_return_rank_ic": metrics["excess_return_rank_ic"],
                "selected_excess_over_universe": metrics["selected_excess_over_universe"],
            })
    return pd.DataFrame(rows)


def evaluate_pool(
    pool: str,
    dataset_dir: Path,
    output_dir: Path,
    horizon: int = HORIZON,
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_POSITIONS,
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
    published_manifest = json.loads(
        (dataset_dir / "trade_outcome_manifest_{}.json".format(model_id)).read_text(encoding="utf-8"))
    alpha_mse = float(published_manifest["selected_alpha"])
    risk_penalty = float(published_manifest["selected_risk_penalty"])
    print("\n{}  model={}  top_k={} (published)  alpha_mse={}  risk_penalty={}  features={}".format(
        pool, model_id, top_k, alpha_mse, risk_penalty, len(feature_columns)))

    samples = {
        split: load_outcome_split(dataset_dir, spec["dataset_id"], split, feature_columns, horizon)
        for split in ("train", "validation", "test")
    }
    print("  rows: train={} validation={} test={}".format(
        *(len(samples[split]) for split in ("train", "validation", "test"))))

    # --- criterion 0, hard gate ------------------------------------------------
    delta = anchor_check(model, samples["test"], horizon, risk_penalty, dataset_dir, model_id)
    retrain_delta = retrain_anchor_check(
        samples["train"], samples["test"], model, alpha_mse, risk_penalty, dataset_dir, model_id)
    print("  anchor 1: published pickle reproduces disk entry_score, max |delta| = {:.2e}".format(delta))
    print("  anchor 2: RETRAINED mse arm reproduces disk entry_score, max |delta| = {:.2e}".format(
        retrain_delta))

    heads = _fit_heads(samples["train"], model["preprocessor"], horizon, alpha_mse)
    print("  train var: excess {:.4e}  mae {:.4e}  ->  alpha_rank: excess {:.1f}  mae {:.1f}".format(
        heads["var_excess_train"], heads["var_mae_train"],
        heads["alpha_rank_excess"], heads["alpha_rank_mae"]))
    print("  scale factors (train): excess {:.6f}  mae {:.6f}".format(
        heads["scale"]["excess"]["rank"], heads["scale"]["mae"]["rank"]))

    scored, composition, metrics = {}, {}, []
    for split in ("validation", "test"):
        scored[split] = attach_published_score(
            dataset_dir, model_id,
            build_arm_scores(samples[split], model["preprocessor"], heads, horizon, risk_penalty),
            split,
        )
        composition[split] = score_composition(scored[split], risk_penalty)
        rows = []
        for arm in ARMS:
            rows.append({"split": split, "arm": arm, **evaluate_outcome_predictions(
                samples[split],
                np.asarray(scored[split]["{}_predicted_excess".format(arm)], dtype=float),
                np.asarray(scored[split]["{}_predicted_mae".format(arm)], dtype=float),
                horizon, risk_penalty, top_k, split, arm,
            )})
        metrics.append(pd.DataFrame(rows))
    metrics = pd.concat(metrics, ignore_index=True)

    print("\n  score composition (must stay near the published arm, or defect 1 is unfixed):")
    for split in ("validation", "test"):
        for _, row in composition[split].iterrows():
            print("    {:<9} {:<13} excess {:5.1%}  mae-term {:5.1%}  (std {:.5f} / {:.5f})".format(
                split, row["arm"], row["excess_variance_share"], row["mae_variance_share"],
                row["mean_xsec_std_excess"], row["mean_xsec_std_mae_term"]))

    def cell(split, arm, column):
        return float(metrics.loc[(metrics["split"] == split) & (metrics["arm"] == arm), column].iloc[0])

    # 6 dp on the rank IC: criterion 1 is decided on it, and on ZZ1000 the arms
    # differ in the fifth decimal (0.067682 vs 0.067699) -- at 4 dp they print
    # identically and the criterion looks like it cannot be read at all.
    print("\n  label cross-section ({:.0%} Top-K):".format(top_k))
    for split in ("validation", "test"):
        for arm in ARMS:
            print("    {:<9} {:<13} rank_ic={:+.6f}  excess/universe={:+.4%}".format(
                split, arm, cell(split, arm, "excess_return_rank_ic"),
                cell(split, arm, "selected_excess_over_universe")))

    sensitivity = alpha_sensitivity(
        samples["train"], {split: samples[split] for split in ("validation", "test")},
        model["preprocessor"], horizon, risk_penalty, top_k, heads["alpha_rank_excess"])
    print("\n  alpha sensitivity (label level only, reported not adopted):")
    for _, row in sensitivity.iterrows():
        print("    alpha={:>10.1f}  {:<11} rank_ic={:+.6f}  excess/universe={:+.4%}".format(
            row["alpha"], row["split"], row["excess_return_rank_ic"],
            row["selected_excess_over_universe"]))

    # --- engine runs -----------------------------------------------------------
    null_returns = load_null_total_returns(
        dataset_dir, model_id, spec["null_tag"], top_k, len(scored["test"]), max_positions=max_positions)
    print("\n  null: {} draws, mean {:+.2%} (verified top_k, rows, cap)".format(
        len(null_returns), null_returns.mean()))

    raw, slots = {}, {}
    for split in ("validation", "test"):
        raw[split], skipped = build_rank_market_data(
            scored[split], price_history_dir, ARMS[0], price_history_dirs=price_history_dirs)
        if not raw[split]:
            raise ValueError("missing price data for {} {}".format(pool, split))
        slots[split] = _score_slots(scored[split], raw[split])
        print("  {} skipped prices: {}".format(split, skipped if skipped else "none"))

    runs, phase_rows = {}, {}
    for split in ("validation", "test"):
        print("\n  running {} over {} weekdays x {} arms = {} runs...".format(
            split, len(weekdays), len(ARMS), len(weekdays) * len(ARMS)))
        rows = []
        for weekday in weekdays:
            for arm in ARMS:
                run = backtest_arm(
                    scored[split], raw[split], slots[split], arm,
                    scored[split]["date"].min(), scored[split]["date"].max(),
                    top_k, horizon, initial_capital, max_positions, config_path,
                    weekday=int(weekday), label=arm)
                runs[(split, arm, int(weekday))] = run
                rows.append({
                    "split": split, "weekday": int(weekday), "arm": arm,
                    **{key: value for key, value in run["performance"].items() if key != "strategy"},
                })
        phase_rows[split] = pd.DataFrame(rows)

    # --- criteria 2, 3, 4: paired over the weekday family ----------------------
    # Paired means "same weekday, candidate minus baseline", computed per weekday
    # *before* averaging.  Averaging each arm over the family first and then
    # subtracting would be the same number only if no weekday were missing, and it
    # would hide the per-weekday spread that criterion 3 is about.
    print("\n  paired phase excess vs {} (percentage points):".format(BASELINE_ARM))
    effects = {}
    for arm in ARMS:
        if arm == BASELINE_ARM:
            continue
        per_split = {}
        for split in ("validation", "test"):
            paired = paired_phase_excess(phase_rows[split], BASELINE_ARM, arm)
            per_split[split] = {
                "phase_mean": float(paired.mean()),
                "phase_min": float(paired.min()),
                "phase_max": float(paired.max()),
                "phases": int(len(paired)),
            }
        effects[arm] = per_split
        print("    {:<13} validation mean {:+7.2f}pp (min {:+7.2f})   test mean {:+7.2f}pp "
              "(min {:+7.2f}, max {:+7.2f})".format(
                  arm,
                  100.0 * per_split["validation"]["phase_mean"], 100.0 * per_split["validation"]["phase_min"],
                  100.0 * per_split["test"]["phase_mean"], 100.0 * per_split["test"]["phase_min"],
                  100.0 * per_split["test"]["phase_max"]))

    summary = phase_rows["test"].groupby("arm", as_index=False).mean(numeric_only=True)
    summary = summary.merge(
        phase_rows["test"].groupby("arm")["total_return"].agg(["min", "max"]).rename(
            columns={"min": "test_phase_min", "max": "test_phase_max"}).reset_index(),
        on="arm", how="left")

    print("\n  net total return (weekday family):")
    for arm in ARMS:
        test_value = float(summary.loc[summary["arm"] == arm, "total_return"].iloc[0])
        validation_value = float(
            phase_rows["validation"].groupby("arm")["total_return"].mean().loc[arm])
        print("    {:<13} validation={:>8.2%}  test mean={:>8.2%} (min {:>7.2%}, max {:>7.2%})".format(
            arm, validation_value, test_value,
            float(summary.loc[summary["arm"] == arm, "test_phase_min"].iloc[0]),
            float(summary.loc[summary["arm"] == arm, "test_phase_max"].iloc[0])))

    # Averaged over the same weekday family as the returns, for the same reason:
    # a single weekday's basket is one draw, and the pre-registered risk ("a rank
    # objective amplifies noise in the realised downside") is a claim about the
    # objective, not about one calendar strike.
    downside_rows = []
    for split in ("validation", "test"):
        for arm in ARMS:
            per_weekday = pd.DataFrame([
                basket_downside(scored[split], runs[(split, arm, int(weekday))]["selected"], horizon)
                for weekday in weekdays
            ])
            downside_rows.append({"split": split, "arm": arm, **per_weekday.mean().to_dict()})
    downside = pd.DataFrame(downside_rows)

    percentiles = {
        arm: float((null_returns <= float(summary.loc[summary["arm"] == arm, "total_return"].iloc[0])).mean())
        for arm in ARMS
    }
    print("\n  test percentile vs the {}-draw null (mean {:+.2%}):".format(
        len(null_returns), null_returns.mean()))
    for arm in ARMS:
        print("    {:<13} {:>7.1%}{}".format(
            arm, percentiles[arm],
            "   <-- the BASELINE, so this criterion cannot discriminate" if arm == BASELINE_ARM else ""))

    print("\n  basket downside, weekday-family mean (the pre-registered noise-amplification check):")
    for split in ("validation", "test"):
        for arm in ARMS:
            row = downside[(downside["split"] == split) & (downside["arm"] == arm)].iloc[0]
            print("    {:<9} {:<13} sel.mae={:+.4f}  sel.excess={:+.4f}  universe.mae={:+.4f}".format(
                split, arm, row["selected_mean_future_mae"],
                row["selected_mean_future_excess"], row["universe_mean_future_mae"]))

    # --- the four gates: reported, never a criterion here ----------------------
    folds = fold_boundaries(
        scored["validation"]["date"].min(), scored["validation"]["date"].max(), FOLD_STEP_MONTHS)
    weekday_mean_equity = {
        arm: phase_mean_equity({
            int(weekday): runs[("validation", arm, int(weekday))]["equity"] for weekday in weekdays})
        for arm in ARMS
    }
    fold_excess = fold_excess_table(
        weekday_mean_equity, folds, initial_capital, baseline_arm=BASELINE_ARM)
    phase_excess = phase_excess_table(
        phase_rows["validation"], phase_rows["validation"][phase_rows["validation"]["arm"] == BASELINE_ARM],
        key_columns=("arm",))
    grid = pd.DataFrame([{
        "arm": arm,
        "validation_total_return": float(
            phase_rows["validation"].groupby("arm")["total_return"].mean().loc[arm]),
        "validation_turnover_multiple": float(
            phase_rows["validation"].groupby("arm")["turnover_multiple"].mean().loc[arm]),
    } for arm in ARMS])
    gates = select_configuration(
        grid, baseline_key={"arm": BASELINE_ARM}, key_columns=("arm",),
        return_column="validation_total_return", turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess, phase_excess=phase_excess)
    print("\n  four gates (REPORTED ONLY -- not a criterion for E4; the fold gate is known "
          "closed at its configured default):")
    print("    {} -- {}".format(
        "NO CHANGE" if gates.row is None else gates.row["arm"], gates.reason))

    # --- verdict ---------------------------------------------------------------
    criteria, verdicts = {}, {}
    for arm in CANDIDATE_ARMS:
        rank_ic_up = (cell("validation", arm, "excess_return_rank_ic")
                      > cell("validation", BASELINE_ARM, "excess_return_rank_ic")
                      and cell("test", arm, "excess_return_rank_ic")
                      > cell("test", BASELINE_ARM, "excess_return_rank_ic"))
        effect = effects[arm]
        crit = {
            "criterion_1_rank_ic_up_both_windows": bool(rank_ic_up),
            "criterion_2_test_phase_mean_ge_2pp": bool(
                effect["test"]["phase_mean"] >= MIN_EXCESS_IMPROVEMENT),
            "criterion_3_test_phase_min_positive": bool(effect["test"]["phase_min"] > 0),
            "criterion_4_same_sign_both_windows": bool(
                effect["validation"]["phase_mean"] > 0 and effect["test"]["phase_mean"] > 0),
            "criterion_5_null_percentile_ge_95": bool(percentiles[arm] >= MIN_PERCENTILE),
        }
        criteria[arm] = crit
        adopted = all(crit[key] for key in (
            "criterion_2_test_phase_mean_ge_2pp",
            "criterion_3_test_phase_min_positive",
            "criterion_4_same_sign_both_windows",
        ))
        if adopted:
            verdicts[arm] = "candidate clears criteria 2-4 -> eligible for an adoption decision"
        elif crit["criterion_1_rank_ic_up_both_windows"]:
            verdicts[arm] = ("rank IC improves and net return does not -> IC and tradability "
                             "separate again (the plan's pre-registered negative result)")
        else:
            verdicts[arm] = "no change"

    print("\n  criteria (observed next to each, so a margin can be read off):")
    for arm in CANDIDATE_ARMS:
        effect = effects[arm]
        observed = {
            "criterion_1_rank_ic_up_both_windows": "val {:+.6f} vs {:+.6f}, test {:+.6f} vs {:+.6f}".format(
                cell("validation", arm, "excess_return_rank_ic"),
                cell("validation", BASELINE_ARM, "excess_return_rank_ic"),
                cell("test", arm, "excess_return_rank_ic"),
                cell("test", BASELINE_ARM, "excess_return_rank_ic")),
            "criterion_2_test_phase_mean_ge_2pp": "{:+.2f}pp vs +2.00pp".format(
                100.0 * effect["test"]["phase_mean"]),
            "criterion_3_test_phase_min_positive": "{:+.2f}pp".format(100.0 * effect["test"]["phase_min"]),
            "criterion_4_same_sign_both_windows": "val {:+.2f}pp, test {:+.2f}pp".format(
                100.0 * effect["validation"]["phase_mean"], 100.0 * effect["test"]["phase_mean"]),
            "criterion_5_null_percentile_ge_95": "{:.1%} vs 95.0% (baseline itself {:.1%})".format(
                percentiles[arm], percentiles[BASELINE_ARM]),
        }
        print("    {}:".format(arm))
        for key, value in criteria[arm].items():
            print("      {:<42} {:<4}  {}".format(key, "PASS" if value else "FAIL", observed[key]))
        print("      verdict: {}".format(verdicts[arm]))

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "rank_target_{}".format(pool)
    metrics.to_csv(output_dir / (prefix + "_label_metrics.csv"), index=False)
    summary.to_csv(output_dir / (prefix + "_test_metrics.csv"), index=False)
    pd.concat(phase_rows.values(), ignore_index=True).to_csv(
        output_dir / (prefix + "_phases.csv"), index=False)
    downside.to_csv(output_dir / (prefix + "_basket_downside.csv"), index=False)
    pd.concat([frame.assign(split=split) for split, frame in composition.items()], ignore_index=True).to_csv(
        output_dir / (prefix + "_score_composition.csv"), index=False)
    sensitivity.to_csv(output_dir / (prefix + "_alpha_sensitivity.csv"), index=False)
    fold_columns = ["fold_{}".format(index) for index in range(len(folds))]
    grid.merge(phase_excess[["arm", "phase_excess_min", "phases"]], on="arm", how="left").merge(
        fold_excess[["arm"] + fold_columns], on="arm", how="left").to_csv(
        output_dir / (prefix + "_validation_grid.csv"), index=False)

    report = {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "risk_penalty": risk_penalty,
        "alpha_mse": alpha_mse,
        "alpha_rank_excess": heads["alpha_rank_excess"],
        "alpha_rank_mae": heads["alpha_rank_mae"],
        "var_excess_train": heads["var_excess_train"],
        "var_mae_train": heads["var_mae_train"],
        "scale_factors": {head: dict(values) for head, values in heads["scale"].items()},
        "anchor_published_delta": delta,
        "anchor_retrain_delta": retrain_delta,
        "baseline_arm": BASELINE_ARM,
        "candidate_arms": list(CANDIDATE_ARMS),
        "arm_targets": {arm: list(ARM_TARGETS[arm]) for arm in ARM_TARGETS},
        "effects": effects,
        "criteria": criteria,
        "verdicts": verdicts,
        "test_percentile_vs_shared_null": percentiles,
        "null_draws": int(len(null_returns)),
        "null_mean_total_return": float(null_returns.mean()),
        "four_gates_reported_only": {
            "row": None if gates.row is None else str(gates.row["arm"]),
            "reason": gates.reason,
            "gates": dict(gates.gates),
        },
        "execution": {
            "horizon": int(horizon),
            "max_positions": max_positions,
            "initial_capital": initial_capital,
            "weekdays": [int(day) for day in weekdays],
            "windows": "both validation and test run the full weekday family",
            "runs_per_pool": 2 * len(weekdays) * len(ARMS),
        },
    }
    (output_dir / (prefix + "_manifest.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="E4: rank objective vs MSE")
    parser.add_argument("--pool", choices=sorted(POOLS), default=None, help="Default: both pools.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--price-history-dirs", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--weekdays", default=",".join(str(day) for day in WEEKDAYS))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    weekdays = [int(value) for value in str(args.weekdays).split(",") if value.strip()]
    dirs = [d.strip() for d in args.price_history_dirs.split(",") if d.strip()] \
        if args.price_history_dirs else None
    pools: List[str] = [args.pool] if args.pool else sorted(POOLS)
    for pool in pools:
        evaluate_pool(
            pool, Path(args.dataset_dir), Path(args.output_dir),
            horizon=args.horizon, initial_capital=args.initial_capital,
            max_positions=args.max_positions, price_history_dir=args.price_history_dir,
            config_path=args.config_path, price_history_dirs=dirs, weekdays=weekdays,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
