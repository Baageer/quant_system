"""E3: does fusing the h=5/10/20 entry scores beat h=10 alone?

The label pipeline computes outcomes at three horizons
(``label_semantics.horizons = [5, 10, 20]``) but the published score only ever
reads the h=10 pair.  E3 asks whether the other two horizons carry information
the h=10 model is throwing away.

Design (pre-registered): fit the published two-head Ridge at each horizon, put
each horizon's ``entry_score`` on a common unit with a **per-date cross-sectional
scale division**, and **average them with equal weight**:
``n(e5) + n(e10) + n(e20)``.  The weights are the pre-registered choice and are
not searched -- searching them would open a second configuration-selection
problem on top of the one the four gates already govern, which is the discipline
E1/E2/E4 were built to protect.

The plan says "per-date z-score".  It is **not** implemented as a z-score, and
fact 2 below gives the measurement that forced the change: the pipeline encodes
an absent score as the literal ``0.0`` and sorts that sentinel against real
scores, so subtracting a per-date mean silently re-ranks names whose unit is
"no score today".  Dividing by the per-date spread removes exactly the unit
difference the plan cares about and, being a positive per-date *factor*, is
provably order-preserving.  What the z-score would have cost is recorded in
fact 2 and in the archive.

Execution stays at h=10.  ``min_holding_days`` is coupled to the horizon
(``ranking_ablation_harness.backtest_arm`` passes it straight through to
``make_rank_strategy``), so "also change the holding period" would be a second
variable moving at once.  Only the score is fused.

**Four things were measured before any result existed, and all four shape the
pre-registration:**

1. **The three horizons score exactly the same rows.**  ``load_outcome_split``
   filters on the *requested* horizon's finite labels, so a horizon with a
   different eligibility pattern would silently move the cross-section and the
   fusion would be comparing baskets drawn from different universes.  Measured
   on both pools and all three splits: the ``(date, symbol)`` key sets are
   identical across horizons.  ``assert_horizon_row_alignment`` enforces it at
   runtime rather than trusting the measurement.

2. **Rescaling a single horizon is engine-neutral only if it does not move the
   missing-score sentinel -- and it is the control.**  ``select_top_k`` ranks the
   score (``method="first"``, so score *values* only break ties by row order) and
   ``_write_rank_score`` feeds ``rank_score`` into the strategy, which sorts by
   ``(-rank_score, symbol)``.  A strictly monotone map preserves both the order
   and the tie sets, so ``B_h10`` must reproduce ``A_published`` -- the disk
   column fit by the published pickle -- **exactly**, not approximately.  That
   equality is a hard gate: it proves in one shot that the retraining path
   reproduces the published model *and* that the fusion's unit-conversion step
   introduces nothing on its own.  (It also settles sum vs mean: ``n5+n10+n20``
   and its mean are the same arm up to the positive scalar 1/3.)

   **This gate fired on the first smoke run, and the reason matters.**  The
   first version *did* subtract the per-date mean.  ``build_rank_market_data``
   fills a name with no score row by ``0.0`` and the strategy sorts that sentinel
   in the same pass as real scores; on 2025-06-30 one of thirteen HS300 test
   candidates (``600745``) was in that state -- selected on 2025-06-27 and bought
   on 2025-06-30, the entry signal being a one-session shift, but absent from the
   scored frame on the entry day.  The sentinel is a hardcoded literal and is
   never itself transformed, so shifting the real scores moves them *past* it:
   the published arm bought ``600745`` and the mean-shifted arm bought ``000708``
   instead.  One slot, cascading to 106 of 818 sessions and **0.89pp** of net
   return (``A`` 0.555974 vs ``B`` 0.564887).  A positive per-date factor cannot
   do that -- it cannot move a real score across zero, and ``0 / std == 0``.
   ``cross_sectional_scale`` therefore divides by the per-date spread and does
   not recentre.

3. **Only h=10 can be anchored; h=5 and h=20 have no published artifact.**  The
   retrain anchor is what makes a model-side A/B comparable to the published
   numbers, and it exists only where a disk ``entry_score`` exists.  For h=5 and
   h=20 the substitute is weaker and is declared as such: each horizon's model
   must have a **positive rank IC against its own labels** in both windows.  That
   is a *sanity check on the pipeline*, not a criterion -- it catches a
   wrong-label-column class of bug, and it cannot certify that the horizon's
   model is any good.  Interpretive rule, fixed in advance: if any horizon's
   own-label rank IC is negative in both windows, the fusion reading is void and
   the next step is to investigate that horizon's model rather than to interpret
   the fusion.

4. **The two heads' label variances do not scale identically across horizons.**
   Measured on train: at h=20 the excess target's variance is 2.0012x the h=10
   one (HS300) while the mae target's is 1.8452x, so the *published combine's*
   head split -- the same 70:30 question E4 had to repair -- drifts by a few
   variance-share points across horizons rather than the ~3000x E4 saw.  It is
   **measured and reported, not corrected**: correcting it would mean applying a
   different combine to each horizon, which is a new per-horizon design choice
   the plan's "do not search the fusion weights" ruling rules out just as
   firmly as it rules out searching the weights themselves.  Declared
   consequence: a negative fusion result cannot be fully separated from this
   drift -- but at a few points of variance share it is a far smaller effect than
   the fusion itself would have to be to matter.

**Alpha across horizons** (the scale rule, chosen before running): ``alpha`` acts
on ``y`` in its own units, and ``Ridge(alpha) on y`` is the same fit as
``Ridge(alpha * c^2) on c*y``.  Holding the shrinkage fixed while the target's
unit changes therefore means ``alpha_h = alpha_10 * var(y_h_train) /
var(y_10_train)``, per head, so that the only thing that differs between the
three models is the horizon and not the effective regularisation.  Published
``alpha`` is 100 (HS300) / 10 (ZZ1000) at h=10; the measured ratios give

=================  ==============  ==============  ==============
pool               head            h=5             h=20
=================  ==============  ==============  ==============
HS300              excess          51.38           200.12
HS300              mae             51.94           184.52
ZZ1000             excess           5.18            19.46
ZZ1000             mae              5.18            17.65
=================  ==============  ==============  ==============

The ratio is computed from the **train** split only, so nothing crosses from the
evaluation windows into the fit.  At h=10 the ratio is exactly 1.0, which is why
``B_h10`` reproduces the published fit bit for bit.

Arms (5, fixed by the design decision):

====================  ================================================
arm                   score
====================  ================================================
``A_published``       disk ``entry_score`` (the incumbent)
``B_h10``             retrained h=10, scale-divided -- the paired baseline
``C_fused_eq``        ``n(e5) + n(e10) + n(e20)`` -- the only candidate
``D_fused_no20``      ``n(e5) + n(e10)``  -- descriptive, **not adoptable**
``E_fused_no5``       ``n(e10) + n(e20)`` -- descriptive, **not adoptable**
====================  ================================================

``D`` and ``E`` answer "which horizon drags" if the fusion is negative.  They are
declared non-adoptable in advance: with five arms, promoting whichever
leave-one-out happens to win is exactly the selection-on-the-winner failure the
gates exist to prevent.

Pre-registered criteria (``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md``
§3 E3), evaluated on the **paired weekday family** on both windows:

1. The fusion arm's net return beats the h=10 arm, clears the four gates, and the
   improvement is >= +2pp.  Reported decomposed, because the four-gate bundle is
   **structurally unreachable** at its configured default: the fold gate is known
   to be permanently closed (measured pass rate 0.0000%), so "clears the four
   gates" would make the criterion unpassable for any arm.  The adoption bar is
   therefore criteria 2-4 below, with the gates printed beside it so the
   omission is explicit rather than silent.
2. Null percentile >= 95% -- **declared in advance to be non-discriminating**:
   the baseline arm itself sits at the 100th percentile, so this passes for
   every arm.  Printed with the baseline's own percentile beside it.
3. Wins on validation only and not on test -> **UNDECIDABLE**.  This is a distinct
   verdict from "no change", not a relabelling of it: one validation window
   cannot separate a real edge from a favourable draw, and the project has been
   burned by that shape before.  A sign *flip* also lands here by this rule (the
   plan's wording is unconditional), so the test reading's sign and size are
   printed next to the verdict -- practically a flip reads as refuted, but the
   pre-registered label is what is reported.

**No tuning.**  Alpha comes from the scale rule, ``risk_penalty`` and ``top_k``
from the published manifests, the fusion weights from the pre-registration, and
the execution horizon from production.  Nothing is re-picked on either window.

Example::

    python -m research.experiments.multi_horizon_fusion_ab
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
# Reused from E4 rather than re-derived: the retrain anchor, the paired-phase
# difference and the head-variance share are one definition each, and a second
# copy in this module is a divergence waiting to happen.  ``pair`` and ``share``
# are pure functions; ``retrain_anchor_check`` is the gate E4 established and is
# fixed at the published horizon, which is the only horizon that has one.
from research.experiments.rank_target_ab import (
    attach_published_score,
    mean_cross_sectional_std,
    paired_phase_excess,
    retrain_anchor_check,
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
    _top_k_mask,
    decision_score,
    evaluate_outcome_predictions,
    load_outcome_split,
    target_columns,
)
from research.market_regime.trade_outcome_ranking_evaluation import published_top_k

PUBLISHED_ARM = "A_published"
BASELINE_ARM = "B_h10"
FUSION_ARM = "C_fused_eq"
FUSED_NO20_ARM = "D_fused_no20"
FUSED_NO5_ARM = "E_fused_no5"
#: ``B_h10`` is the scale-divided h=10 score, so it doubles as that horizon's
#: diagnostic column; the alias exists so every engine arm is a column and no
#: call site has to know that two names mean the same vector.  The other two are
#: ``N_h5`` / ``N_h20`` -- normalized, but *not* z-scores; see
#: ``cross_sectional_scale`` for why that distinction is load-bearing here.
NORMALIZED_COLUMNS = {5: "N_h5", 10: "B_h10", 20: "N_h20"}
ARMS = (PUBLISHED_ARM, BASELINE_ARM, FUSION_ARM, FUSED_NO20_ARM, FUSED_NO5_ARM)
#: The only arm whose result can be adopted.  Criteria are evaluated for the
#: others too, but their verdicts carry the ``DESCRIPTIVE`` prefix so a reader
#: cannot mistake a leave-one-out win for a candidate.
CANDIDATE_ARM = FUSION_ARM
DESCRIPTIVE_ARMS = (FUSED_NO20_ARM, FUSED_NO5_ARM)
#: ``arm -> the horizons whose normalized scores it averages``.
ARM_HORIZONS = {
    BASELINE_ARM: (10,),
    FUSION_ARM: (5, 10, 20),
    FUSED_NO20_ARM: (5, 10),
    FUSED_NO5_ARM: (10, 20),
}
#: Diagnostic-only single-horizon columns (h=10 is ``B_h10``).
DIAGNOSTIC_COLUMNS = (
    NORMALIZED_COLUMNS[5],
    NORMALIZED_COLUMNS[10],
    NORMALIZED_COLUMNS[20],
)

HORIZONS = (5, 10, 20)
#: The published model's horizon.  It is the anchor's horizon and the execution
#: horizon; ``--horizon`` exists only so the caller can see the coupling, and
#: anything else is refused rather than silently run with one of the two moved.
EXECUTION_HORIZON = 10
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
MIN_EXCESS_IMPROVEMENT = 0.02
MIN_PERCENTILE = 0.95

#: ``dataset_id`` is read off the published manifests: it is what makes
#: ``load_outcome_split`` return the rows the published model was fitted on.
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


def _column(samples: pd.DataFrame, name: str) -> np.ndarray:
    return np.asarray(pd.to_numeric(samples[name], errors="raise"), dtype=float)


def _scale_parts(values: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """Per-date size / std of ``values``, plus which rows are usable.

    A date is usable when it has at least two names and a non-zero spread.  An
    unusable date is not dropped: every row of it scales to 0.0, which is the
    honest encoding of "this date carries no cross-sectional information".
    Dropping the rows instead would change the cross-section the engine sees, and
    such a date is already a total tie for ``select_top_k`` anyway.
    """

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": np.asarray(values, dtype=float)})
    grouped = frame.groupby("date")["value"]
    frame["size"] = grouped.transform("size").to_numpy(dtype=float)
    frame["std"] = grouped.transform("std", ddof=1).to_numpy(dtype=float)
    frame["usable"] = (frame["size"] > 1) & frame["std"].notna() & (frame["std"] > 0)
    return frame


def cross_sectional_scale(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Divide by the per-date cross-sectional spread -- **scale only, no location**.

    The plan asks for a per-date z-score, and the *reason* is to make the three
    horizons commensurable before they are averaged: a 20-day excess return has
    about twice the variance of a 10-day one, so an unweighted sum of raw scores
    would be a weighted sum in disguise.  Dividing by the per-date spread removes
    exactly that unit difference, and subtracting the per-date mean removes
    something the plan never asked for -- while, in this pipeline, silently
    changing the strategy.

    The pipeline treats ``0.0`` as a *real* score.  ``build_rank_market_data``
    fills an absent score with ``0.0``, ``_write_rank_score`` does the same, and
    the strategy sorts candidates by ``(-rank_score, symbol)`` in the same pass --
    so a name that was selected yesterday and has no score row today carries that
    sentinel into the comparison.

    The decisive detail is that ``0.0`` is a **hardcoded literal that never goes
    through the transform**: it is not the transformed value of anything.  So it
    is the *real* scores that move relative to it.  A positive per-date factor
    cannot move them across zero, so the sentinel's rank position is bit-identical
    to the published arm's; a per-date mean shift moves them across it whenever
    the date's mean has the opposite sign, which is the common case.  Measured on
    the HS300 test window: on 2025-06-30 exactly one of thirteen candidates
    (``600745``) was in that state -- selected on 2025-06-27, bought on
    2025-06-30 because the entry signal is a one-session shift, but absent from
    the scored frame on the entry day.  The mean-shifted arm bought ``000708`` in
    its place: one slot, cascading to 106 of 818 sessions and **0.89pp** of net
    return.

    Scale division also preserves ties (``x == y`` implies ``x/f == y/f``), which
    matters because ``select_top_k`` breaks ties by row order and ``rank_score``
    by symbol.  That is what makes ``B_h10`` reproduce ``A_published`` bit for bit
    -- see ``assert_scaling_is_engine_neutral``, a gate rather than an assumption
    precisely because the first version of this function *did* subtract the mean
    and the gate caught it.
    """

    values = np.asarray(values, dtype=float)
    parts = _scale_parts(values, dates)
    scaled = np.zeros(len(parts), dtype=float)
    usable = parts["usable"].to_numpy(dtype=bool)
    scaled[usable] = values[usable] / parts["std"].to_numpy(dtype=float)[usable]
    if not np.isfinite(scaled).all():
        raise ValueError("cross-sectional scaling produced non-finite values")
    return scaled


def degenerate_dates(values: np.ndarray, dates: pd.Series) -> List[pd.Timestamp]:
    """The dates ``cross_sectional_scale`` maps to an all-zero block."""

    parts = _scale_parts(values, dates)
    return [pd.Timestamp(value) for value in parts.loc[~parts["usable"], "date"].unique()]


def assert_horizon_row_alignment(by_horizon: Mapping[int, pd.DataFrame], split: str) -> None:
    """Every horizon must score the same cross-section, row for row.

    ``load_outcome_split`` filters on the requested horizon's finite labels, so a
    horizon whose labels are missing on a different set of rows would move the
    cross-section underneath the fusion.  The measured answer is that they agree;
    asserting it keeps that a fact about the data rather than an assumption in
    the code, and the failure mode it guards (three arms of a fusion averaged
    over three different universes) is invisible in any output table.
    """

    reference = by_horizon[EXECUTION_HORIZON].loc[:, ["date", "symbol"]].reset_index(drop=True)
    for horizon, samples in by_horizon.items():
        keys = samples.loc[:, ["date", "symbol"]].reset_index(drop=True)
        if len(keys) != len(reference):
            raise ValueError(
                "{} rows differ across horizons: h={} has {} against h={}'s {} -- the "
                "horizons are not scoring the same cross-section".format(
                    split, horizon, len(keys), EXECUTION_HORIZON, len(reference)))
        if not keys.equals(reference):
            raise ValueError(
                "{} keys differ across horizons: h={} is not row-aligned with h={}".format(
                    split, horizon, EXECUTION_HORIZON))


def fit_horizon_heads(
    train_samples: pd.DataFrame,
    preprocessor: Mapping[str, object],
    horizon: int,
    alpha_reference: float,
    var_reference: Mapping[str, float],
) -> Dict[str, object]:
    """Fit both published heads at ``horizon`` with the scale-consistent alpha.

    ``alpha_reference`` / ``var_reference`` describe the published h=10 pair.  The
    per-head alpha is ``alpha_10 * var(y_h) / var(y_10)``, which is the Ridge
    identity ``Ridge(a) on y == Ridge(a * c^2) on c * y`` solved for a fixed
    shrinkage: scaling the target's unit by ``c`` changes its variance by
    ``c^2``, so the alpha that keeps the same effective fit scales with the
    variance.  At h=10 the ratio is exactly 1.0 and the fit is the published one.

    Both variances are read off the **train** split, so the rule has no
    look-ahead: it is a property of the labels the model is fitted to.
    """

    excess_column, mae_column = target_columns(horizon)
    features = transform_features(train_samples, preprocessor)
    y_excess = _column(train_samples, excess_column)
    y_mae = _column(train_samples, mae_column)
    var_excess = float(np.var(y_excess))
    var_mae = float(np.var(y_mae))
    if var_excess <= 0 or var_mae <= 0:
        raise ValueError("h={} has a degenerate target; the scale rule is undefined".format(horizon))
    ratio_excess = var_excess / float(var_reference["excess"])
    ratio_mae = var_mae / float(var_reference["mae"])
    alpha_excess = float(alpha_reference) * ratio_excess
    alpha_mae = float(alpha_reference) * ratio_mae
    return {
        "horizon": int(horizon),
        "excess_model": Ridge(alpha=alpha_excess).fit(features, y_excess),
        "mae_model": Ridge(alpha=alpha_mae).fit(features, y_mae),
        "alpha_excess": float(alpha_excess),
        "alpha_mae": float(alpha_mae),
        "var_excess": var_excess,
        "var_mae": var_mae,
        "ratio_excess": float(ratio_excess),
        "ratio_mae": float(ratio_mae),
        "train_rows": int(len(train_samples)),
        "feature_count": int(features.shape[1]),
    }


def head_variance_share(
    excess: np.ndarray, mae: np.ndarray, risk_penalty: float, dates: pd.Series
) -> Dict[str, float]:
    """The combine's two-head variance split, measured per date and averaged.

    The published score adds an excess term and a downside term that are both in
    *return* units, so their relative weight inside the score is a property of
    their cross-sectional spreads.  E3 does not change the combine, but the
    horizons' labels do not scale identically (measured: h=20's excess variance
    is 2.00x the h=10 one while the mae's is 1.85x), so this drifts slightly
    across horizons and the drift is reported rather than assumed away.
    """

    var_excess = mean_cross_sectional_std(excess, dates) ** 2
    var_mae = mean_cross_sectional_std(float(risk_penalty) * np.maximum(-np.asarray(mae, dtype=float), 0.0),
                                       dates) ** 2
    total = var_excess + var_mae
    return {
        "mean_xsec_std_excess": float(np.sqrt(var_excess)),
        "mean_xsec_std_mae_term": float(np.sqrt(var_mae)),
        "excess_variance_share": float(var_excess / total) if total > 0 else float("nan"),
        "mae_variance_share": float(var_mae / total) if total > 0 else float("nan"),
    }


def build_horizon_scores(
    split_samples: pd.DataFrame,
    preprocessor: Mapping[str, object],
    heads_by_horizon: Mapping[int, Mapping[str, object]],
    risk_penalty: float,
    horizon: int,
) -> pd.DataFrame:
    """Raw and normalized ``entry_score`` for every horizon, plus the fused arms.

    The raw columns are kept because the fused arms cannot be checked without
    them: the normalization's only job is a per-date change of unit, and the gate
    that it did nothing else is "``select_top_k`` on the raw h=10 column equals
    ``select_top_k`` on the normalized one", which needs both.
    """

    frame = split_samples.loc[:, ["date", "symbol"]].copy()
    for name in target_columns(horizon):
        frame[name] = _column(split_samples, name)
    features = transform_features(split_samples, preprocessor)
    for value in HORIZONS:
        heads = heads_by_horizon[value]
        predicted_excess = np.asarray(heads["excess_model"].predict(features), dtype=float)
        predicted_mae = np.asarray(heads["mae_model"].predict(features), dtype=float)
        frame["S_h{}".format(value)] = decision_score(predicted_excess, predicted_mae, risk_penalty)
        frame["h{}_predicted_excess".format(value)] = predicted_excess
        frame["h{}_predicted_mae".format(value)] = predicted_mae
        frame[NORMALIZED_COLUMNS[value]] = cross_sectional_scale(
            frame["S_h{}".format(value)], frame["date"])
    for arm, horizons in ARM_HORIZONS.items():
        frame[arm] = sum(frame[NORMALIZED_COLUMNS[value]] for value in horizons)
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    # Compared against the input *unsorted*.  Sorting both sides first would make
    # the check unfalsifiable -- two sorts of the same rows always agree -- while
    # the property the callers actually rely on is that the loader's frame comes
    # back in the same order, because ``evaluate_outcome_predictions`` is handed
    # the two positionally.
    source = split_samples.loc[:, ["date", "symbol"]].reset_index(drop=True)
    if not frame.loc[:, ["date", "symbol"]].equals(source):
        raise ValueError(
            "build_horizon_scores reordered the rows: the loader's frame is passed positionally to "
            "evaluate_outcome_predictions, so a reorder scores one row's prediction against another "
            "row's label")
    return frame


def assert_scaling_is_engine_neutral(scored: pd.DataFrame, top_k: float) -> None:
    """The raw and normalized h=10 scores must select the same basket.

    This is the gate that makes ``B_h10`` a control rather than a second
    candidate.  If it fails, the normalization is doing something other than a
    change of unit, every fused arm inherits that, and no comparison in this
    experiment means what it says.  Note the limit of what it can see:
    ``select_top_k`` is not the only consumer of the score -- the strategy also
    sorts the raw ``rank_score`` -- so this gate passing is necessary, not
    sufficient, and the return-level control in ``evaluate_pool`` is what covers
    the rest.
    """

    raw = _top_k_mask(scored["date"], scored["S_h{}".format(EXECUTION_HORIZON)].to_numpy(dtype=float), top_k)
    scaled = _top_k_mask(scored["date"], scored[BASELINE_ARM].to_numpy(dtype=float), top_k)
    published = _top_k_mask(scored["date"], scored[PUBLISHED_ARM].to_numpy(dtype=float), top_k)
    if not np.array_equal(raw, scaled):
        raise ValueError(
            "the scaling changed the h=10 basket ({} rows differ); it is not engine-neutral and "
            "no fused arm here is comparable to the published one".format(int(np.sum(raw != scaled))))
    if not np.array_equal(raw, published):
        raise ValueError(
            "the retrained h=10 basket differs from the published basket ({} rows); the training "
            "path does not reproduce the published model".format(int(np.sum(raw != published))))


def daily_rank_correlation(dates: pd.Series, left: np.ndarray, right: np.ndarray) -> float:
    """Mean over dates of the within-date Spearman correlation.

    Mirrors ``trade_outcome_model._daily_rank_ic``: dates with a constant side
    are skipped rather than counted as zero, because a constant has no ordering
    to correlate with.
    """

    frame = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "left": np.asarray(left, dtype=float),
        "right": np.asarray(right, dtype=float),
    })
    values = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 2 or group["left"].nunique() < 2 or group["right"].nunique() < 2:
            continue
        values.append(float(group["left"].rank().corr(group["right"].rank())))
    return float(np.mean(values)) if values else float("nan")


def cross_horizon_correlation(scored: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Pairwise daily rank correlation between the horizons' normalized scores.

    The diagnostic that tells a null result apart from a no-op: if the three
    normalized scores correlate near 1.0, equal-weight fusion is close to a
    rescaled copy of any one of them, and "fusion changed nothing" is arithmetic
    rather than evidence about the horizons.
    """

    rows = []
    for left in columns:
        row = {"score_column": left}
        for right in columns:
            row[right] = (1.0 if left == right
                          else daily_rank_correlation(scored["date"], scored[left], scored[right]))
        rows.append(row)
    return pd.DataFrame(rows)


def basket_overlap(
    scored: pd.DataFrame, columns: Sequence[str], top_k: float, reference_column: str
) -> pd.DataFrame:
    """How much each score column's Top-K basket shares with the reference's.

    Measured at the label level with ``_top_k_mask`` -- the same mask
    ``evaluate_outcome_predictions`` uses -- so it is the pre-engine selection,
    not the held book.  It says whether an arm is even a different strategy:
    E6 measured that two arms at the same basket width can hold different names
    and diverge by tens of percentage points, so overlap is the cheap first read
    on whether a fused arm has anything new to say.
    """

    reference = _top_k_mask(scored["date"], scored[reference_column].to_numpy(dtype=float), top_k)
    reference_count = int(reference.sum())
    rows = []
    for column in columns:
        mask = _top_k_mask(scored["date"], scored[column].to_numpy(dtype=float), top_k)
        intersection = int(np.sum(reference & mask))
        union = int(np.sum(reference | mask))
        rows.append({
            "score_column": column,
            "selected_count": int(mask.sum()),
            "overlap_with_reference": float(intersection / reference_count) if reference_count else float("nan"),
            "jaccard": float(intersection / union) if union else float("nan"),
        })
    return pd.DataFrame(rows)


def selection_diagnostic(
    samples: pd.DataFrame, scores: np.ndarray, top_k: float, label_horizon: int
) -> Dict[str, float]:
    """What an arbitrary score's Top-K basket earns, on one label horizon.

    Deliberately **not** ``evaluate_outcome_predictions``: that function derives
    the score from a head pair, and a fused score has no head pair.  Feeding it a
    fused score dressed up as one head (with a zero mae head, say) would return
    plausible-looking RMSE and rank-IC columns computed against a *prediction
    that was never made*.  This returns only the quantities that are well defined
    for any score vector: the basket's realised outcomes and its excess over the
    universe mean.

    The label horizon is fixed at the execution horizon, because "what the
    basket earned" only has a tradable meaning over the period the strategy
    actually holds.
    """

    excess_column, mae_column = target_columns(label_horizon)
    actual_excess = np.asarray(pd.to_numeric(samples[excess_column], errors="raise"), dtype=float)
    actual_mae = np.asarray(pd.to_numeric(samples[mae_column], errors="raise"), dtype=float)
    selected = _top_k_mask(samples["date"], np.asarray(scores, dtype=float), top_k)
    selected_excess = actual_excess[selected]
    universe_mean = float(np.mean(actual_excess))
    return {
        "label_horizon": int(label_horizon),
        "selected_count": int(selected.sum()),
        "selected_mean_excess_return": float(np.mean(selected_excess)),
        "selected_median_excess_return": float(np.median(selected_excess)),
        "selected_positive_return_rate": float(np.mean(selected_excess > 0)),
        "selected_mean_mae": float(np.mean(actual_mae[selected])),
        "universe_mean_excess_return": universe_mean,
        "selected_excess_over_universe": float(np.mean(selected_excess) - universe_mean),
    }


def fusion_verdict(
    validation_phase_mean: float,
    test_phase_mean: float,
    test_phase_min: float,
    min_improvement: float = MIN_EXCESS_IMPROVEMENT,
) -> Tuple[Dict[str, bool], str]:
    """Criteria 1 (decomposed) plus the pre-registered UNDECIDABLE branch.

    The branch order matters and is fixed here, not at the call site: a candidate
    that clears the adoption bar is adopted regardless of its validation reading,
    and ``UNDECIDABLE`` is only reachable when the adoption bar was **not** met.
    Without that order a fused arm that won both windows but had one negative
    weekday would fall into the undecidable branch's condition and be filed as
    "we cannot tell" instead of "it failed criterion 2".
    """

    criteria = {
        "criterion_1_test_phase_mean_ge_2pp": bool(float(test_phase_mean) >= float(min_improvement)),
        "criterion_2_test_phase_min_positive": bool(float(test_phase_min) > 0),
        "criterion_3_same_sign_both_windows": bool(
            float(validation_phase_mean) > 0 and float(test_phase_mean) > 0),
    }
    if all(criteria.values()):
        return criteria, "candidate clears the adoption criteria -> eligible for an adoption decision"
    if (float(validation_phase_mean) >= float(min_improvement)
            and float(test_phase_mean) < float(min_improvement)):
        return criteria, (
            "wins on validation only -> UNDECIDABLE (pre-registered): one validation window cannot "
            "separate a real edge from a favourable draw")
    return criteria, "no change"


def evaluate_pool(
    pool: str,
    dataset_dir: Path,
    output_dir: Path,
    horizon: int = EXECUTION_HORIZON,
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    price_history_dirs: Optional[Sequence[str]] = None,
    weekdays: Sequence[int] = WEEKDAYS,
) -> Dict[str, object]:
    if int(horizon) != EXECUTION_HORIZON:
        raise ValueError(
            "E3 fuses scores and still executes at h={}; running at h={} would move the holding "
            "period as well and is a different experiment".format(EXECUTION_HORIZON, horizon))
    spec = POOLS[pool]
    model_id = spec["model_id"]
    top_k = published_top_k(dataset_dir, model_id)
    model = load_published_model(dataset_dir, model_id)
    feature_columns = list(model["feature_columns"])
    published_manifest = json.loads(
        (dataset_dir / "trade_outcome_manifest_{}.json".format(model_id)).read_text(encoding="utf-8"))
    alpha_reference = float(published_manifest["selected_alpha"])
    risk_penalty = float(published_manifest["selected_risk_penalty"])
    print("\n{}  model={}  top_k={} (published)  alpha(h10)={}  risk_penalty={}  features={}".format(
        pool, model_id, top_k, alpha_reference, risk_penalty, len(feature_columns)))

    by_horizon = {
        value: {
            split: load_outcome_split(dataset_dir, spec["dataset_id"], split, feature_columns, value)
            for split in ("train", "validation", "test")
        }
        for value in HORIZONS
    }
    for split in ("train", "validation", "test"):
        assert_horizon_row_alignment({value: by_horizon[value][split] for value in HORIZONS}, split)
        print("  {} rows: {} (identical across h={})".format(
            split, len(by_horizon[EXECUTION_HORIZON][split]), list(HORIZONS)))
    samples = by_horizon[EXECUTION_HORIZON]

    # --- gate: the published pickle, then the retraining path ------------------
    published_delta = anchor_check(model, samples["test"], EXECUTION_HORIZON, risk_penalty, dataset_dir, model_id)
    retrain_delta = retrain_anchor_check(
        samples["train"], samples["test"], model, alpha_reference, risk_penalty, dataset_dir, model_id)
    print("  anchor 1: published pickle reproduces disk entry_score, max |delta| = {:.2e}".format(
        published_delta))
    print("  anchor 2: RETRAINED h=10 reproduces disk entry_score, max |delta| = {:.2e}".format(
        retrain_delta))

    var_reference = {
        head: float(np.var(_column(samples["train"], target_columns(EXECUTION_HORIZON)[index])))
        for index, head in enumerate(("excess", "mae"))
    }
    heads_by_horizon = {
        value: fit_horizon_heads(
            by_horizon[value]["train"], model["preprocessor"], value, alpha_reference, var_reference)
        for value in HORIZONS
    }
    print("\n  scale-consistent alpha (var measured on train; ratio is var(h)/var(h10)):")
    print("    {:<5} {:>13} {:>13} {:>9} {:>9} {:>12} {:>12}".format(
        "h", "var(excess)", "var(mae)", "r_exc", "r_mae", "alpha_exc", "alpha_mae"))
    for value in HORIZONS:
        heads = heads_by_horizon[value]
        print("    {:<5} {:>13.5e} {:>13.5e} {:>9.4f} {:>9.4f} {:>12.2f} {:>12.2f}".format(
            value, heads["var_excess"], heads["var_mae"], heads["ratio_excess"], heads["ratio_mae"],
            heads["alpha_excess"], heads["alpha_mae"]))

    scored, composition, sensitivity = {}, {}, {}
    for split in ("validation", "test"):
        frame = build_horizon_scores(
            samples[split], model["preprocessor"], heads_by_horizon, risk_penalty, EXECUTION_HORIZON)
        scored[split] = attach_published_score(dataset_dir, model_id, frame, split)
        assert_scaling_is_engine_neutral(scored[split], top_k)
        rows = []
        for value in HORIZONS:
            heads = heads_by_horizon[value]
            excess_column, mae_column = target_columns(value)
            rows.append({
                "horizon": int(value),
                **head_variance_share(
                    _column(scored[split], "h{}_predicted_excess".format(value)),
                    _column(scored[split], "h{}_predicted_mae".format(value)),
                    risk_penalty, scored[split]["date"]),
                "alpha_excess": heads["alpha_excess"],
                "alpha_mae": heads["alpha_mae"],
                "target_column_excess": excess_column,
                "target_column_mae": mae_column,
            })
        composition[split] = pd.DataFrame(rows)
        # Read off ``scored[split]``, not the pre-merge frame: the published
        # column only exists after ``attach_published_score`` has joined it.
        sensitivity[split] = pd.DataFrame([
            {"split": split, "score_column": name, **selection_diagnostic(
                scored[split], _column(scored[split], name), top_k, EXECUTION_HORIZON)}
            for name in (PUBLISHED_ARM, *DIAGNOSTIC_COLUMNS, FUSION_ARM, FUSED_NO20_ARM, FUSED_NO5_ARM)
        ])

    for split in ("validation", "test"):
        print("\n  {}: degenerate (single-name or zero-spread) dates per horizon: {}".format(
            split, {value: len(degenerate_dates(scored[split]["S_h{}".format(value)], scored[split]["date"]))
                    for value in HORIZONS}))

    print("\n  combine head variance share (a few points of drift across horizons is expected: "
          "the two labels do not scale identically):")
    for split in ("validation", "test"):
        for _, row in composition[split].iterrows():
            print("    {:<9} h={:<3} excess {:5.1%}  mae-term {:5.1%}  (std {:.5f} / {:.5f})".format(
                split, int(row["horizon"]), row["excess_variance_share"], row["mae_variance_share"],
                row["mean_xsec_std_excess"], row["mean_xsec_std_mae_term"]))

    print("\n  cross-horizon rank correlation of the normalized scores:")
    for split in ("validation", "test"):
        correlation = cross_horizon_correlation(scored[split], DIAGNOSTIC_COLUMNS)
        values = "  ".join("{}/{} {:+.4f}".format(a.split("_")[1], b.split("_")[1], correlation.loc[i, b])
                           for i, a in enumerate(DIAGNOSTIC_COLUMNS)
                           for b in DIAGNOSTIC_COLUMNS[i + 1:])
        print("    {:<9} {}".format(split, values))

    print("\n  basket overlap with {} (Top-K label-level selection):".format(PUBLISHED_ARM))
    for split in ("validation", "test"):
        overlap = basket_overlap(
            scored[split],
            (PUBLISHED_ARM, *DIAGNOSTIC_COLUMNS, FUSION_ARM, FUSED_NO20_ARM, FUSED_NO5_ARM),
            top_k, PUBLISHED_ARM)
        for _, row in overlap.iterrows():
            print("    {:<9} {:<13} overlap {:6.1%}  jaccard {:6.1%}  selected {}".format(
                split, row["score_column"], row["overlap_with_reference"], row["jaccard"],
                int(row["selected_count"])))

    print("\n  Top-K basket measured on the EXECUTED horizon (h={}) -- selection only, no head "
          "pair is involved:".format(EXECUTION_HORIZON))
    for split in ("validation", "test"):
        for _, row in sensitivity[split].iterrows():
            print("    {:<9} {:<13} sel.excess/universe={:+.4%}  sel.mae={:+.4f}".format(
                split, row["score_column"], row["selected_excess_over_universe"], row["selected_mean_mae"]))

    # Per-horizon predictive metrics on each horizon's OWN labels.  The engine
    # never reads these, and for h!=10 there is no published model to anchor
    # against, so the only thing they are used for is the pre-declared sanity
    # check that each horizon's model is not simply broken.
    label_rows = []
    for split in ("validation", "test"):
        for value in HORIZONS:
            label_rows.append({"split": split, **evaluate_outcome_predictions(
                # That horizon's own frame, which is what carries its label columns.
                # Positional pairing with ``scored`` is licensed by the alignment
                # gate above, not assumed here.
                by_horizon[value][split],
                _column(scored[split], "h{}_predicted_excess".format(value)),
                _column(scored[split], "h{}_predicted_mae".format(value)),
                value, risk_penalty, top_k, split, "horizon_{}".format(value),
            )})
    label_metrics = pd.DataFrame(label_rows)
    print("\n  own-label predictive metrics (sanity check only -- h=5/h=20 have NO published anchor):")
    for _, row in label_metrics.iterrows():
        print("    {:<9} h={:<3} rank_ic(excess)={:+.6f}  mae_rank_ic={:+.6f}  rmse(excess)={:.5f}".format(
            row["split"], int(row["horizon"]), row["excess_return_rank_ic"], row["mae_rank_ic"],
            row["excess_return_rmse"]))
    broken = [
        value for value in HORIZONS
        if all(float(label_metrics.loc[(label_metrics["split"] == split) & (label_metrics["horizon"] == value),
                                       "excess_return_rank_ic"].iloc[0]) < 0
               for split in ("validation", "test"))
    ]
    if broken:
        print("    !! h={} has a NEGATIVE own-label rank IC in BOTH windows.  By the pre-registered "
              "interpretive rule the fusion reading below is VOID: investigate that horizon's model "
              "rather than reading the fusion.".format(broken))

    def cell(split, model_name, column):
        return float(label_metrics.loc[
            (label_metrics["split"] == split) & (label_metrics["model"] == model_name), column].iloc[0])

    # --- engine runs -----------------------------------------------------------
    null_returns = load_null_total_returns(
        dataset_dir, model_id, spec["null_tag"], top_k, len(scored["test"]), max_positions=max_positions)
    print("\n  null: {} draws, mean {:+.2%} (verified top_k, rows, cap)".format(
        len(null_returns), null_returns.mean()))

    raw, slots = {}, {}
    for split in ("validation", "test"):
        raw[split], skipped = build_rank_market_data(
            scored[split], price_history_dir, PUBLISHED_ARM, price_history_dirs=price_history_dirs)
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
                    top_k, EXECUTION_HORIZON, initial_capital, max_positions, config_path,
                    weekday=int(weekday), label=arm)
                runs[(split, arm, int(weekday))] = run
                rows.append({
                    "split": split, "weekday": int(weekday), "arm": arm,
                    **{key: value for key, value in run["performance"].items() if key != "strategy"},
                })
        phase_rows[split] = pd.DataFrame(rows)

    # --- the control: B_h10 must BE A_published, not merely resemble it --------
    for split in ("validation", "test"):
        published = phase_rows[split].loc[phase_rows[split]["arm"] == PUBLISHED_ARM].sort_values("weekday")
        control = phase_rows[split].loc[phase_rows[split]["arm"] == BASELINE_ARM].sort_values("weekday")
        left = published["total_return"].to_numpy(dtype=float)
        right = control["total_return"].to_numpy(dtype=float)
        if not np.array_equal(left, right):
            raise ValueError(
                "the normalized h=10 arm does not reproduce the published arm on {}: max |delta| = "
                "{:.3e}.  The normalization is not a pure change of unit, so every fused arm is "
                "uninterpretable".format(split, float(np.max(np.abs(left - right)))))
    print("\n  control: B_h10 reproduces A_published on every weekday of both windows "
          "(a positive per-date factor cannot reorder the candidates)")

    # --- criteria: paired over the weekday family ------------------------------
    # Paired means "same weekday, candidate minus baseline", computed per weekday
    # before averaging; see ``paired_phase_excess``.
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
        validation_value = float(phase_rows["validation"].groupby("arm")["total_return"].mean().loc[arm])
        print("    {:<13} validation={:>8.2%}  test mean={:>8.2%} (min {:>7.2%}, max {:>7.2%})".format(
            arm, validation_value, test_value,
            float(summary.loc[summary["arm"] == arm, "test_phase_min"].iloc[0]),
            float(summary.loc[summary["arm"] == arm, "test_phase_max"].iloc[0])))

    downside_rows = []
    for split in ("validation", "test"):
        for arm in ARMS:
            per_weekday = pd.DataFrame([
                basket_downside(scored[split], runs[(split, arm, int(weekday))]["selected"],
                                EXECUTION_HORIZON)
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

    print("\n  basket downside, weekday-family mean:")
    for split in ("validation", "test"):
        for arm in ARMS:
            row = downside[(downside["split"] == split) & (downside["arm"] == arm)].iloc[0]
            print("    {:<9} {:<13} sel.mae={:+.4f}  sel.excess={:+.4f}  universe.mae={:+.4f}".format(
                split, arm, row["selected_mean_future_mae"], row["selected_mean_future_excess"],
                row["universe_mean_future_mae"]))

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
    print("\n  four gates (REPORTED ONLY -- not a criterion for E3; the fold gate is known "
          "closed at its configured default, which is why the plan's 'clears the four gates' "
          "clause is treated as unreachable):")
    print("    {} -- {}".format("NO CHANGE" if gates.row is None else gates.row["arm"], gates.reason))

    # --- verdicts --------------------------------------------------------------
    criteria, verdicts = {}, {}
    for arm in (CANDIDATE_ARM, *DESCRIPTIVE_ARMS):
        effect = effects[arm]
        crit, verdict = fusion_verdict(
            effect["validation"]["phase_mean"], effect["test"]["phase_mean"], effect["test"]["phase_min"])
        criteria[arm] = crit
        verdicts[arm] = verdict if arm == CANDIDATE_ARM else "DESCRIPTIVE (not adoptable): " + verdict

    print("\n  criteria (observed next to each, so a margin can be read off):")
    for arm in (CANDIDATE_ARM, *DESCRIPTIVE_ARMS):
        effect = effects[arm]
        observed = {
            "criterion_1_test_phase_mean_ge_2pp": "{:+.2f}pp vs +2.00pp".format(
                100.0 * effect["test"]["phase_mean"]),
            "criterion_2_test_phase_min_positive": "{:+.2f}pp".format(100.0 * effect["test"]["phase_min"]),
            "criterion_3_same_sign_both_windows": "val {:+.2f}pp, test {:+.2f}pp".format(
                100.0 * effect["validation"]["phase_mean"], 100.0 * effect["test"]["phase_mean"]),
        }
        rank_ic_note = " | own-label excess rank IC " + "  ".join(
            "h{} val {:+.6f} test {:+.6f}".format(value, cell("validation", "horizon_{}".format(value),
                                                              "excess_return_rank_ic"),
                                                  cell("test", "horizon_{}".format(value),
                                                       "excess_return_rank_ic"))
            for value in HORIZONS)
        print("    {}{}".format(arm, rank_ic_note))
        for key, value in criteria[arm].items():
            print("      {:<42} {:<4}  {}".format(key, "PASS" if value else "FAIL", observed[key]))
        print("      null percentile (criterion 5, declared non-discriminating): {:.1%} "
              "(baseline itself {:.1%})".format(percentiles[arm], percentiles[BASELINE_ARM]))
        print("      verdict: {}".format(verdicts[arm]))

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "multi_horizon_{}".format(pool)
    label_metrics.to_csv(output_dir / (prefix + "_label_metrics.csv"), index=False)
    pd.concat([frame.assign(split=split) for split, frame in sensitivity.items()],
              ignore_index=True).to_csv(output_dir / (prefix + "_selection_diagnostics.csv"), index=False)
    pd.concat([frame.assign(split=split) for split, frame in composition.items()],
              ignore_index=True).to_csv(output_dir / (prefix + "_score_composition.csv"), index=False)
    pd.concat([cross_horizon_correlation(scored[split], DIAGNOSTIC_COLUMNS).assign(split=split)
               for split in ("validation", "test")],
              ignore_index=True).to_csv(output_dir / (prefix + "_horizon_correlation.csv"), index=False)
    pd.concat([basket_overlap(scored[split],
                              (PUBLISHED_ARM, *DIAGNOSTIC_COLUMNS, FUSION_ARM, FUSED_NO20_ARM, FUSED_NO5_ARM),
                              top_k, PUBLISHED_ARM).assign(split=split)
               for split in ("validation", "test")],
              ignore_index=True).to_csv(output_dir / (prefix + "_basket_overlap.csv"), index=False)
    summary.to_csv(output_dir / (prefix + "_test_metrics.csv"), index=False)
    pd.concat(phase_rows.values(), ignore_index=True).to_csv(output_dir / (prefix + "_phases.csv"), index=False)
    downside.to_csv(output_dir / (prefix + "_basket_downside.csv"), index=False)
    fold_columns = ["fold_{}".format(index) for index in range(len(folds))]
    grid.merge(phase_excess[["arm", "phase_excess_min", "phases"]], on="arm", how="left").merge(
        fold_excess[["arm"] + fold_columns], on="arm", how="left").to_csv(
        output_dir / (prefix + "_validation_grid.csv"), index=False)

    report = {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "risk_penalty": risk_penalty,
        "alpha_reference": alpha_reference,
        "horizons": [int(value) for value in HORIZONS],
        "execution_horizon": EXECUTION_HORIZON,
        "arm_horizons": {arm: [int(value) for value in horizons] for arm, horizons in ARM_HORIZONS.items()},
        "candidate_arm": CANDIDATE_ARM,
        "descriptive_arms": list(DESCRIPTIVE_ARMS),
        "alpha_by_horizon": {
            str(value): {"excess": heads_by_horizon[value]["alpha_excess"],
                         "mae": heads_by_horizon[value]["alpha_mae"],
                         "ratio_excess": heads_by_horizon[value]["ratio_excess"],
                         "ratio_mae": heads_by_horizon[value]["ratio_mae"],
                         "var_excess": heads_by_horizon[value]["var_excess"],
                         "var_mae": heads_by_horizon[value]["var_mae"]}
            for value in HORIZONS},
        "anchor_published_delta": published_delta,
        "anchor_retrain_delta": retrain_delta,
        "head_variance_share": {split: composition[split].to_dict(orient="records")
                                for split in ("validation", "test")},
        "horizon_rank_ic_both_windows_negative": broken,
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
            "horizon": EXECUTION_HORIZON,
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
    parser = argparse.ArgumentParser(description="E3: multi-horizon score fusion vs h=10 alone")
    parser.add_argument("--pool", choices=sorted(POOLS), default=None, help="Default: both pools.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=EXECUTION_HORIZON,
                        help="Execution horizon; must be {} (the published one).".format(EXECUTION_HORIZON))
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
