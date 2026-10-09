"""E3's frozen design: the alpha scale rule, the fusion, and the two hard gates.

Two things in this experiment are arithmetic that a later edit could quietly
change the meaning of, so both are pinned numerically rather than argued for in
prose:

* the **per-horizon alpha rule**, which is the only thing keeping "the horizons
  differ in nothing but the horizon" true -- at the reference horizon it has to be
  the exact identity, because that is what makes the h=10 arm reproduce the
  published model;
* the **cross-sectional scale**, whose only licence is being a per-date change of
  unit.  If it ever stops being monotone (or stops preserving ties), every fused
  arm silently becomes a different arm, and the ``B_h10`` control is the thing
  that would catch it -- so the property is tested directly here, not only
  through the gate.  One test pins the sharper version: the missing-score
  sentinel ``0.0`` must not move relative to real scores, which is what forbids a
  mean shift and is why this is a scale division rather than a z-score.

The rest is the evaluation discipline: horizons must score the same
cross-section, a fused score has no head pair to compute error metrics against,
and the pre-registered UNDECIDABLE branch must not swallow a failure of the
adoption bar.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments import multi_horizon_fusion_ab
from research.experiments.multi_horizon_fusion_ab import (
    ARMS,
    ARM_HORIZONS,
    BASELINE_ARM,
    CANDIDATE_ARM,
    DESCRIPTIVE_ARMS,
    DIAGNOSTIC_COLUMNS,
    EXECUTION_HORIZON,
    FUSED_NO5_ARM,
    FUSED_NO20_ARM,
    FUSION_ARM,
    HORIZONS,
    NORMALIZED_COLUMNS,
    POOLS,
    PUBLISHED_ARM,
    assert_horizon_row_alignment,
    assert_scaling_is_engine_neutral,
    basket_overlap,
    build_horizon_scores,
    cross_horizon_correlation,
    cross_sectional_scale,
    daily_rank_correlation,
    degenerate_dates,
    fit_horizon_heads,
    fusion_verdict,
    head_variance_share,
    selection_diagnostic,
)
from research.market_regime.trade_outcome_model import target_columns

#: The only columns the identity preprocessor hands to Ridge; a blanket identity
#: would feed the ``date`` column to sklearn as a feature, which raises rather
#: than being silently ignored.
FEATURE_COLUMNS = ("feature_a", "feature_b")


@pytest.fixture(autouse=True)
def _bypass_preprocessing(monkeypatch):
    monkeypatch.setattr(
        multi_horizon_fusion_ab, "transform_features",
        lambda samples, pre: samples.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float))


def _dates(count, day="2022-07-20"):
    return pd.Series(pd.to_datetime([day] * count))


def _symbols(count):
    return ["S{:03d}".format(index) for index in range(count)]


def _target_frame(horizon, excess, mae, day="2022-07-20"):
    """A frame carrying ``horizon``'s target columns and nothing else but keys."""

    excess_column, mae_column = target_columns(horizon)
    count = len(excess)
    frame = pd.DataFrame({
        "date": pd.to_datetime([day] * count),
        "symbol": _symbols(count),
        excess_column: np.asarray(excess, dtype=float),
        mae_column: np.asarray(mae, dtype=float),
    })
    for index, name in enumerate(FEATURE_COLUMNS):
        frame[name] = np.arange(count, dtype=float) + index
    return frame


class _StubHead:
    """A head that returns a fixed vector, so the arithmetic is what is tested."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def predict(self, features):
        return self._values


def _stub_heads(values_by_horizon):
    return {
        horizon: {"excess_model": _StubHead(excess), "mae_model": _StubHead(mae)}
        for horizon, (excess, mae) in values_by_horizon.items()
    }


# --------------------------------------------------------------------------- #
# the frozen design
# --------------------------------------------------------------------------- #

def test_the_arm_set_and_the_only_candidate_are_pinned():
    assert ARMS == ("A_published", "B_h10", "C_fused_eq", "D_fused_no20", "E_fused_no5")
    assert BASELINE_ARM == "B_h10"
    assert CANDIDATE_ARM == "C_fused_eq"
    assert DESCRIPTIVE_ARMS == ("D_fused_no20", "E_fused_no5")
    # A leave-one-out arm is running a different question and must never be the
    # thing that gets adopted; the verdict string carries the marker.
    assert CANDIDATE_ARM not in DESCRIPTIVE_ARMS


def test_the_fused_arms_average_exactly_the_horizons_they_name():
    assert ARM_HORIZONS[FUSION_ARM] == (5, 10, 20)
    assert ARM_HORIZONS[BASELINE_ARM] == (10,)
    assert ARM_HORIZONS[FUSED_NO20_ARM] == (5, 10)
    assert ARM_HORIZONS[FUSED_NO5_ARM] == (10, 20)
    # Every fused arm contains h=10, or it would not be a variation on the baseline.
    for arm, horizons in ARM_HORIZONS.items():
        assert 10 in horizons, arm


def test_the_baseline_column_is_the_h10_normalized_column():
    """``B_h10`` and ``N_h10`` are one vector under two names; if they ever drift
    apart the control stops being the thing the fusion is compared against."""

    assert NORMALIZED_COLUMNS[10] == BASELINE_ARM
    assert sorted(NORMALIZED_COLUMNS) == list(HORIZONS)
    assert NORMALIZED_COLUMNS[5] in DIAGNOSTIC_COLUMNS
    assert NORMALIZED_COLUMNS[20] in DIAGNOSTIC_COLUMNS


def test_the_execution_horizon_is_the_published_one_and_it_is_in_the_grid():
    assert EXECUTION_HORIZON == 10
    assert EXECUTION_HORIZON in HORIZONS


def test_the_pool_registry_names_the_published_dataset_ids():
    """A wrong ``dataset_id`` silently loads a different cross-section."""

    assert POOLS["hs300"]["dataset_id"] == "excess_hs300_pit_v1"
    assert POOLS["zz1000"]["dataset_id"] == "excess_zz1000_pit_v1"
    assert POOLS["hs300"]["model_id"] == "hs300_pit_outcome"
    assert POOLS["zz1000"]["model_id"] == "zz1000_pit_outcome"


# --------------------------------------------------------------------------- #
# the alpha scale rule
# --------------------------------------------------------------------------- #

def test_the_alpha_rule_scales_with_the_target_variance():
    """``Ridge(a) on y`` is ``Ridge(a c^2) on c y``, so holding the shrinkage
    fixed while the horizon changes the target's unit means alpha follows the
    variance -- per head, because the two heads do not scale together."""

    excess = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    mae = [0.02, 0.01, 0.03, 0.015, 0.025, 0.005]
    train = _target_frame(20, excess, mae)
    var_excess = float(np.var(excess))
    var_mae = float(np.var(mae))

    heads = fit_horizon_heads(
        train, {}, 20, 100.0, {"excess": var_excess / 4.0, "mae": var_mae / 2.0})

    assert heads["alpha_excess"] == pytest.approx(400.0)
    assert heads["alpha_mae"] == pytest.approx(200.0)
    assert heads["ratio_excess"] == pytest.approx(4.0)
    assert heads["ratio_mae"] == pytest.approx(2.0)


def test_the_alpha_rule_is_the_exact_identity_at_the_reference_horizon():
    """The h=10 arm has to be the published fit, not merely a fit at the same
    alpha: anything but an exact 1.0 here and the control is off by a rounding."""

    excess = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    mae = [0.02, 0.01, 0.03, 0.015, 0.025, 0.005]
    train = _target_frame(10, excess, mae)

    heads = fit_horizon_heads(
        train, {}, 10, 100.0,
        {"excess": float(np.var(excess)), "mae": float(np.var(mae))})

    assert heads["ratio_excess"] == 1.0
    assert heads["ratio_mae"] == 1.0
    assert heads["alpha_excess"] == 100.0


def test_the_alpha_rule_reads_only_the_split_it_is_given():
    """The rule is a property of the labels the model is fitted to, so the
    variances it reports must be the ones it was handed -- not, say, a
    module-level constant measured somewhere else."""

    excess = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    mae = [0.02, 0.01, 0.03, 0.015, 0.025, 0.005]
    heads = fit_horizon_heads(
        _target_frame(5, excess, mae), {}, 5, 100.0, {"excess": 1.0, "mae": 1.0})

    assert heads["var_excess"] == pytest.approx(float(np.var(excess)))
    assert heads["var_mae"] == pytest.approx(float(np.var(mae)))
    assert heads["train_rows"] == len(excess)


def test_a_degenerate_target_is_rejected():
    """A zero-variance target makes the scale rule 0/0 and would silently give
    the head no regularisation at all."""

    with pytest.raises(ValueError, match="degenerate target"):
        fit_horizon_heads(
            _target_frame(5, [0.01] * 6, [0.02, 0.01, 0.03, 0.015, 0.025, 0.005]),
            {}, 5, 100.0, {"excess": 0.001, "mae": 0.001})


# --------------------------------------------------------------------------- #
# the cross-sectional scale
# --------------------------------------------------------------------------- #

def test_the_scale_divides_by_the_spread_within_a_date():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    scaled = cross_sectional_scale(values, _dates(4))

    assert scaled.std(ddof=1) == pytest.approx(1.0)
    # Scale only: the level is deliberately *not* removed.  See the sentinel test.
    assert scaled == pytest.approx(values / values.std(ddof=1))


def test_the_scale_does_not_recentre_a_zero_sentinel():
    """**Regression test for the bug the ``B_h10`` gate caught.**

    ``build_rank_market_data`` fills a name with no score row by ``0.0`` and the
    strategy sorts that sentinel against real scores.  Crucially the ``0.0`` is a
    **hardcoded literal that is never passed through the transform** -- it is not
    the transformed value of anything -- so it is the *real* scores that move
    relative to it:

    * a positive per-date factor leaves every real score on the same side of zero
      it was on, so the sentinel's rank position is bit-identical to the
      published arm's;
    * a per-date mean shift moves the reals across it whenever the date's mean has
      the opposite sign, which is the common case here.

    Measured on the HS300 test window the shift cost one slot on 2025-06-30, the
    only affected date, and 0.89pp of net return.  Pinned rather than commented
    because the gate that caught it only covers the basket, not ``rank_score``.
    """

    # A date whose real scores are all negative -- as the raw h=10 scores were on
    # the measured date -- with one name carrying the missing-score sentinel.
    real = np.array([-0.004, -0.003, -0.002])
    values = np.append(real, 0.0)
    sentinel = 3

    def rank_of_sentinel(real_scores):
        stacked = np.append(np.asarray(real_scores, dtype=float), 0.0)
        return int(np.sum(stacked > 0.0))

    scaled = cross_sectional_scale(values, _dates(4))
    assert scaled[sentinel] == 0.0
    assert rank_of_sentinel(scaled[:3]) == rank_of_sentinel(real) == 0

    centered = (real - real.mean()) / real.std(ddof=1)
    assert rank_of_sentinel(centered) > rank_of_sentinel(real)


def test_the_scale_never_mixes_dates():
    """Two dates on wildly different scales must each be scaled on their own: a
    pooled scale would leave the large-scale date dominating the fused sum, which
    is the whole thing this step exists to prevent."""

    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"] * 3))
    values = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    scaled = cross_sectional_scale(values, dates)

    assert scaled[:3].std(ddof=1) == pytest.approx(1.0)
    assert scaled[3:].std(ddof=1) == pytest.approx(1.0)
    # Identical *relative* positions, so the two dates map to the same triple.
    assert scaled[:3] == pytest.approx(scaled[3:])


def test_a_single_name_date_becomes_zero_and_is_reported():
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"]))
    values = np.array([1.0, 2.0, 3.0, 99.0])
    scaled = cross_sectional_scale(values, dates)

    assert scaled[3] == 0.0
    assert np.isfinite(scaled).all()
    assert degenerate_dates(values, dates) == [pd.Timestamp("2022-07-21")]


def test_a_zero_spread_date_becomes_zero_and_is_reported():
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"] * 3))
    values = np.array([1.0, 2.0, 3.0, 7.0, 7.0, 7.0])
    scaled = cross_sectional_scale(values, dates)

    assert np.all(scaled[3:] == 0.0)
    assert degenerate_dates(values, dates) == [pd.Timestamp("2022-07-21")]


def test_the_scale_is_monotone_and_preserves_ties():
    """This is the arithmetic behind the ``B_h10`` gate: the engine reads the
    score's order and its tie sets (``rank(method="first")`` for the basket,
    ``(-rank_score, symbol)`` for entries), so a transform that reordered or
    split ties would change the strategy without changing any reported number
    that this file checks."""

    values = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 4.0])
    scaled = cross_sectional_scale(values, _dates(6))

    order = np.argsort(values, kind="stable")
    assert np.all(np.diff(scaled[order]) >= 0)
    assert scaled[2] == scaled[3]
    assert scaled[1] < scaled[0] < scaled[5] < scaled[4]


def test_the_scale_commutes_with_maximum():
    """``max(-s * mae, 0) == s * max(-mae, 0)`` for a positive scalar: the same
    identity the head share relies on when it measures the mae term's weight."""

    mae = np.array([-0.02, 0.01, -0.005, 0.03, -0.001, 0.02])
    scaled = cross_sectional_scale(mae, _dates(6))

    assert (scaled > 0).any() and (scaled < 0).any()
    assert np.maximum(-3.0 * mae, 0.0) == pytest.approx(3.0 * np.maximum(-mae, 0.0))


# --------------------------------------------------------------------------- #
# the row-alignment gate
# --------------------------------------------------------------------------- #

def test_row_alignment_accepts_a_shared_cross_section():
    frames = {horizon: _target_frame(horizon, [0.1] * 4, [0.2] * 4) for horizon in HORIZONS}
    assert_horizon_row_alignment(frames, "train")


def test_row_alignment_rejects_a_row_count_difference():
    """A horizon with a different eligibility pattern moves the cross-section
    under the fusion; the failure is invisible in every output table."""

    frames = {horizon: _target_frame(horizon, [0.1] * 4, [0.2] * 4) for horizon in HORIZONS}
    frames[20] = frames[20].iloc[:-1]
    with pytest.raises(ValueError, match="rows differ across horizons"):
        assert_horizon_row_alignment(frames, "train")


def test_row_alignment_rejects_the_same_size_of_different_names():
    frames = {horizon: _target_frame(horizon, [0.1] * 4, [0.2] * 4) for horizon in HORIZONS}
    frames[5] = frames[5].copy()
    frames[5]["symbol"] = ["X001", "X002", "X003", "X004"]
    with pytest.raises(ValueError, match="keys differ across horizons"):
        assert_horizon_row_alignment(frames, "train")


# --------------------------------------------------------------------------- #
# the fusion
# --------------------------------------------------------------------------- #

def _scored_frame():
    count = 6
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * count))
    frame = pd.DataFrame({"date": dates, "symbol": _symbols(count)})
    for index, name in enumerate(FEATURE_COLUMNS):
        frame[name] = np.arange(count, dtype=float) + index
    for name in target_columns(EXECUTION_HORIZON):
        frame[name] = np.linspace(0.01, 0.02, count)
    return frame


def test_the_fusion_is_the_plain_sum_of_the_z_scores():
    """Equal weight, unsearched: the plan forbids fitting fusion weights, and a
    weighted sum would reopen the configuration-selection problem the gates
    exist to prevent."""

    samples = _scored_frame()
    heads = _stub_heads({
        5: ([0.01, 0.02, -0.01, 0.005, 0.03, -0.02], [0.01] * 6),
        10: ([0.02, -0.01, 0.03, 0.01, -0.02, 0.005], [0.01] * 6),
        20: ([0.005, 0.03, -0.02, 0.02, 0.01, -0.01], [0.01] * 6),
    })
    frame = build_horizon_scores(samples, {}, heads, 0.25, EXECUTION_HORIZON)

    assert frame[FUSION_ARM].to_numpy() == pytest.approx(
        (frame[NORMALIZED_COLUMNS[5]] + frame[NORMALIZED_COLUMNS[10]]
         + frame[NORMALIZED_COLUMNS[20]]).to_numpy())
    assert frame[FUSED_NO20_ARM].to_numpy() == pytest.approx(
        (frame[NORMALIZED_COLUMNS[5]] + frame[NORMALIZED_COLUMNS[10]]).to_numpy())
    assert frame[FUSED_NO5_ARM].to_numpy() == pytest.approx(
        (frame[NORMALIZED_COLUMNS[10]] + frame[NORMALIZED_COLUMNS[20]]).to_numpy())
    assert frame[BASELINE_ARM].to_numpy() == pytest.approx(
        frame[NORMALIZED_COLUMNS[10]].to_numpy())


def test_the_fusion_mean_is_the_same_arm_as_the_sum():
    """``select_top_k`` and ``rank_score`` consume only the score's order, so the
    sum and its mean are one arm up to the positive scalar 1/3.  Pinned because
    the plan writes the sum and a reader may reasonably expect the mean."""

    values = np.array([0.3, -0.1, 0.2, 0.05, -0.4, 0.15])

    def baskets(scaled):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2022-07-20"] * 6), "score": scaled})
        ranks = frame.groupby("date", sort=False)["score"].rank(method="first", ascending=False)
        return (ranks <= 3).to_numpy()

    assert np.array_equal(baskets(values), baskets(values / 3.0))


def test_build_horizon_scores_rejects_an_unsorted_input():
    """The loader's order is the contract: ``evaluate_outcome_predictions`` is
    handed the samples and the scores positionally."""

    samples = _scored_frame().iloc[[3, 0, 1, 2, 4, 5]].reset_index(drop=True)
    heads = _stub_heads({
        5: (np.arange(6, dtype=float) / 100.0, [0.01] * 6),
        10: (np.arange(6, dtype=float) / 200.0, [0.01] * 6),
        20: (np.arange(6, dtype=float) / 300.0, [0.01] * 6),
    })
    with pytest.raises(ValueError, match="reordered the rows"):
        build_horizon_scores(samples, {}, heads, 0.25, EXECUTION_HORIZON)


# --------------------------------------------------------------------------- #
# the normalization no-op gate
# --------------------------------------------------------------------------- #

def _gate_frame(published=None, scaled=None):
    count = 8
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * count))
    raw = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * count),
        "symbol": _symbols(count),
        "S_h10": raw,
        "B_h10": cross_sectional_scale(raw, dates),
        "A_published": raw.copy(),
    })
    if published is not None:
        frame["A_published"] = published
    if scaled is not None:
        frame["B_h10"] = scaled
    return frame


def test_the_gate_accepts_a_monotone_change_of_unit():
    assert_scaling_is_engine_neutral(_gate_frame(), 0.5)


def test_the_gate_rejects_a_scaling_that_moves_the_basket():
    """If this fires, the normalization is doing something other than a change of
    unit and every fused arm inherits it.  It is the gate that caught the
    per-date mean subtraction -- on real data, through the sentinel name rather
    than through this synthetic swap."""

    scaled = _gate_frame()["B_h10"].to_numpy().copy()
    scaled[0], scaled[7] = scaled[7], scaled[0]
    with pytest.raises(ValueError, match="scaling changed the h=10 basket"):
        assert_scaling_is_engine_neutral(_gate_frame(scaled=scaled), 0.5)


def test_the_gate_rejects_a_retrain_that_moves_the_basket():
    """The same call also covers "the retraining path reproduces the published
    model": a training-pipeline difference would otherwise be read as a fusion
    result."""

    published = _gate_frame()["A_published"].to_numpy().copy()
    published[0], published[7] = published[7], published[0]
    with pytest.raises(ValueError, match="retrained h=10 basket differs"):
        assert_scaling_is_engine_neutral(_gate_frame(published=published), 0.5)


# --------------------------------------------------------------------------- #
# the diagnostics
# --------------------------------------------------------------------------- #

def test_the_correlation_matrix_is_symmetric_with_a_unit_diagonal():
    count = 6
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * count))
    frame = pd.DataFrame({
        "date": dates,
        "N_h5": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "B_h10": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        "N_h20": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    correlation = cross_horizon_correlation(frame, DIAGNOSTIC_COLUMNS)

    assert correlation.loc[0, DIAGNOSTIC_COLUMNS[0]] == 1.0
    assert correlation.loc[0, DIAGNOSTIC_COLUMNS[1]] == pytest.approx(1.0)
    assert correlation.loc[0, DIAGNOSTIC_COLUMNS[2]] == pytest.approx(-1.0)
    assert correlation.loc[0, DIAGNOSTIC_COLUMNS[1]] == correlation.loc[1, DIAGNOSTIC_COLUMNS[0]]


def test_a_constant_side_is_skipped_rather_than_counted_as_zero():
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"] * 3))
    left = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    right = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])

    assert daily_rank_correlation(dates, left, right) == pytest.approx(-1.0)


def test_basket_overlap_is_complete_against_itself_and_zero_against_a_reversal():
    count = 10
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * count))
    frame = pd.DataFrame({
        "date": dates,
        "symbol": _symbols(count),
        "A_published": np.arange(count, dtype=float),
        "C_fused_eq": 9.0 - np.arange(count, dtype=float),
    })
    overlap = basket_overlap(frame, ("A_published", "C_fused_eq"), 0.3, "A_published")

    assert overlap.loc[0, "overlap_with_reference"] == pytest.approx(1.0)
    assert overlap.loc[1, "overlap_with_reference"] == pytest.approx(0.0)
    assert overlap.loc[0, "jaccard"] == pytest.approx(1.0)


def test_selection_diagnostic_measures_the_basket_on_the_label_horizon():
    excess_column, mae_column = target_columns(10)
    count = 10
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * count),
        "symbol": _symbols(count),
        excess_column: np.arange(count, dtype=float) / 100.0,
        mae_column: -np.arange(count, dtype=float) / 1000.0,
    })
    # A score that ranks the LAST rows highest, so the basket is 5..9.
    diagnostic = selection_diagnostic(frame, np.arange(count, dtype=float), 0.5, 10)

    assert diagnostic["label_horizon"] == 10
    assert diagnostic["selected_count"] == 5
    assert diagnostic["selected_mean_excess_return"] == pytest.approx(
        float(np.mean(np.arange(5, 10) / 100.0)))
    assert diagnostic["selected_excess_over_universe"] == pytest.approx(
        float(np.mean(np.arange(5, 10) / 100.0) - np.mean(np.arange(10) / 100.0)))


def test_selection_diagnostic_reports_no_predictive_metrics():
    """A fused score has no head pair, so there is no prediction to score errors
    against.  Dressing one up as a head (a zero mae head, say) would emit
    plausible-looking RMSE and rank-IC numbers for a prediction never made."""

    excess_column, mae_column = target_columns(10)
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 4),
        "symbol": _symbols(4),
        excess_column: [0.01, 0.02, 0.03, 0.04],
        mae_column: [-0.01, -0.02, -0.03, -0.04],
    })
    diagnostic = selection_diagnostic(frame, np.arange(4, dtype=float), 0.5, 10)

    for forbidden in ("excess_return_rmse", "mae_rmse", "excess_return_rank_ic", "mae_rank_ic"):
        assert forbidden not in diagnostic


def test_head_variance_share_sums_to_one_and_ignores_a_positive_mae_head():
    count = 6
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * count))
    excess = np.array([0.02, -0.01, 0.03, 0.005, -0.02, 0.01])

    share = head_variance_share(excess, np.array([0.01] * count), 0.25, dates)
    assert share["excess_variance_share"] == pytest.approx(1.0)
    assert share["mae_variance_share"] == pytest.approx(0.0)

    # ``max(-mae, 0)``: a *negative* mae is a downside forecast and does move it.
    share = head_variance_share(excess, -np.abs(excess) * 10.0, 0.25, dates)
    assert share["excess_variance_share"] + share["mae_variance_share"] == pytest.approx(1.0)
    assert share["mae_variance_share"] > 0.5


# --------------------------------------------------------------------------- #
# the pre-registered verdict logic
# --------------------------------------------------------------------------- #

def test_a_candidate_that_clears_the_bar_is_adoptable():
    criteria, verdict = fusion_verdict(0.03, 0.05, 0.01)

    assert all(criteria.values())
    assert "eligible for an adoption decision" in verdict


def test_a_validation_only_win_is_undecidable():
    """The plan's own branch: one validation window cannot separate a real edge
    from a favourable draw, so this is a distinct verdict, not a relabelling."""

    criteria, verdict = fusion_verdict(0.06, 0.005, -0.01)

    assert not criteria["criterion_1_test_phase_mean_ge_2pp"]
    assert "UNDECIDABLE" in verdict


def test_a_sign_flip_lands_in_the_undecidable_branch():
    """Pre-registered literally: the plan's wording is unconditional, so a flip
    is reported as UNDECIDABLE with the test reading printed beside it, rather
    than being quietly reclassified as refuted."""

    _, verdict = fusion_verdict(0.06, -0.08, -0.12)
    assert "UNDECIDABLE" in verdict


def test_a_validation_win_does_not_mask_a_failed_phase_min():
    """Branch order is load-bearing.  An arm that beat the bar on both windows
    but had one negative weekday did NOT pass the adoption criteria, and the
    undecidable branch's condition must not swallow it."""

    criteria, verdict = fusion_verdict(0.10, 0.05, -0.01)

    assert criteria["criterion_1_test_phase_mean_ge_2pp"]
    assert not criteria["criterion_2_test_phase_min_positive"]
    assert "no change" == verdict


def test_losing_both_windows_is_no_change():
    criteria, verdict = fusion_verdict(-0.03, -0.05, -0.09)

    assert not any(criteria.values())
    assert verdict == "no change"


def test_a_win_that_merely_ties_the_bar_is_not_a_win():
    criteria, _ = fusion_verdict(0.05, 0.02, 0.01)
    assert criteria["criterion_1_test_phase_mean_ge_2pp"]


def test_evaluate_pool_refuses_to_move_the_execution_horizon():
    """Fusing the score and also changing the holding period would be two
    variables at once; the CLI flag exists only to make the coupling visible."""

    with pytest.raises(ValueError, match="still executes at h=10"):
        multi_horizon_fusion_ab.evaluate_pool("hs300", None, None, horizon=5)
