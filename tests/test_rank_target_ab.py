"""E4's frozen design, and the two silent reweights the pre-registration fixes.

The plan document says "Ridge unchanged, only ``y`` changes".  That sentence
cannot be executed literally: ``y``'s units propagate into ``entry_score``
through the combine (which silently becomes a mae-head ablation) and into Ridge's
``alpha`` (which silently stops regularising).  Both repairs are arithmetic, so
both are pinned here rather than argued for in prose -- if the scaling rule or the
alpha conversion ever changes, the arm named ``C_rank_excess`` would stop being
the arm the report describes.

The other half is the evaluation discipline: the phase family must be **paired**
before it is averaged, and the arm scores must stay row-aligned with the labels
they will be scored against.  Both have silent failure modes (an unbalanced family
changes the number; a reorder scores one row's prediction against another row's
label), so both raise.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments import rank_target_ab
from research.experiments.rank_target_ab import (
    ALPHA_MULTIPLIERS,
    ARMS,
    ARM_TARGETS,
    BASELINE_ARM,
    CANDIDATE_ARMS,
    POOLS,
    PUBLISHED_ARM,
    alpha_sensitivity,
    arm_predictions,
    attach_published_score,
    build_arm_scores,
    combine,
    cross_sectional_normal_rank,
    mean_cross_sectional_std,
    paired_phase_excess,
    retrain_anchor_check,
    score_composition,
    target_scale_ratio,
)
from research.market_regime.trade_outcome_model import decision_score

DATES = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"] * 3))


def _dates(count, day="2022-07-20"):
    return pd.Series(pd.to_datetime([day] * count))


#: The only columns the identity preprocessor hands to Ridge.  A real preprocessor
#: needs the full 28-column frame and would test nothing this file is about, but a
#: blanket ``return samples`` would feed the ``date`` column to sklearn as a
#: feature -- which raises rather than being silently ignored.
FEATURE_COLUMNS = ("feature_a", "feature_b")


@pytest.fixture(autouse=True)
def _bypass_preprocessing(monkeypatch):
    monkeypatch.setattr(
        rank_target_ab, "transform_features",
        lambda samples, pre: samples.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float))


# --------------------------------------------------------------------------- #
# the frozen design
# --------------------------------------------------------------------------- #

def test_the_arm_set_and_the_control_arm_are_pinned():
    assert ARMS == ("A_published", "B_mse", "C_rank_excess", "D_rank_both")
    assert BASELINE_ARM == "B_mse"
    assert CANDIDATE_ARMS == ("C_rank_excess", "D_rank_both")


def test_the_control_arm_retargets_nothing_and_the_published_arm_has_no_target():
    """``B_mse`` is what proves retraining is a no-op; it must not be a rank arm."""

    assert ARM_TARGETS["B_mse"] == ("mse", "mse")
    assert ARM_TARGETS["C_rank_excess"] == ("rank", "mse")
    assert ARM_TARGETS["D_rank_both"] == ("rank", "rank")
    assert PUBLISHED_ARM not in ARM_TARGETS


def test_the_pool_registry_names_the_published_dataset_ids():
    """A wrong ``dataset_id`` silently loads a different cross-section."""

    assert POOLS["hs300"]["dataset_id"] == "excess_hs300_pit_v1"
    assert POOLS["zz1000"]["dataset_id"] == "excess_zz1000_pit_v1"
    assert POOLS["hs300"]["model_id"] == "hs300_pit_outcome"
    assert POOLS["zz1000"]["model_id"] == "zz1000_pit_outcome"


def test_the_alpha_scan_includes_the_preregistered_reference():
    """The scan is a sensitivity check around the scale-consistent alpha, so the
    multiplier list has to contain it -- otherwise "reported, not adopted" would be
    reporting a grid that never touched the actual arm."""

    assert 1.0 in ALPHA_MULTIPLIERS
    assert sorted(ALPHA_MULTIPLIERS) == list(ALPHA_MULTIPLIERS)
    assert len(set(ALPHA_MULTIPLIERS)) == len(ALPHA_MULTIPLIERS)


# --------------------------------------------------------------------------- #
# the rank transform
# --------------------------------------------------------------------------- #

def test_the_rank_transform_maps_to_finite_normal_quantiles():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    ranks = cross_sectional_normal_rank(values, _dates(4))

    assert np.isfinite(ranks).all()
    # (rank - 0.5) / 4 -> 0.125, 0.375, 0.625, 0.875
    assert ranks == pytest.approx([-1.150349, -0.318639, 0.318639, 1.150349], abs=1e-6)
    assert np.all(np.diff(ranks) > 0)


def test_the_rank_transform_never_reaches_the_infinite_quantiles():
    """``rank / n`` would put the top row at ``ppf(1) == +inf`` and one such row
    makes the whole fit non-finite; the offset by 0.5 is what prevents it."""

    ranks = cross_sectional_normal_rank(np.arange(1.0, 21.0), _dates(20))

    assert np.isfinite(ranks).all()
    assert np.max(np.abs(ranks)) < 3.0


def test_the_rank_transform_gives_tied_labels_the_same_rank():
    ranks = cross_sectional_normal_rank(np.array([5.0, 5.0, 7.0, 9.0]), _dates(4))

    assert ranks[0] == pytest.approx(ranks[1])
    assert ranks[2] > ranks[1]


def test_a_single_name_date_maps_to_zero():
    """One name carries no ordering information; the neutral value is what keeps
    such a date in the panel rather than dropping it or sending it to infinity."""

    assert cross_sectional_normal_rank(np.array([42.0]), _dates(1)) == pytest.approx([0.0])


def test_the_rank_transform_is_per_date():
    """The same value must rank differently on different days -- a pooled rank
    would leak the market's day-to-day level into the label."""

    dates = pd.Series(pd.to_datetime(["2022-07-20", "2022-07-20", "2022-07-21", "2022-07-21"]))
    ranks = cross_sectional_normal_rank(np.array([9.0, 1.0, 1.0, 9.0]), dates)

    assert ranks == pytest.approx([0.674490, -0.674490, -0.674490, 0.674490], abs=1e-6)


def test_the_rank_transform_is_invariant_to_a_monotone_rescale():
    """Ridge sees only the target, so this is what makes the arm a statement about
    *order* rather than about the label's units."""

    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 5))
    raw = np.array([0.01, -0.02, 0.04, 0.03, -0.05])

    assert cross_sectional_normal_rank(raw, dates) == pytest.approx(
        cross_sectional_normal_rank(raw * 1000.0 + 7.0, dates))


# --------------------------------------------------------------------------- #
# defect 1 -- the combine weight
# --------------------------------------------------------------------------- #

def test_the_cross_sectional_std_averages_within_date_spreads_not_the_pool():
    """The strategy compares names *inside* a date, so this has to be the mean of
    the per-date spreads -- the pooled spread would be dominated by days that
    happen to be volatile, which is not the quantity the score is read on."""

    values = np.array([1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
    dates = pd.Series(pd.to_datetime(["2022-07-20"] * 3 + ["2022-07-21"] * 3))

    assert mean_cross_sectional_std(values, dates) == pytest.approx((1.0 + 10.0) / 2.0)


def test_the_scale_ratio_restores_the_reference_head_s_spread():
    dates = _dates(3)
    base = np.array([1.0, 2.0, 3.0])
    candidate = np.array([10.0, 20.0, 30.0])

    scale = target_scale_ratio(base, candidate, dates)

    assert scale == pytest.approx(0.1)
    assert mean_cross_sectional_std(scale * candidate, dates) == pytest.approx(
        mean_cross_sectional_std(base, dates))


def test_the_scale_ratio_is_positive_so_it_cannot_reorder_a_head():
    """If ``s`` could be negative it would flip the head's ranking instead of
    rescaling it, and the arm would measure a sign error."""

    scale = target_scale_ratio(np.array([1.0, 2.0, 3.0]), np.array([-1.0, -2.0, -3.0]), _dates(3))

    assert scale > 0


def test_a_head_with_no_spread_is_rejected_rather_than_silently_rescaled():
    with pytest.raises(ValueError) as excinfo:
        target_scale_ratio(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]), _dates(3))

    assert "no cross-sectional spread" in str(excinfo.value)


def test_scaling_commutes_with_the_downside_max():
    """``max(-s * mae, 0) == s * max(-mae, 0)`` for ``s > 0`` -- which is why the
    scaling can be applied to the head before the combine at all."""

    mae = np.array([-0.02, 0.03, -0.05])
    scale = 0.4

    assert np.maximum(-(scale * mae), 0.0) == pytest.approx(scale * np.maximum(-mae, 0.0))


def test_combine_is_the_published_decision_score():
    excess = np.array([0.01, -0.02, 0.03])
    mae = np.array([-0.02, -0.04, 0.05])

    assert combine(excess, mae, 0.25) == pytest.approx(decision_score(excess, mae, 0.25))


def test_the_mse_arm_reproduces_the_published_score_exactly():
    """``s == 1`` is the identity case: the control arm is the published formula
    applied to re-fitted heads, which is the whole point of having it."""

    excess, mae, penalty = np.array([0.01, -0.02]), np.array([-0.03, 0.04]), 0.25

    assert combine(excess, mae, penalty) == pytest.approx(decision_score(excess, mae, penalty))


def test_the_published_arm_is_never_predicted_here():
    with pytest.raises(ValueError) as excinfo:
        arm_predictions({"models": {}, "scale": {}}, np.zeros((1, 1)), PUBLISHED_ARM)

    assert "read off disk" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# defect 2 -- the regularisation scale
# --------------------------------------------------------------------------- #

def _train_frame(count=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * (count // 2) + ["2020-01-03"] * (count // 2)),
        "symbol": ["{:06d}.SZ".format(i) for i in range(count)],
        "feature_a": rng.normal(size=count),
        "feature_b": rng.normal(size=count),
        "future_excess_return_10d": rng.normal(scale=0.07, size=count),
        "future_mae_10d": rng.normal(scale=0.05, size=count),
    })


def test_the_rank_alpha_is_the_mse_alpha_in_the_rank_target_s_units():
    """``Ridge(a) on y == Ridge(a / c**2) on y / c``.  Without this the rank arm
    would be fitted at ~180x weaker regularisation and any loss would be
    attributable to the shrinkage rather than to the objective."""

    train = _train_frame()
    heads = rank_target_ab._fit_heads(train, {}, 10, 100.0)

    variance = float(np.var(train["future_excess_return_10d"].to_numpy()))
    assert heads["alpha_rank_excess"] == pytest.approx(100.0 / variance)
    assert heads["alpha_rank_mae"] == pytest.approx(
        100.0 / float(np.var(train["future_mae_10d"].to_numpy())))
    # The rank target has ~unit variance, so the converted alpha is far larger.
    assert heads["alpha_rank_excess"] > 100.0


def test_the_scale_factors_are_one_for_the_mse_head_and_measured_on_train():
    """``s`` for the MSE head is 1 *by construction* (it is the reference), and the
    rank head's factor is a train-window ratio -- never a validation or test one."""

    train = _train_frame()
    heads = rank_target_ab._fit_heads(train, {}, 10, 100.0)
    features = rank_target_ab.transform_features(train, {})
    fitted_excess = heads["models"][("excess", "mse")].predict(features)
    fitted_rank = heads["models"][("excess", "rank")].predict(features)

    assert heads["scale"]["excess"]["mse"] == pytest.approx(1.0)
    assert heads["scale"]["mae"]["mse"] == pytest.approx(1.0)
    assert heads["scale"]["excess"]["rank"] == pytest.approx(
        target_scale_ratio(fitted_excess, fitted_rank, train["date"]))


def test_a_degenerate_target_variance_is_rejected():
    """``alpha / var(y)`` is undefined at zero variance, and an infinite alpha
    would be silently accepted by Ridge."""

    train = _train_frame()
    train["future_excess_return_10d"] = 0.0

    with pytest.raises(ValueError) as excinfo:
        rank_target_ab._fit_heads(train, {}, 10, 100.0)

    assert "no variance" in str(excinfo.value)


def _anchor_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20", "2022-07-21"]),
        "symbol": ["000001.SZ", "000002.SZ"],
        "feature_a": [0.5, 1.5],
        "feature_b": [-0.5, 0.25],
        "future_excess_return_10d": [0.01, 0.02],
        "future_mae_10d": [-0.03, -0.04],
    })


def test_the_alpha_sensitivity_reports_label_metrics_per_multiplier():
    """It exists so a reader can see whether a label-level conclusion survives the
    shrinkage choice -- so every multiplier must produce a row on every window."""

    train = _train_frame(count=60, seed=1)
    splits = {"validation": train.copy(), "test": train.copy()}
    scan = alpha_sensitivity(train, splits, {}, 10, 0.25, 0.05, 1.0)

    assert sorted(set(scan["alpha_multiplier"])) == pytest.approx(sorted(ALPHA_MULTIPLIERS))
    assert set(scan["split"]) == {"validation", "test"}
    assert len(scan) == len(ALPHA_MULTIPLIERS) * 2
    # Every cell of the grid is present, not just every multiplier.
    assert set(zip(scan["alpha_multiplier"], scan["split"])) == {
        (multiplier, split) for multiplier in ALPHA_MULTIPLIERS for split in ("validation", "test")
    }
    assert scan["excess_return_rank_ic"].notna().all()


# --------------------------------------------------------------------------- #
# score composition -- the direct evidence defect 1 is repaired
# --------------------------------------------------------------------------- #

def test_the_composition_splits_the_score_s_between_date_variance_between_heads():
    """For the published arm the split is measured (not assumed), so a frame whose
    two heads have equal within-date spread must read 50:50."""

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 4),
        "A_published_predicted_excess": [0.01, 0.02, 0.03, 0.04],
        "A_published_predicted_mae": [-0.01, -0.02, -0.03, -0.04],
    })
    row = score_composition(frame, 1.0, arms=(PUBLISHED_ARM,)).iloc[0]

    assert row["excess_variance_share"] == pytest.approx(0.5)
    assert row["mae_variance_share"] == pytest.approx(0.5)


def test_a_fully_positive_mae_head_contributes_no_variance_to_the_score():
    """The head enters only through ``max(-mae, 0)``, so a head that predicts no
    downside at all is not merely down-weighted -- it is absent from the score.
    Reading the raw head's spread here would report a weight it never has."""

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 4),
        "A_published_predicted_excess": [0.01, 0.02, 0.03, 0.04],
        "A_published_predicted_mae": [0.01, 0.02, 0.03, 0.04],
    })
    row = score_composition(frame, 1.0, arms=(PUBLISHED_ARM,)).iloc[0]

    assert row["excess_variance_share"] == pytest.approx(1.0)
    assert row["mae_variance_share"] == pytest.approx(0.0)


def test_the_composition_flags_a_rank_arm_left_on_the_wrong_scale():
    """The failure this guards: a rank head at ~unit spread beside a return-unit
    mae head puts essentially all of the score's variance on one head, which is a
    mae-head ablation wearing E4's name."""

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 4),
        "C_rank_excess_predicted_excess": [1.0, 2.0, 3.0, 4.0],
        "C_rank_excess_predicted_mae": [0.01, 0.02, 0.03, 0.04],
    })
    row = score_composition(frame, 0.25, arms=("C_rank_excess",)).iloc[0]

    assert row["excess_variance_share"] > 0.99


def test_the_composition_uses_the_scaled_mae_term():
    """The mae head enters the score only through ``-penalty * max(-mae, 0)``, so a
    composition that used the raw head would report a share the score never has."""

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 4),
        "B_mse_predicted_excess": [0.01, 0.02, 0.03, 0.04],
        "B_mse_predicted_mae": [-0.01, -0.02, -0.03, -0.04],
    })

    assert score_composition(frame, 0.25, arms=(BASELINE_ARM,)).iloc[0][
        "mae_variance_share"] == pytest.approx(0.25 ** 2 / (1.0 + 0.25 ** 2))
    assert score_composition(frame, 0.0, arms=(BASELINE_ARM,)).iloc[0][
        "mae_variance_share"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# row alignment
# --------------------------------------------------------------------------- #

def test_arm_scores_are_row_aligned_with_the_samples_they_came_from():
    train = _train_frame()
    split = _train_frame(seed=2)
    heads = rank_target_ab._fit_heads(train, {}, 10, 100.0)

    scored = build_arm_scores(split, {}, heads, 10, 0.25)

    assert scored.loc[:, ["date", "symbol"]].equals(
        split.loc[:, ["date", "symbol"]].sort_values(["date", "symbol"]).reset_index(drop=True))
    for arm in ("B_mse", "C_rank_excess", "D_rank_both"):
        assert arm in scored.columns
        assert "{}_predicted_excess".format(arm) in scored.columns
        assert "{}_predicted_mae".format(arm) in scored.columns


def test_arm_scores_do_not_reorder_the_targets():
    """The predictions are handed to ``evaluate_outcome_predictions`` positionally,
    so a reorder would score one row's prediction against another row's label."""

    train = _train_frame()
    split = _train_frame(seed=3)
    heads = rank_target_ab._fit_heads(train, {}, 10, 100.0)

    scored = build_arm_scores(split, {}, heads, 10, 0.25)

    assert scored["future_excess_return_10d"].to_numpy() == pytest.approx(
        split.sort_values(["date", "symbol"])["future_excess_return_10d"].to_numpy())


def test_a_duplicated_key_in_the_published_file_is_rejected(tmp_path, monkeypatch):
    """A duplicated ``(date, symbol)`` would fan the merge out; every downstream row
    count would then be wrong while still looking plausible."""

    samples = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20", "2022-07-21"]),
        "symbol": ["000001.SZ", "000002.SZ"],
    })
    path = tmp_path / "trade_outcome_test_predictions_m.csv"
    pd.concat([samples, samples]).assign(entry_score=0.0,
                                         predicted_excess_return_10d=0.0,
                                         predicted_mae_10d=0.0).to_csv(path, index=False)

    with pytest.raises(Exception):
        attach_published_score(tmp_path, "m", samples, "test")


def test_an_unmatched_row_in_the_published_file_is_named(tmp_path):
    samples = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20", "2022-07-21"]),
        "symbol": ["000001.SZ", "000002.SZ"],
    })
    disk = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"]),
        "symbol": ["000009.SZ"],
        "entry_score": [0.0],
        "predicted_excess_return_10d": [0.0],
        "predicted_mae_10d": [0.0],
    })
    disk.to_csv(tmp_path / "trade_outcome_test_predictions_m.csv", index=False)

    with pytest.raises(ValueError) as excinfo:
        attach_published_score(tmp_path, "m", samples, "test")

    assert "unmatched" in str(excinfo.value)


def test_the_published_score_join_keeps_the_published_column_names(tmp_path):
    samples = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20", "2022-07-21"]),
        "symbol": ["000001.SZ", "000002.SZ"],
    })
    pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20", "2022-07-21"]),
        "symbol": ["000001.SZ", "000002.SZ"],
        "entry_score": [0.01, 0.02],
        "predicted_excess_return_10d": [0.03, 0.04],
        "predicted_mae_10d": [-0.05, -0.06],
    }).to_csv(tmp_path / "trade_outcome_test_predictions_m.csv", index=False)

    merged = attach_published_score(tmp_path, "m", samples, "test")

    # Every arm's head columns share one naming scheme, so the metric loop can
    # address them uniformly.
    assert merged["A_published"].to_numpy() == pytest.approx([0.01, 0.02])
    assert merged["A_published_predicted_excess"].to_numpy() == pytest.approx([0.03, 0.04])
    assert merged["A_published_predicted_mae"].to_numpy() == pytest.approx([-0.05, -0.06])


# --------------------------------------------------------------------------- #
# the retrain anchor
# --------------------------------------------------------------------------- #

def _write_anchor_disk(tmp_path, samples, scores, model_id="m"):
    frame = samples.copy()
    frame["entry_score"] = scores
    frame.to_csv(tmp_path / "trade_outcome_test_predictions_{}.csv".format(model_id), index=False)


def test_the_retrain_anchor_accepts_a_reproduced_score(tmp_path):
    train = _anchor_frame()
    model = {"feature_columns": [], "preprocessor": {}}
    # A constant Ridge prediction is achieved by an all-zero target.
    train = train.assign(future_excess_return_10d=0.0, future_mae_10d=0.0)
    test = _anchor_frame().assign(future_excess_return_10d=0.0, future_mae_10d=0.0)
    _write_anchor_disk(tmp_path, test, [0.0, 0.0])

    delta = retrain_anchor_check(train, test, model, 100.0, 0.25, tmp_path, "m")

    assert delta == pytest.approx(0.0, abs=1e-15)


def test_the_retrain_anchor_rejects_a_score_it_cannot_reproduce(tmp_path):
    train = _anchor_frame()
    model = {"feature_columns": [], "preprocessor": {}}
    _write_anchor_disk(tmp_path, train, [0.5, 0.5])

    with pytest.raises(ValueError) as excinfo:
        retrain_anchor_check(train, train, model, 100.0, 0.25, tmp_path, "m")

    assert "retrain anchor FAILED" in str(excinfo.value)


def test_the_retrain_anchor_rejects_a_different_row_count(tmp_path):
    train = _anchor_frame()
    model = {"feature_columns": [], "preprocessor": {}}
    _write_anchor_disk(tmp_path, train, [0.0, 0.0])
    longer = pd.concat([train, train.iloc[[0]].assign(symbol="000003.SZ")], ignore_index=True)

    with pytest.raises(ValueError) as excinfo:
        retrain_anchor_check(train, longer, model, 100.0, 0.25, tmp_path, "m")

    assert "row count differs" in str(excinfo.value)


def test_the_retrain_anchor_rejects_same_length_different_names(tmp_path):
    """A length-only check would let this through and report a training defect."""

    train = _anchor_frame()
    model = {"feature_columns": [], "preprocessor": {}}
    _write_anchor_disk(tmp_path, train, [0.0, 0.0])
    other = train.assign(symbol=["000001.SZ", "000009.SZ"])

    with pytest.raises(ValueError) as excinfo:
        retrain_anchor_check(train, other, model, 100.0, 0.25, tmp_path, "m")

    assert "keys differ" in str(excinfo.value)


def test_the_retrain_anchor_tolerates_row_order(tmp_path):
    """Row order is presentation; both sides are sorted before the diff."""

    train = _anchor_frame().assign(future_excess_return_10d=0.0, future_mae_10d=0.0)
    model = {"feature_columns": [], "preprocessor": {}}
    _write_anchor_disk(tmp_path, train, [0.0, 0.0])

    delta = retrain_anchor_check(
        train, train.iloc[::-1].reset_index(drop=True), model, 100.0, 0.25, tmp_path, "m")

    assert delta == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# the phase family must be paired before it is averaged
# --------------------------------------------------------------------------- #

def _phase_frame(rows):
    return pd.DataFrame(rows)


def test_paired_phase_excess_subtracts_within_a_weekday():
    frame = _phase_frame([
        {"weekday": 0, "arm": BASELINE_ARM, "total_return": 0.10},
        {"weekday": 0, "arm": "C_rank_excess", "total_return": 0.14},
        {"weekday": 2, "arm": BASELINE_ARM, "total_return": 0.20},
        {"weekday": 2, "arm": "C_rank_excess", "total_return": 0.19},
    ])

    paired = paired_phase_excess(frame, BASELINE_ARM, "C_rank_excess")

    assert paired.to_dict() == pytest.approx({0: 0.04, 2: -0.01})


def test_an_unbalanced_family_does_not_silently_compare_averaged_windows():
    """The failure this guards: with a weekday missing for one arm, subtracting the
    two family means compares a short window against a long one and the phase
    statistic is computed on a number that is part of the answer."""

    frame = _phase_frame([
        {"weekday": 0, "arm": BASELINE_ARM, "total_return": 0.10},
        {"weekday": 1, "arm": BASELINE_ARM, "total_return": 0.50},
        {"weekday": 0, "arm": "D_rank_both", "total_return": 0.20},
    ])

    paired = paired_phase_excess(frame, BASELINE_ARM, "D_rank_both")

    assert paired.to_dict() == pytest.approx({0: 0.10})
    difference_of_means = (
        frame.loc[frame["arm"] == "D_rank_both", "total_return"].mean()
        - frame.loc[frame["arm"] == BASELINE_ARM, "total_return"].mean())
    assert paired.mean() != pytest.approx(difference_of_means)


def test_a_missing_arm_is_named():
    frame = _phase_frame([{"weekday": 0, "arm": BASELINE_ARM, "total_return": 0.10}])

    with pytest.raises(ValueError) as excinfo:
        paired_phase_excess(frame, BASELINE_ARM, "C_rank_excess")

    assert "C_rank_excess" in str(excinfo.value)
