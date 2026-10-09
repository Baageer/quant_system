"""P0-4's two load-bearing pieces: the statistic, and the threshold rule.

The statistic has one silent failure worth pinning -- a fold that cannot be
measured is counted as a **loss**, matching what the gate does with NaN, so an
arm scored on 26 usable folds out of 29 must say so rather than quietly reading
as "26/29".

The threshold rule is a *selection* input, the category of code that has gone
wrong silently four times in this repo.  The test that matters is that it reads
only the null: it takes no candidate argument at all, so the strongest available
statement is that its output is a property of the null multiset -- invariant to
draw order, and equal to hand-computed binomial tails on an exact reference.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.fold_gate_calibration import (
    _phase_mean_equity,
    _win_counts,
    calibrate_fold_threshold,
    fold_agreement,
    read_grid_agreements,
)


def test_the_baseline_never_agrees_with_itself_on_any_fold():
    """This is what makes the baseline's own row useless to the gate, and why
    gate 0 (``excess > 0``) is separate rather than folded into this one."""

    folds = [0.01, -0.02, 0.03, 0.0, 0.04]

    scored = fold_agreement(folds, folds)

    assert scored["wins"] == 0.0
    assert scored["agreement"] == 0.0


def test_an_unmeasurable_fold_counts_as_a_loss_and_is_reported():
    arm = [0.05, float("nan"), 0.05, 0.05]
    baseline = [0.01, 0.01, 0.01, 0.01]

    scored = fold_agreement(arm, baseline)

    # 3 wins out of 4 folds, one of which could not be measured.
    assert scored["wins"] == 3.0
    assert scored["folds"] == 4.0
    assert scored["unscored_folds"] == 1.0
    assert scored["agreement"] == 0.75


def test_a_tie_is_not_a_win():
    """``> 0`` is strict in the gate, so equality must not score."""

    assert fold_agreement([0.02, 0.01], [0.01, 0.01])["wins"] == 1.0


def test_mismatched_fold_vectors_are_refused():
    with pytest.raises(ValueError) as excinfo:
        fold_agreement([0.01, 0.02], [0.01])

    assert "length" in str(excinfo.value)


def test_an_empty_fold_vector_is_refused():
    with pytest.raises(ValueError):
        fold_agreement([], [])


def test_agreements_that_are_not_multiples_of_one_over_folds_are_refused():
    """The inverse of ``wins / folds`` is only defined on that lattice; anything
    else was measured on a different fold count and would round to a plausible
    vector."""

    with pytest.raises(ValueError) as excinfo:
        _win_counts(np.array([0.5, 0.31]), 12)

    assert "multiples" in str(excinfo.value)


def binomial_reference(folds):
    """An exact ``Binomial(folds, 0.5)`` sample: k repeated C(folds, k) times."""

    from math import comb

    return np.repeat(np.arange(folds + 1), [comb(folds, k) for k in range(folds + 1)])


def test_the_rule_reproduces_the_binomial_tail_it_is_built_on():
    """For ``Binomial(11, 0.5)`` the tails are C(11,k)+...+/2048, so at alpha=0.05
    the smallest k with a tail at or below it is 9 (0.0327; k=8 gives 0.1133)."""

    calibration = calibrate_fold_threshold(
        binomial_reference(11).astype(float) / 11.0, 11, alpha=0.05, bootstrap=0)

    assert calibration["theta"] == pytest.approx(9 / 11)


def test_the_rule_takes_the_smallest_qualifying_k_not_the_largest():
    """At alpha=0.001 only k=11 qualifies (1/2048); a rule that scanned
    downwards would return something else on the alpha=0.05 case above."""

    tight = calibrate_fold_threshold(
        binomial_reference(11).astype(float) / 11.0, 11, alpha=0.001, bootstrap=0)

    assert tight["theta"] == pytest.approx(1.0)
    assert tight["theta_is_reachable"] is True


def test_a_null_that_always_wins_makes_the_statistic_unusable():
    """Every draw winning every fold means even k=folds exceeds alpha: the
    finding is about the *statistic*, so theta must go past 1 rather than be
    clamped to it."""

    calibration = calibrate_fold_threshold(np.ones(50), 29, alpha=0.05, bootstrap=0)

    assert calibration["theta"] == pytest.approx(30 / 29)
    assert calibration["theta_is_reachable"] is False


def test_the_null_pass_rate_at_theta_is_at_or_below_alpha():
    values = binomial_reference(11).astype(float) / 11.0

    calibration = calibrate_fold_threshold(values, 11, alpha=0.05, bootstrap=0)

    assert calibration["null_pass_rate_at_theta"] <= 0.05


def test_theta_is_a_property_of_the_null_multiset_not_its_order():
    values = binomial_reference(11).astype(float) / 11.0
    shuffled = np.random.default_rng(7).permutation(values)

    assert (calibrate_fold_threshold(values, 11, bootstrap=0)["theta"]
            == calibrate_fold_threshold(shuffled, 11, bootstrap=0)["theta"])


def test_the_reported_threshold_is_the_conservative_end_of_the_bootstrap():
    values = binomial_reference(11).astype(float) / 11.0

    calibration = calibrate_fold_threshold(values, 11, alpha=0.05, bootstrap=200)

    assert calibration["theta_bootstrap_upper"] >= calibration["theta"]


def test_the_bootstrap_spread_is_reported_beside_the_crude_maximum():
    """The pre-registered bound is ``max`` over replications, which on a discrete
    statistic is deliberately crude; the percentile is carried alongside so the
    two can be compared rather than confusing one for the other."""

    values = binomial_reference(11).astype(float) / 11.0

    calibration = calibrate_fold_threshold(values, 11, alpha=0.05, bootstrap=200)

    assert calibration["theta"] <= calibration["theta_bootstrap_p975"]
    assert calibration["theta_bootstrap_p975"] <= calibration["theta_bootstrap_upper"]


def test_the_default_threshold_is_priced_against_the_configured_one():
    """The report has to carry the size of the *configured* gate next to the
    calibrated one, or the comparison cannot be made from the output alone."""

    calibration = calibrate_fold_threshold(
        binomial_reference(11).astype(float) / 11.0, 11, alpha=0.05, bootstrap=0)

    assert calibration["null_pass_rate_at_configured"] > 0


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError) as excinfo:
        calibrate_fold_threshold(np.array([0.5]), 2, alpha=0.0, bootstrap=0)

    assert "alpha" in str(excinfo.value)


def test_the_variance_ratio_reports_over_dispersion():
    """The second pre-registered output: how much of the 29-fold spread is
    independent information rather than one bet repeated."""

    concentrated = np.full(50, 14.0) / 29.0
    spread = np.repeat(np.arange(12, 17), 10).astype(float) / 29.0

    tight = calibrate_fold_threshold(concentrated, 29, bootstrap=0)
    wide = calibrate_fold_threshold(spread, 29, bootstrap=0)

    assert tight["null_variance_ratio_vs_binomial"] < wide["null_variance_ratio_vs_binomial"]


def test_phase_mean_equity_averages_the_family():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    by_weekday = {0: pd.Series([200.0, 200.0, 200.0], index=dates),
                  1: pd.Series([100.0, 100.0, 100.0], index=dates)}

    averaged = _phase_mean_equity(by_weekday)

    assert list(averaged) == [150.0, 150.0, 150.0]


def test_grid_agreements_are_absent_when_e6_has_not_been_run(tmp_path):
    assert read_grid_agreements(tmp_path, "hs300", 29) is None


def test_grid_agreements_refuse_a_fold_count_they_were_not_measured_on(tmp_path):
    pd.DataFrame({"arm": ["tk0.05_mp20"], "fold_0": [0.1], "fold_1": [0.2]}).to_csv(
        tmp_path / "basket_width_hs300_validation_grid.csv", index=False)

    with pytest.raises(ValueError) as excinfo:
        read_grid_agreements(tmp_path, "hs300", 29)

    assert "fold columns" in str(excinfo.value)


def test_grid_agreements_are_read_as_positive_fold_counts(tmp_path):
    pd.DataFrame({
        "arm": ["a", "b"], "fold_0": [0.1, -0.1], "fold_1": [0.2, 0.2], "fold_2": [0.3, 0.3],
    }).to_csv(tmp_path / "basket_width_zz1000_validation_grid.csv", index=False)

    agreements = read_grid_agreements(tmp_path, "zz1000", 3)

    assert list(agreements) == [1.0, 2.0 / 3.0]
