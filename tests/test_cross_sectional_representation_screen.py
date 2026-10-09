"""The two claims the revised E9 step 0 rests on, plus the screen's own pieces.

The load-bearing claims are both negative, and both are the kind that would
otherwise be discovered after a run rather than before one:

1. **The pre-registered feature-level screen cannot see the representation.**
   Cross-sectional demean / z-score / rank are strictly increasing within date,
   so every daily rank -- and with it the whole IC / quantile-monotonicity ladder
   -- is unchanged.  ``test_the_whole_rank_ladder_is_blind_to_the_representation``
   is that statement as an executable fact.
2. **The pre-registered proof that demeaning is an identity only holds for
   frozen weights.**  The plan exempted variant A from testing on algebraic
   grounds; a *refit* on the demeaned panel is a different model, because the
   design matrix changed.  ``test_frozen_weights_are_rank_invariant_but_a_refit_is_not``
   pins both halves -- the identity where it is real, and its failure where it
   matters.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.cross_sectional_representation_screen import (
    apply_representation,
    check_fidelity,
    daily_ic_series,
    demean_within_date,
    fit_arm,
    half_window_deltas,
    load_model_settings,
    paired_ic_delta,
    screen_decision,
    score_rank_correlation,
    top_k_overlap,
)
from research.market_regime.feature_study_core import daily_rank_ics, quantile_group_returns


def build_panel(seed=11, dates=6, symbols=30, features=("x1", "x2", "x3")):
    # 30 symbols per date on purpose: the ladder's own minimums (20 paired rows
    # for a daily IC, 5x the quantile count for the group table) would silently
    # drop every day of a smaller panel, and an empty comparison passes.

    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=dates, freq="B")
    rows = []
    for day, date in enumerate(index):
        # A shared per-date level shift makes demeaning a real change to the
        # design matrix, not just a relabelling.
        level = day * 3.0
        for symbol in range(symbols):
            value = level + rng.normal(size=len(features))
            row = {"date": date, "symbol": "S{:04d}.SZ".format(symbol)}
            row.update({name: value[i] for i, name in enumerate(features)})
            row["future_excess_return_10d"] = value[0] - value[1] + rng.normal(scale=0.3)
            row["future_mae_10d"] = -abs(value[2]) - 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_whole_rank_ladder_is_blind_to_the_representation():
    """Mean IC, ICIR, win rate, win count, observation count, quantile spread and
    monotonicity are all identical under demean / z-score / rank.

    This is why the pre-registered step 0 could not have closed E9: whatever it
    reported for the transforms would have been the raw feature's own numbers,
    read as evidence for a transform that never entered the calculation.
    """

    panel = build_panel()
    features = ["x1", "x2", "x3"]
    actual = panel["future_excess_return_10d"].to_numpy(dtype=float)

    for variant in ("demean", "zscore", "rank"):
        represented = apply_representation(panel, features, variant)
        for feature in features:
            raw = panel[feature].to_numpy(dtype=float)
            moved = represented[feature].to_numpy(dtype=float)
            np.testing.assert_allclose(daily_rank_ics(panel["date"], actual, raw),
                                       daily_rank_ics(panel["date"], actual, moved),
                                       rtol=0, atol=1e-12)
            raw_group = quantile_group_returns(panel["date"], actual, raw)
            moved_group = quantile_group_returns(panel["date"], actual, moved)
            assert raw_group["top_bottom_spread"] == pytest.approx(moved_group["top_bottom_spread"], abs=1e-12)
            assert raw_group["monotonic_corr"] == pytest.approx(moved_group["monotonic_corr"], abs=1e-12)


def test_frozen_weights_are_rank_invariant_but_a_refit_is_not():
    """The identity the pre-registration proved is real -- for fixed weights.

    Applied to a demeaned panel with the *raw* coefficients, every date shifts by
    the same constant and nothing is reordered.  Refitting on the demeaned panel
    changes the coefficients, which is the case E9 actually proposes and the case
    the plan wrongly treated as a no-op.
    """

    panel = build_panel(seed=5)
    features = ["x1", "x2", "x3"]
    raw = panel[features].to_numpy(dtype=float)
    demeaned = apply_representation(panel, features, "demean")[features].to_numpy(dtype=float)
    actual = panel["future_excess_return_10d"].to_numpy(dtype=float)

    weights = np.array([1.0, -2.0, 0.5])
    frozen_raw = raw @ weights
    frozen_demeaned = demeaned @ weights
    for _, group in panel.groupby("date"):
        assert list(pd.Series(frozen_raw[group.index]).rank()) == list(pd.Series(frozen_demeaned[group.index]).rank())

    sklearn_ridge = pytest.importorskip("sklearn.linear_model").Ridge
    refit_raw = sklearn_ridge(alpha=1.0).fit(raw, actual).coef_
    refit_demeaned = sklearn_ridge(alpha=1.0).fit(demeaned, actual).coef_
    assert not np.allclose(refit_raw, refit_demeaned, atol=1e-8)


def test_the_representation_actually_centres_and_scales_each_date():
    panel = build_panel()
    features = ["x1", "x2"]

    demeaned = apply_representation(panel, features, "demean")
    assert demeaned.groupby("date")[features].mean().abs().to_numpy().max() < 1e-12

    scored = apply_representation(panel, features, "zscore")
    assert scored.groupby("date")[features].mean().abs().to_numpy().max() < 1e-12
    np.testing.assert_allclose(scored.groupby("date")[features].std().to_numpy(), 1.0, atol=1e-9)

    ranked = apply_representation(panel, features, "rank")
    grouped = ranked.groupby("date")[features]
    assert grouped.max().to_numpy().min() == pytest.approx(1.0)
    assert grouped.min().to_numpy().max() > 0.0


def test_the_raw_variant_is_a_copy_not_an_alias():
    panel = build_panel()

    represented = apply_representation(panel, ["x1"], "raw")
    represented.loc[0, "x1"] = 999.0

    assert panel.loc[0, "x1"] != 999.0


def test_an_unknown_representation_is_refused():
    with pytest.raises(ValueError) as excinfo:
        apply_representation(build_panel(), ["x1"], "centre")

    assert "variant" in str(excinfo.value)


def test_demeaning_a_design_matrix_preserves_each_dates_ranking():
    """``demean_post`` is the same identity one level later, applied to the
    standardized matrix, so it must not reorder anything either."""

    panel = build_panel()
    matrix = panel[["x1", "x2"]].to_numpy(dtype=float)

    centred = demean_within_date(matrix, panel["date"])

    np.testing.assert_allclose(pd.DataFrame(centred).groupby(panel["date"].to_numpy()).mean().to_numpy(),
                               0.0, atol=1e-12)
    for _, group in panel.groupby("date"):
        left = pd.Series(matrix[group.index, 0]).rank().to_numpy()
        right = pd.Series(centred[group.index, 0]).rank().to_numpy()
        np.testing.assert_array_equal(left, right)


def test_the_post_standardisation_control_leaves_the_penalty_scale_alone():
    """``demean`` rescales every column, so Ridge's penalty sees a different
    problem; ``demean_post`` is the control that holds that scale fixed."""

    panel = build_panel()
    features = ["x1", "x2", "x3"]

    raw = fit_arm(panel, features, "raw", alpha=100.0)
    demean = fit_arm(panel, features, "demean", alpha=100.0)
    demean_post = fit_arm(panel, features, "demean_post", alpha=100.0)

    from research.experiments.cross_sectional_representation_screen import _scale_ratio

    assert _scale_ratio(raw, demean_post) == pytest.approx(1.0)
    assert _scale_ratio(raw, demean) != pytest.approx(1.0)


def test_the_top_k_overlap_separates_identical_from_disjoint_baskets():
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    reference = np.tile(np.arange(10.0), 3)
    dates = np.repeat(dates.to_numpy(), 10)

    assert top_k_overlap(dates, reference, reference, top_k=0.2) == pytest.approx(1.0)

    flipped = -reference
    assert top_k_overlap(dates, reference, flipped, top_k=0.2) == pytest.approx(0.0)

    # One of the top two (index 9) is shared, the other (index 8) is displaced by
    # index 7 -- so exactly half the basket survives.
    partly = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 20, 8, 9], dtype=float), 3)
    assert top_k_overlap(dates, reference, partly, top_k=0.2) == pytest.approx(0.5)


def test_paired_deltas_are_taken_day_by_day_not_between_means():
    """Two arms whose IC *means* are equal but whose days differ must not read as
    a zero effect -- the screen compares the same day twice."""

    reference = pd.Series([0.02, 0.04, 0.06, 0.08], index=pd.date_range("2021-01-04", periods=4, freq="B"))
    candidate = pd.Series([0.04, 0.06, 0.04, 0.06], index=reference.index)

    paired = paired_ic_delta(reference, candidate)

    assert paired["mean_delta"] == pytest.approx(0.0)
    assert paired["days"] == 4
    halves = half_window_deltas(reference, candidate)
    assert halves["first_half_delta"] > 0 > halves["second_half_delta"]


def test_an_arm_that_never_moves_its_basket_cannot_advance():
    """Even a large IC delta is worthless if the engine would buy the same names:
    the portfolio is the basket, so the screen must block on that alone."""

    decision = screen_decision(mean_delta=0.05, first_half_delta=0.05, second_half_delta=0.05,
                               overlap=1.0)

    assert decision["advances"] is False
    assert decision["checks"]["basket_moves"] is False


def test_a_gain_that_only_holds_in_one_half_is_blocked():
    decision = screen_decision(mean_delta=0.05, first_half_delta=0.06, second_half_delta=-0.01,
                               overlap=0.5)

    assert decision["advances"] is False
    assert decision["checks"]["both_halves"] is False


def test_a_positive_but_small_delta_is_blocked():
    decision = screen_decision(mean_delta=0.001, first_half_delta=0.01, second_half_delta=0.01,
                               overlap=0.5)

    assert decision["advances"] is False
    assert decision["checks"]["ic_delta"] is False


def test_a_variant_clearing_every_check_advances():
    decision = screen_decision(mean_delta=0.02, first_half_delta=0.02, second_half_delta=0.02,
                               overlap=0.5)

    assert decision["advances"] is True
    assert all(decision["checks"].values())


def test_a_nan_delta_cannot_advance():
    decision = screen_decision(mean_delta=float("nan"), first_half_delta=float("nan"),
                               second_half_delta=float("nan"), overlap=0.5)

    assert decision["advances"] is False


def test_constant_days_are_dropped_from_the_ic_series():
    dates = pd.to_datetime(["2021-01-04"] * 4 + ["2021-01-05"] * 4)

    series = daily_ic_series(dates, np.array([1.0, 2.0, 3.0, 4.0] * 2),
                             np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0, 9.0, 9.0]))

    assert len(series) == 1
    assert series.index[0] == pd.Timestamp("2021-01-04")


def test_score_rank_correlation_is_one_for_an_arm_against_itself():
    panel = build_panel()
    score = panel["x1"].to_numpy(dtype=float)

    assert score_rank_correlation(panel["date"], score, score) == pytest.approx(1.0)


def test_model_settings_come_from_the_manifest(tmp_path):
    (tmp_path / "trade_outcome_manifest_demo.json").write_text(
        '{"feature_columns": ["a", "b"], "selected_alpha": 10.0, "selected_risk_penalty": 0.25,'
        ' "top_k": 0.05, "dataset_id": "excess_demo_v1"}')

    settings = load_model_settings(tmp_path, "demo")

    assert settings["feature_columns"] == ["a", "b"]
    assert settings["alpha"] == 10.0
    assert settings["risk_penalty"] == 0.25


def test_the_fidelity_check_reports_zero_for_a_faithful_refit(tmp_path):
    panel = build_panel()
    scores = {"predicted_excess": panel["x1"].to_numpy(dtype=float),
              "predicted_mae": panel["x2"].to_numpy(dtype=float),
              "entry_score": panel["x3"].to_numpy(dtype=float)}
    stored = panel.loc[:, ["date", "symbol"]].assign(
        predicted_excess_return_10d=scores["predicted_excess"],
        predicted_mae_10d=scores["predicted_mae"],
        entry_score=scores["entry_score"])
    stored.to_csv(tmp_path / "predictions.csv", index=False)

    result = check_fidelity({}, panel, scores, tmp_path / "predictions.csv")

    assert result["overlap_rows"] == len(panel)
    # The stored file round-trips through CSV, so "identical" is at text precision.
    assert result["max_abs_diff_entry_score"] == pytest.approx(0.0, abs=1e-12)


def test_the_fidelity_check_reports_a_misaligned_symbol(tmp_path):
    panel = build_panel()
    scores = {"predicted_excess": panel["x1"].to_numpy(dtype=float),
              "predicted_mae": panel["x2"].to_numpy(dtype=float),
              "entry_score": panel["x3"].to_numpy(dtype=float)}
    stored = panel.loc[:, ["date", "symbol"]].assign(
        predicted_excess_return_10d=scores["predicted_excess"] + 1.0,
        predicted_mae_10d=scores["predicted_mae"],
        entry_score=scores["entry_score"])
    stored.loc[0, "symbol"] = "S9999.SZ"
    stored.to_csv(tmp_path / "predictions.csv", index=False)

    result = check_fidelity({}, panel, scores, tmp_path / "predictions.csv")

    assert result["overlap_rows"] == len(panel) - 1
    assert result["max_abs_diff_entry_score"] == pytest.approx(0.0, abs=1e-12)
    assert result["max_abs_diff_predicted_excess_return_10d"] == pytest.approx(1.0)
