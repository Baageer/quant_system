"""E2's arms must differ only in the score column, and by arm -- not by position.

The ablation is only interpretable if ``A_full`` reproduces the published arm
exactly, so it is read from the disk ``entry_score`` rather than recombined from
the two raw head columns: the raw columns round-trip through CSV, so recomputing
them differs by ~1e-16 and the incumbent would stop being bit-identical to the
published ledger.  Any difference between arms then has exactly one cause.

The other trap is orientation, and it already bit once.  ``predicted_mae_10d`` is
stored as a return and is negative for essentially every row, while the score
*subtracts* ``max(-pred_mae, 0)`` -- so the downside head raises the score the
*shallower* the predicted drawdown, and an arm built as ``-predicted_mae`` ranks
the basket backwards.  A literal-pinning test cannot catch that (it was written
against the wrong literal and passed); the orientation test derives the expected
ordering from the published combination formula instead.

The remaining trap is ordering.  ``fold_excess_table`` returns arms in sorted order
while the grid is built in ``ARMS`` order, so anything that pairs the two by
position silently attributes one arm's folds to another.  That is the failure
mode this file's ordering tests exist for.
"""

import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.two_head_ablation import (
    ARMS,
    BASELINE_ARM,
    basket_downside,
    fold_excess_table,
    load_ablation_scores,
    load_null_total_returns,
)


def make_prediction_frame(rows=None):
    return pd.DataFrame(rows or [
        {
            "date": "2022-07-20", "symbol": "000001.SZ",
            "future_excess_return_10d": 0.01, "future_mae_10d": -0.02,
            "predicted_excess_return_10d": 0.005, "predicted_mae_10d": -0.04,
            # Deliberately NOT predicted_excess - 0.25 * max(-predicted_mae, 0):
            # the loader must return this value untouched, not recompute it.
            "entry_score": 0.123456789, "risk_penalty": 0.25,
        },
        {
            "date": "2022-07-21", "symbol": "000002.SZ",
            "future_excess_return_10d": -0.03, "future_mae_10d": -0.05,
            "predicted_excess_return_10d": -0.002, "predicted_mae_10d": -0.01,
            "entry_score": -0.0045, "risk_penalty": 0.25,
        },
    ])


def write_predictions(tmp_path, frame, model_id="hs300_pit_outcome", split="test"):
    path = tmp_path / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    frame.to_csv(path, index=False)
    return path


def test_a_full_is_the_disk_column_verbatim(tmp_path):
    write_predictions(tmp_path, make_prediction_frame())

    scores = load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")

    assert np.array_equal(
        np.asarray(scores["A_full"], dtype=float),
        np.asarray(scores["entry_score"], dtype=float),
    )


def test_arm_orientation_matches_the_score_convention(tmp_path):
    """Each arm must be order-isomorphic to that head's term in ``entry_score``.

    Pinning a literal (``A_mae == -predicted_mae``) is what let an inverted
    downside arm through: the sign *looks* natural and is backwards, because the
    published score subtracts ``max(-pred_mae, 0)`` and ``predicted_mae_10d`` is
    itself negative.  So assert the property that actually has to hold -- an arm
    must rank rows the same way its head moves the score.
    """

    frame = make_prediction_frame()
    base = frame.iloc[[0]].copy()
    frame = pd.concat([frame, base, base], ignore_index=True)
    frame.loc[0, "predicted_mae_10d"] = -0.04
    frame.loc[1, "predicted_mae_10d"] = -0.01
    frame.loc[2, "predicted_mae_10d"] = -0.09
    frame.loc[3, "predicted_mae_10d"] = 0.02  # a positive prediction: flat term
    write_predictions(tmp_path, frame)

    scores = load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")
    penalty = 0.25

    # The head's contribution to the published score, exactly as combined.
    contribution = -penalty * np.maximum(-np.asarray(scores["predicted_mae_10d"], float), 0.0)
    assert np.array_equal(
        np.argsort(np.asarray(scores["A_mae"], float)),
        np.argsort(contribution),
    )
    # The shallower predicted drawdown must score higher, not lower.
    shallower = np.asarray(scores["predicted_mae_10d"], float) > -0.05
    mae = np.asarray(scores["A_mae"], float)
    assert mae[shallower].max() >= mae[~shallower].min()

    full = np.asarray(scores["A_full"], float)
    excess_contribution = np.asarray(scores["predicted_excess_return_10d"], float)
    # ``A_excess`` alone reproduces that head's term up to the positive factor 1.
    assert np.allclose(np.asarray(scores["A_excess"], float), excess_contribution)
    assert np.argsort(np.asarray(scores["A_excess"], float)).tolist() == \
        np.argsort(excess_contribution).tolist()
    # And the three arms are not accidentally the same column.
    assert not np.array_equal(np.argsort(full), np.argsort(mae))


def test_all_three_arms_are_present(tmp_path):
    write_predictions(tmp_path, make_prediction_frame())

    scores = load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")

    assert set(ARMS) <= set(scores.columns)


def test_a_missing_head_column_is_named_rather_than_defaulted(tmp_path):
    frame = make_prediction_frame().drop(columns=["predicted_mae_10d"])
    write_predictions(tmp_path, frame)

    with pytest.raises(ValueError) as excinfo:
        load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")

    assert "predicted_mae_10d" in str(excinfo.value)


def test_a_row_unusable_by_any_arm_is_dropped_for_all_of_them(tmp_path):
    """Otherwise the per-symbol slot map would be right for one arm and off by a row."""

    frame = make_prediction_frame()
    frame.loc[0, "predicted_mae_10d"] = np.nan
    write_predictions(tmp_path, frame)

    scores = load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")

    assert len(scores) == 1


def test_a_mixed_risk_penalty_is_rejected(tmp_path):
    frame = make_prediction_frame()
    frame.loc[0, "risk_penalty"] = 0.5
    write_predictions(tmp_path, frame)

    with pytest.raises(ValueError) as excinfo:
        load_ablation_scores(tmp_path, "hs300_pit_outcome", "test")

    assert "risk_penalty" in str(excinfo.value)


def test_fold_excess_is_keyed_by_arm_not_by_position():
    """The arms come back sorted; the values must follow the arm name."""

    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    baseline = pd.Series(100.0 * (1.01 ** np.arange(40)), index=dates)
    equity = {
        "A_full": baseline,
        # Tracked the baseline exactly, so its excess is exactly zero in every
        # fold -- and no other arm in this fixture can be.  If the fold columns
        # were paired by position, ``A_excess`` (second alphabetically, first in
        # ARMS order after A_full) would land on a compounding arm's numbers.
        "A_excess": baseline.copy(),
        "A_mae": pd.Series(100.0 * (1.02 ** np.arange(40)), index=dates),
    }
    folds = [
        {"trade_start": dates[0], "trade_end": dates[20]},
        {"trade_start": dates[20], "trade_end": dates[-1] + pd.Timedelta(days=1)},
    ]

    table = fold_excess_table(equity, folds, 100.0, baseline_arm="A_full").set_index("arm")

    assert table.loc["A_excess", "fold_0"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["A_excess", "fold_1"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["A_mae", "fold_0"] > 0
    assert table.loc["A_mae", "fold_1"] > 0


def test_the_baseline_arm_is_the_published_combination():
    assert BASELINE_ARM == "A_full"


def test_basket_downside_separates_the_basket_from_the_universe():
    scores = pd.DataFrame({
        "future_excess_return_10d": [0.10, 0.02],
        "future_mae_10d": [-0.01, -0.09],
    })

    measured = basket_downside(scores, pd.Series([True, False]), 10)

    assert measured["selected_mean_future_mae"] == pytest.approx(-0.01)
    assert measured["universe_mean_future_mae"] == pytest.approx(-0.05)
    assert measured["selected_mean_future_excess"] == pytest.approx(0.10)


def write_null(tmp_path, model_id, tag, top_k, rows, split=None, returns=(0.01, 0.02)):
    pd.DataFrame({"total_return": returns}).to_csv(
        tmp_path / "random_portfolio_{}_{}_draws.csv".format(model_id, tag), index=False)
    summary = {"model_id": model_id, "top_k": top_k, "test_rows": rows}
    if split is not None:
        summary["split"] = split
    (tmp_path / "random_portfolio_{}_{}_summary.json".format(model_id, tag)).write_text(
        json.dumps(summary), encoding="utf-8")


def test_the_null_is_returned_when_it_fits(tmp_path):
    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    null = load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert list(null) == [0.01, 0.02]


def test_a_missing_null_is_named(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert "hs300_pit_outcome" in str(excinfo.value)


def test_a_null_at_another_width_is_rejected(tmp_path):
    """The tags are not uniform, and a same-window null can still be the wrong width."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.10, 218898)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert "top_k" in str(excinfo.value)


def test_a_null_drawn_on_another_window_is_rejected(tmp_path):
    """HS300's forward-2026 null is also top_k=0.05; only the row count separates them."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 36767)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert "test_rows" in str(excinfo.value)


def test_a_validation_split_null_is_rejected(tmp_path):
    write_null(tmp_path, "zz1000_pit_outcome", "zz1000_pit", 0.1, 543025, split="validation")

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "zz1000_pit_outcome", "zz1000_pit", 0.1, 543025)

    assert "split" in str(excinfo.value)
