"""The guards that keep step 1's scores attached to the right rows.

Step 1 hands the production chain a score *column* rather than a score vector, so
the thing that can go wrong silently is alignment: the panel this module refits
and the predictions file it backtests are two separate reads of the same window,
and a one-row slip would score every name against its neighbour's signal without
raising anything.  Two guards exist for that, plus the drift check that keeps the
baseline arm equal to the incumbent the gates difference against.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.cross_sectional_representation_backtest import (
    attach_variant_scores,
    fold_columns,
    fold_win_counts,
    phase_mean_equity,
)


def build_rows(periods=4, symbols=3, entry_score=0.0):
    dates = pd.date_range("2021-01-04", periods=periods, freq="B")
    return pd.DataFrame({
        "date": np.repeat(dates, symbols),
        "symbol": np.tile(["S0001.SZ", "S0002.SZ", "S0003.SZ"], periods),
        "entry_score": np.full(periods * symbols, float(entry_score)),
    })


def build_arm(value):
    """A stand-in arm whose ``entry_score`` is a constant, so drift is detectable."""

    class _Model:
        def __init__(self, constant):
            self.constant = constant

        def predict(self, design):
            return np.full(len(design), self.constant)

    return {
        "variant": "raw",
        "feature_columns": ["x1"],
        "preprocessor": {"feature_columns": ["x1"], "medians": np.zeros(1),
                         "means": np.zeros(1), "scales": np.ones(1)},
        "excess_model": _Model(value),
        "mae_model": _Model(0.0),
    }


def test_scores_are_attached_when_the_rows_line_up():
    stored = build_rows()
    panel = stored.loc[:, ["date", "symbol"]].assign(x1=0.0, future_excess_return_10d=0.0,
                                                     future_mae_10d=0.0, auxiliary_label_eligible="true")

    frame = attach_variant_scores(stored, panel, {"raw": build_arm(0.0)}, risk_penalty=0.0)

    assert "raw" in frame.columns
    assert np.allclose(frame["entry_score"], frame["raw"])


def test_a_shifted_panel_is_refused_rather_than_scored():
    stored = build_rows()
    panel = stored.loc[:, ["date", "symbol"]].assign(x1=0.0, future_excess_return_10d=0.0,
                                                     future_mae_10d=0.0, auxiliary_label_eligible="true")
    panel = panel.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError) as excinfo:
        attach_variant_scores(stored, panel, {"raw": build_arm(0.0)}, risk_penalty=0.0)

    assert "order" in str(excinfo.value)


def test_a_panel_of_a_different_length_is_refused():
    stored = build_rows()
    panel = stored.iloc[:-1].loc[:, ["date", "symbol"]].assign(
        x1=0.0, future_excess_return_10d=0.0, future_mae_10d=0.0, auxiliary_label_eligible="true")

    with pytest.raises(ValueError) as excinfo:
        attach_variant_scores(stored, panel, {"raw": build_arm(0.0)}, risk_penalty=0.0)

    assert "rows" in str(excinfo.value)


def test_a_raw_arm_that_drifts_from_the_published_score_is_refused():
    """The gates difference every candidate against the raw arm, so a raw arm
    that is not the incumbent silently turns the baseline into a stranger."""

    stored = build_rows()
    panel = stored.loc[:, ["date", "symbol"]].assign(x1=0.0, future_excess_return_10d=0.0,
                                                     future_mae_10d=0.0, auxiliary_label_eligible="true")

    with pytest.raises(ValueError) as excinfo:
        attach_variant_scores(stored, panel, {"raw": build_arm(5.0)}, risk_penalty=0.0)

    assert "reproduce" in str(excinfo.value)


def test_phase_mean_equity_averages_the_weekday_family():
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    family = {0: pd.Series([100.0, 100.0, 100.0], index=dates),
              4: pd.Series([200.0, 200.0, 200.0], index=dates)}

    assert list(phase_mean_equity(family)) == [150.0, 150.0, 150.0]


def test_an_empty_phase_family_is_refused():
    with pytest.raises(ValueError):
        phase_mean_equity({})


def test_fold_columns_use_the_gates_own_prefix_rule():
    """Deliberately the *loose* rule: ``select_configuration`` counts folds with
    a bare ``startswith("fold_")``, so a tidier helper here would report a
    different denominator than the one the gate divided by."""

    table = pd.DataFrame({"arm": ["a"], "fold_0": [0.1], "fold_11": [0.2],
                          "fold_excess_min": [0.0], "phase_excess_min": [1.0]})

    assert fold_columns(table) == ["fold_0", "fold_11", "fold_excess_min"]
    assert fold_columns(pd.DataFrame({"arm": ["a"], "excess": [0.0]})) == []


def test_fold_wins_a_strict_win_only_and_are_reported_per_arm():
    """The baseline's own row is identically zero, so it wins nothing -- which is
    why gate 0 (``excess > 0``) has to be a separate gate."""

    table = pd.DataFrame({
        "arm": ["raw", "demean"],
        "fold_0": [0.0, 0.01],
        "fold_1": [0.0, 0.00],
        "fold_2": [0.0, -0.01],
    })

    assert fold_win_counts(table) == {"raw": 0, "demean": 1}


def test_fold_win_counts_refuse_a_table_without_fold_columns():
    with pytest.raises(ValueError) as excinfo:
        fold_win_counts(pd.DataFrame({"arm": ["raw"], "excess": [0.0]}))

    assert "fold_*" in str(excinfo.value)
