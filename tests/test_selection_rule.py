"""Each gate of the selection rule must be able to reject on its own.

The rule exists because ``sort_values(...).iloc[0]`` selected a 22-trade
degenerate arm on the hysteresis grid (test -35.39%) and a below-baseline arm on
the turnover grid.  A gate that cannot be shown to reject anything is not a
gate, so each test below puts exactly one candidate in front of exactly one gate
and checks that it dies there -- and that the row is *kept out* rather than
merely ranked lower.

The phase gate gets an extra test of its own
(:func:`test_phase_gate_is_paired_and_not_biased_against_candidates`): an early
version compared the candidate's worst phase against the baseline's *average*
and rejected an arm that beat the baseline on every single phase.  That bug is
worth a regression test because it fails silently in the direction of "no
change", which looks conservative rather than broken.
"""

import pandas as pd
import pytest

from research.market_regime.selection_rule import (
    fold_returns_from_equity,
    phase_excess_min,
    phase_excess_table,
    select_configuration,
)


BASELINE = {"arm": "baseline"}


def make_grid():
    """Four ways to fail plus one clean winner."""

    return pd.DataFrame([
        {"arm": "baseline", "total_return": 1.00, "turnover_multiple": 4.0},
        {"arm": "good", "total_return": 1.30, "turnover_multiple": 3.0},
        {"arm": "thin", "total_return": 1.50, "turnover_multiple": 0.4},
        {"arm": "fragile", "total_return": 1.40, "turnover_multiple": 3.2},
        {"arm": "worse", "total_return": 0.80, "turnover_multiple": 3.0},
    ])


def make_phase_excess(values=None):
    """Paired worst-case phase excess, keyed by arm.

    ``fragile`` wins on the phase average but loses on its worst paired phase;
    ``thin`` and ``good`` clear it.
    """

    values = values or {"baseline": 0.0, "good": 0.02, "thin": 0.05,
                        "fragile": -0.04, "worse": -0.30}
    return pd.DataFrame([{"arm": arm, "phase_excess_min": value}
                         for arm, value in values.items()])


def row_for(outcome, arm):
    return outcome.table[outcome.table["arm"] == arm].iloc[0]


def test_clean_winner_is_chosen():
    outcome = select_configuration(
        make_grid(), baseline_key=BASELINE, phase_excess=make_phase_excess())
    assert outcome.arm == "good"
    assert outcome.changed
    # Phase and turnover ran; folds were not supplied and must be reported as
    # skipped rather than as passed.
    assert outcome.gates == {"phase": True, "turnover": True, "folds": False}
    assert "gates not run: folds" in outcome.reason


def test_turnover_floor_rejects_the_degenerate_arm():
    """The 22-trade arm: best return on the grid, barely trades at all."""

    outcome = select_configuration(
        make_grid(), baseline_key=BASELINE, phase_excess=make_phase_excess())
    thin = row_for(outcome, "thin")
    assert thin["excess"] > row_for(outcome, "good")["excess"]  # it wins on return
    assert not thin["passes_turnover"]
    assert thin["reject_reason"] == "turnover"


def test_phase_gate_rejects_an_arm_that_only_wins_on_average():
    outcome = select_configuration(
        make_grid(), baseline_key=BASELINE, phase_excess=make_phase_excess())
    fragile = row_for(outcome, "fragile")
    assert fragile["excess"] > 0
    assert fragile["phase_excess_min"] < 0
    assert not fragile["passes_phase"]
    assert fragile["reject_reason"] == "phase"


def test_phase_gate_is_paired_and_not_biased_against_candidates():
    """Regression: an arm that beats the baseline on *every* phase must pass.

    The baseline here is the lucky member of its own family (1.00 against a
    0.7833 average), and the candidate's worst phase (0.71) sits below that
    average while still beating the baseline on its own worst phase (0.65).  An
    unpaired gate -- "candidate's worst vs baseline's average" -- rejects it.
    """

    baseline_phases = [1.00, 0.70, 0.65]
    candidate_phases = [1.05, 0.80, 0.71]
    assert min(candidate_phases) < sum(baseline_phases) / 3.0, "fixture must fail an unpaired gate"

    assert phase_excess_min(candidate_phases, baseline_phases) == pytest.approx(0.05)
    # Pairing is by rank, so permuting either family cannot change the answer.
    assert phase_excess_min(candidate_phases[::-1], baseline_phases) == pytest.approx(0.05)

    frame = phase_excess_table(
        pd.DataFrame([{"arm": "cand", "total_return": value} for value in candidate_phases]),
        pd.DataFrame([{"total_return": value} for value in baseline_phases]),
        key_columns=("arm",))
    assert frame["phase_excess_min"].iloc[0] == pytest.approx(0.05)
    assert frame["phases"].iloc[0] == 3


def test_phase_excess_must_be_paired_data_not_a_column_name():
    """The gate takes differences, so absolute levels must not be accepted."""

    grid = make_grid()
    grid["phase_min"] = [1.0, 1.1, 1.4, 0.9, 0.7]  # absolute levels, the old shape
    with pytest.raises(ValueError, match="paired"):
        select_configuration(grid, baseline_key=BASELINE,
                             phase_excess=grid[["arm", "phase_min"]])
    # Declared as absent, the rule still runs -- and says the gate did not run.
    outcome = select_configuration(grid, baseline_key=BASELINE)
    assert outcome.gates["phase"] is False
    assert "gates not run: phase" in outcome.reason
    # Without the phase gate ``fragile`` wins, which is why the outcome has to
    # name the gates it skipped.
    assert outcome.arm == "fragile"


def test_fold_gate_rejects_a_candidate_that_flips_sign():
    """The "same sign across splits" gate, on a candidate that wins 3 of 4 folds."""

    grid = make_grid()
    folds = pd.DataFrame([
        {"arm": arm, "fold_0": values[0], "fold_1": values[1],
         "fold_2": values[2], "fold_3": values[3]}
        for arm, values in {
            "baseline": (0.0, 0.0, 0.0, 0.0),
            "good": (0.05, 0.06, -0.02, 0.07),
        }.items()
    ])

    strict = select_configuration(grid, baseline_key=BASELINE, fold_excess=folds)
    assert strict.row is None, "a fold losing to the baseline must stop selection"
    assert "folds" in strict.reason
    assert row_for(strict, "good")["fold_agreement"] == pytest.approx(0.75)

    relaxed = select_configuration(
        grid, baseline_key=BASELINE, fold_excess=folds, min_fold_agreement=0.75)
    assert relaxed.arm == "good"


def test_nothing_beats_the_baseline_means_change_nothing():
    """The old selector always returned a row; this one is allowed to decline."""

    grid = make_grid()
    grid.loc[grid["arm"] != "baseline", "total_return"] = 0.5
    outcome = select_configuration(grid, baseline_key=BASELINE)
    assert outcome.row is None
    assert not outcome.changed
    assert row_for(outcome, "baseline")["reject_reason"] == "baseline"


def test_equal_to_the_baseline_is_not_beating_it():
    """Ties are not improvements -- otherwise the grid adopts the baseline as a win."""

    grid = make_grid()
    # Every other arm is clearly worse, so ``good`` is the only near contender.
    grid.loc[grid["arm"].isin(["thin", "fragile", "worse"]), "total_return"] = 0.5
    grid.loc[grid["arm"] == "good", "total_return"] = 1.00
    outcome = select_configuration(grid, baseline_key=BASELINE)
    assert outcome.row is None, "excess must be strictly positive"
    assert row_for(outcome, "good")["reject_reason"] == "beats_baseline"


def test_each_gate_is_blamed_for_its_own_condition():
    """A rejection must name the gate that actually stopped it.

    On the real turnover grid every candidate failed the level gate while some
    of them cleared the phase floor, and the reason string read
    ``beats_baseline+phase`` -- as if the phase gate had done independent work.
    It had not: ``passes_phase`` used to repeat ``excess > 0``, so gate 1 merely
    implied gate 0 and every level rejection was double-counted as a phase
    rejection.  The level comparison belongs to gate 0 alone.
    """

    grid = make_grid()
    grid.loc[grid["arm"] == "worse", "total_return"] = 0.90
    phase_excess = make_phase_excess({"baseline": 0.0, "good": 0.02, "thin": 0.05,
                                      "fragile": -0.04, "worse": 0.10})
    outcome = select_configuration(grid, baseline_key=BASELINE, phase_excess=phase_excess)
    worse = row_for(outcome, "worse")
    assert worse["excess"] < 0 and worse["phase_excess_min"] > 0, "fixture must fail only the level"
    assert worse["passes_phase"] is True or bool(worse["passes_phase"]) is True
    assert worse["reject_reason"] == "beats_baseline"


def test_a_grid_keyed_by_something_other_than_arm_still_selects():
    """Not every grid names its rows ``arm``.

    ``tradability_ranking_evaluation`` is keyed by ``top_k`` alone.  ``.arm`` and
    the reason string used to read ``row["arm"]`` directly, which turned a
    perfectly good selection into a KeyError -- or, worse, a mislabelled reason.
    """

    grid = pd.DataFrame([
        {"top_k": 0.005, "validation_total_return": 0.10, "validation_turnover_multiple": 1.0},
        {"top_k": 0.050, "validation_total_return": 0.50, "validation_turnover_multiple": 3.0},
        {"top_k": 0.100, "validation_total_return": 0.30, "validation_turnover_multiple": 3.5},
    ])
    outcome = select_configuration(
        grid, baseline_key={"top_k": 0.100}, key_columns=("top_k",),
        return_column="validation_total_return",
        turnover_column="validation_turnover_multiple")

    assert outcome.row is not None
    assert float(outcome.row["top_k"]) == 0.050
    assert outcome.arm == "0.05"
    assert outcome.arm in outcome.reason
    # A basket an order of magnitude thinner barely trades, and the turnover
    # floor rejects it regardless of how the return looks.
    assert not outcome.table.set_index("top_k").loc[0.005, "passes_turnover"]


def test_rerunning_on_its_own_table_is_idempotent():
    """The rule must accept its own output as a grid.

    ``outcome.table`` carries ``phase_excess_min`` and ``fold_*``.  Feeding it
    back used to collide in the merge, which pandas resolves by renaming ours to
    ``_x``/``_y`` -- so the rule read the stale copy, or raised.  Both are
    silent failures of exactly the kind this module exists to prevent.
    """

    grid = make_grid()
    phase_excess = make_phase_excess()
    folds = pd.DataFrame([
        {"arm": arm, "fold_0": 0.01, "fold_1": 0.01}
        for arm in grid["arm"]
    ])
    first = select_configuration(grid, baseline_key=BASELINE,
                                 phase_excess=phase_excess, fold_excess=folds)
    again = select_configuration(first.table, baseline_key=BASELINE,
                                 phase_excess=phase_excess, fold_excess=folds)
    assert again.arm == first.arm == "good"
    assert again.reason == first.reason


def test_baseline_key_must_be_unambiguous():
    grid = pd.concat([make_grid(), make_grid()], ignore_index=True)
    with pytest.raises(ValueError, match="matched 2 rows"):
        select_configuration(grid, baseline_key=BASELINE)


def test_fold_returns_slice_the_equity_curve_without_gapping():
    """Folds must compound to the whole window: no double-counting the boundary mark."""

    index = pd.bdate_range("2024-01-01", periods=10)
    equity = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41,
                        161.05, 177.16, 194.87, 214.36, 235.79], index=index)
    folds = [{"trade_start": index[0], "trade_end": index[5]},
             {"trade_start": index[5], "trade_end": index[-1] + pd.Timedelta(days=1)}]
    blocks = fold_returns_from_equity(equity, folds, initial_capital=100.0)
    assert (1.0 + pd.Series(blocks)).prod() == pytest.approx(235.79 / 100.0)
