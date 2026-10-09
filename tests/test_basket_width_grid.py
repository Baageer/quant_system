"""E6's two load-bearing pieces: the width algebra, and the turnover floor.

The width algebra is what the experiment's *first* draft asserted too strongly --
it claimed two cells with the same ``min(ceil(n*top_k), max_positions)`` vector
were the same strategy, and the engine falsified that (the exit rule reads basket
membership too).  :func:`width_groups` is the surviving, weaker claim, so what it
must and must not group is pinned here.

``calibrate_turnover_floor`` is a *selection* input, which is the category of
code that has gone wrong silently four times in this repo.  The test that matters
is the last one: permuting the return column must not move the floor.  A floor
that reads returns is a tuned threshold wearing a measured one's clothes.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.basket_width_grid import (
    TOP_K_GRID,
    calibrate_turnover_floor,
    cell_name,
    date_sizes,
    effective_widths,
    width_groups,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN


def make_scores(date_sizes_by_date, score_column=SCORE_COLUMN):
    """A cross-section whose per-date size is exactly what the caller asks for."""

    rows = []
    for date, size in date_sizes_by_date.items():
        for index in range(size):
            rows.append({
                "date": pd.Timestamp(date),
                "symbol": "{:06d}.SZ".format(index),
                score_column: float(size - index),
            })
    return pd.DataFrame(rows)


def test_cell_names_round_trip_the_grid_coordinates():
    assert cell_name(0.05, 20) == "tk0.05_mp20"
    assert cell_name(0.1, 10) == "tk0.1_mp10"


def test_date_sizes_counts_each_cross_section():
    scores = make_scores({"2020-01-02": 3, "2020-01-03": 5})

    sizes = date_sizes(scores)

    assert list(sizes) == [3, 3, 3, 5, 5, 5, 5, 5]


def test_effective_width_is_the_capped_basket():
    scores = make_scores({"2020-01-02": 100})

    # ceil(100 * 0.05) == 5, uncapped
    assert float(effective_widths(scores, 0.05, 20).iloc[0]) == 5.0
    # ceil(100 * 0.30) == 30, capped at 20
    assert float(effective_widths(scores, 0.30, 20).iloc[0]) == 20.0
    # the same cell at a wider cap is a different width, not the same one
    assert float(effective_widths(scores, 0.30, 40).iloc[0]) == 30.0


def test_top_k_outside_the_unit_interval_is_refused():
    scores = make_scores({"2020-01-02": 10})

    with pytest.raises(ValueError):
        effective_widths(scores, 0.0, 20)


def test_cells_that_reach_the_cap_together_share_a_width_vector():
    # 300 names: ceil(300*0.10)=30 and ceil(300*0.30)=90 both exceed the cap of
    # 20, so both hold the top 20 names -- the same construction.
    scores = make_scores({"2020-01-02": 300})

    groups = width_groups(scores, (0.10, 0.30), (20,))

    assert groups[(0.10, 20)] == groups[(0.30, 20)] == [(0.10, 20), (0.30, 20)]


def test_a_wider_cap_is_a_different_group_even_at_the_same_basket():
    """Same names, half the size per name -- an exposure difference, so it must
    not be grouped with the cell it shares a basket with."""

    scores = make_scores({"2020-01-02": 100})

    groups = width_groups(scores, (0.05,), (20, 40))

    assert groups[(0.05, 20)] == [(0.05, 20)]
    assert groups[(0.05, 40)] == [(0.05, 40)]


def test_a_narrow_cell_is_never_grouped_with_a_capped_one():
    scores = make_scores({"2020-01-02": 100})

    groups = width_groups(scores, (0.01, 0.30), (20,))

    assert groups[(0.01, 20)] == [(0.01, 20)]
    assert groups[(0.30, 20)] == [(0.30, 20)]


def test_the_width_vector_is_per_date_not_an_average():
    """Cells are grouped by the whole vector, not by a summary of it: these two
    have the same *median* width (1) and differ on one date out of three."""

    scores = make_scores({"2020-01-02": 100, "2020-01-03": 10, "2020-01-06": 10})

    # 0.05 -> (5, 1, 1); 0.10 -> (10, 1, 1).  Same median, not grouped.
    assert width_groups(scores, (0.05, 0.10), (20,))[(0.05, 20)] == [(0.05, 20)]


def make_turnover_grid(turnovers, returns):
    return pd.DataFrame({
        "arm": list(turnovers),
        "validation_total_return": [returns[arm] for arm in turnovers],
        "validation_turnover_multiple": [turnovers[arm] for arm in turnovers],
    })


def test_the_floor_is_the_most_permissive_value_of_the_plateau():
    """Two thresholds that exclude the same set are equivalent, and the precedent
    (``probability_trade_evaluation``'s 0.05) takes the smaller of them: do not
    keep a restriction that is not doing any work."""

    grid = make_turnover_grid(
        {"base": 1.0, "near": 0.90, "mid": 0.40, "degenerate": 0.02},
        {"base": 0.10, "near": 0.20, "mid": 0.30, "degenerate": 0.90})

    calibration = calibrate_turnover_floor(grid, "base")

    # 0.05..0.35 all exclude exactly {"degenerate"} -- the plateau.  0.50 is
    # stricter and additionally drops "mid".  The floor is the smallest of the
    # plateau, so the returned floor is 0.05 and not 0.35 or 0.50.
    assert calibration["floor"] == 0.05
    assert calibration["plateau"] == [0.05, 0.10, 0.15, 0.25, 0.35]
    assert calibration["excluded_at_floor"] == ["degenerate"]


def test_a_floor_that_excludes_nothing_is_still_returned():
    grid = make_turnover_grid({"base": 1.0, "other": 0.9}, {"base": 0.1, "other": 0.2})

    calibration = calibrate_turnover_floor(grid, "base")

    assert calibration["excluded_at_floor"] == []


def test_an_absent_baseline_is_named():
    grid = make_turnover_grid({"base": 1.0}, {"base": 0.1})

    with pytest.raises(ValueError) as excinfo:
        calibrate_turnover_floor(grid, "nope")

    assert "nope" in str(excinfo.value)


def test_a_nonpositive_baseline_turnover_is_refused():
    """A ratio against zero is not a small floor, it is a division by zero that
    pandas will happily carry to inf and the gate will then read as 'passes'."""

    grid = make_turnover_grid(
        {"base": 0.0, "other": 1.0}, {"base": 0.1, "other": 0.2})

    with pytest.raises(ValueError) as excinfo:
        calibrate_turnover_floor(grid, "base")

    assert "turnover" in str(excinfo.value)


def test_the_floor_does_not_read_returns():
    """The whole point of calling it measured rather than tuned.  If permuting the
    return column moves the floor, the floor is choosing a winner."""

    turnovers = {"base": 1.0, "a": 0.90, "b": 0.50, "c": 0.02}
    low = make_turnover_grid(turnovers, {"base": 0.1, "a": 0.9, "b": -0.5, "c": 0.3})
    high = make_turnover_grid(turnovers, {"base": 0.1, "a": -0.5, "b": 0.9, "c": 0.3})

    assert (calibrate_turnover_floor(low, "base")["floor"]
            == calibrate_turnover_floor(high, "base")["floor"])
    assert (calibrate_turnover_floor(low, "base")["excluded_at_floor"]
            == calibrate_turnover_floor(high, "base")["excluded_at_floor"])


def test_the_pinned_grid_keeps_the_published_values_on_it():
    """The baseline cell has to be *on* the grid or the four gates have no bar to
    compare against; the published ``TOP_K_GRID`` is the first six entries."""

    assert TOP_K_GRID[:6] == (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
