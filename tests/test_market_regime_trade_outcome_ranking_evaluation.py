"""The direct outcome model must select its basket width by the rule, not the grid argmax.

``run_outcome_ranking_evaluation`` kept ``float(validation_grid.iloc[0]["top_k"])``
long after ``select_configuration`` replaced that pattern everywhere else, and it
was the one site on the *production* path (the published ``hs300_pit_outcome`` /
``zz1000_pit_outcome`` widths, and everything ``stock_selection`` reads off
them).  The grids it gets are the same shape as the tradability module's, so
every gate had content -- this was a missed wiring, not an inapplicable rule.

Two things therefore need pinning, and neither is obvious from reading either
module:

1. **The baseline is per-model, not a constant.**  The two published pools sit
   at different widths (0.05 and 0.10), so a single module-level default would
   put one pool's candidates in front of the other pool's bar.  An unknown model
   id has to raise rather than fall back to a plausible-looking number.
2. **The old and new rules disagree on real grids, and we assert the new one.**
   Each grid below is built so that ``sort_values(...).iloc[0]`` picks a
   different row than the rule does; the test states both answers so a silent
   regression reads as a wrong number rather than a passing test.
"""

import inspect

import pandas as pd
import pytest

from research.market_regime import trade_outcome_ranking_evaluation as outcome_ranking
from research.market_regime.tradability_ranking_evaluation import (
    BASELINE_TOP_K,
    select_top_k_on_validation,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    PUBLISHED_TOP_K,
    resolve_baseline_top_k,
)


def argmax_top_k(grid):
    """What the retired selector returned, spelled out so the disagreement is visible."""

    return float(
        grid.sort_values(
            ["validation_total_return", "validation_turnover_multiple", "top_k"],
            ascending=[False, True, True],
        ).iloc[0]["top_k"]
    )


def make_grid(rows):
    return pd.DataFrame(
        [
            {"top_k": top_k, "validation_total_return": ret, "validation_turnover_multiple": turn}
            for top_k, ret, turn in rows
        ]
    )


# The real shape of the problem: a narrow basket wins on validation return purely
# by construction, and its turnover is far below the incumbent's.
WIDE_ARGMAX_GRID = make_grid([
    (0.005, 5.00, 1.0),
    (0.05, 1.00, 4.0),
    (0.10, 0.90, 4.0),
])


def test_the_argmax_and_the_rule_disagree_on_this_grid():
    """Guards the fixture: if these ever agree the regression tests stop testing anything."""

    assert argmax_top_k(WIDE_ARGMAX_GRID) == 0.005


def test_degenerate_narrow_basket_does_not_displace_the_incumbent(capsys):
    selected = select_top_k_on_validation(WIDE_ARGMAX_GRID, 0.05)

    assert selected == 0.05
    assert "NO CHANGE" in capsys.readouterr().out


def test_a_candidate_that_beats_the_incumbent_on_every_gate_is_taken(capsys):
    grid = make_grid([
        (0.02, 3.00, 3.0),
        (0.05, 1.00, 4.0),
        (0.10, 0.50, 4.0),
    ])

    assert select_top_k_on_validation(grid, 0.05) == 0.02


def test_incumbent_winning_on_return_means_nothing_changes():
    grid = make_grid([
        (0.05, 2.00, 4.0),
        (0.10, 1.50, 4.0),
    ])

    assert select_top_k_on_validation(grid, 0.05) == 0.05


def test_tradability_default_is_unchanged_at_its_own_baseline():
    """The new keyword must not move the sibling module off ``BASELINE_TOP_K``.

    This grid is read with 0.10 as the incumbent, so 0.05 loses on return and
    the answer is 0.10.  Had the default leaked to the outcome model's 0.05, the
    same grid would select 0.05 instead.
    """

    grid = make_grid([
        (0.05, 1.00, 4.0),
        (0.10, 2.00, 4.0),
    ])

    assert BASELINE_TOP_K == 0.10
    assert select_top_k_on_validation(grid) == 0.10


def test_a_baseline_off_the_grid_fails_loudly():
    grid = make_grid([
        (0.05, 1.00, 4.0),
        (0.10, 0.90, 4.0),
    ])

    with pytest.raises(ValueError):
        select_top_k_on_validation(grid, 0.03)


def test_the_two_published_pools_do_not_share_a_baseline():
    """A single module constant is provably wrong here, not merely inflexible."""

    assert resolve_baseline_top_k("hs300_pit_outcome") == 0.05
    assert resolve_baseline_top_k("zz1000_pit_outcome") == 0.10


def test_an_explicit_override_wins_over_the_registry():
    assert resolve_baseline_top_k("hs300_pit_outcome", 0.10) == 0.10


def test_an_unknown_model_id_raises_instead_of_guessing():
    """The example id must stay unregistered, or this stops testing anything.

    It was ``zz500_pit_outcome`` until that pool was built and registered; the
    name is deliberately one no pool can plausibly take, so registering the
    next pool cannot quietly turn this into an assertion about a real entry.
    """

    unregistered = "no_such_pool_pit_outcome"

    with pytest.raises(ValueError) as excinfo:
        resolve_baseline_top_k(unregistered)

    assert unregistered in str(excinfo.value)


def test_every_recorded_baseline_is_a_width_the_grid_can_offer():
    from research.market_regime.tradability_ranking_evaluation import TOP_K_GRID

    assert set(PUBLISHED_TOP_K.values()) <= set(TOP_K_GRID)


def test_the_run_wires_the_baseline_that_was_resolved():
    """The signature is the contract; the CLI default must not silently bypass it."""

    parameters = inspect.signature(outcome_ranking.run_outcome_ranking_evaluation).parameters

    assert "baseline_top_k" in parameters
    assert parameters["baseline_top_k"].default is None


def test_the_retired_selector_is_not_reachable_from_this_module():
    """The bug being fixed was a *missed site*, so assert on the expression itself.

    Behavioural coverage above cannot see which row the caller read; this can.
    The match is the exact retired expression, not a loose ``iloc[0]``, so
    prose and comments mentioning it stay legal.
    """

    source = inspect.getsource(outcome_ranking)

    assert "select_top_k_on_validation" in source
    assert 'validation_grid.iloc[0]["top_k"]' not in source
