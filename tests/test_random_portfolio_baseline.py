"""A random null is only a null if it holds the same basket width as the real arm.

``random_portfolio_baseline`` drew its null at a fixed ``--top-k 0.1`` while the
``ARMS`` default pointed at the retired static-pool models.  Both halves of that
were wrong for the current pools, and neither announced itself: the widths are
*not* uniform across pools (``hs300_pit_outcome`` published 0.05, ``zz1000_pit_outcome``
0.10), so an HS300 null drawn at 0.1 is wider than the arm it is supposed to
stand in for -- and that shows up as apparent alpha, or as apparent decay,
rather than as an error.

Two things are worth pinning, and the second is the one that actually rotted:

1. The width is **read from the published manifest**, not defaulted and not
   looked up in the selector's registry (they are different jobs -- see
   :func:`published_top_k`), and a missing manifest raises.
2. ``ARMS`` agrees with ``forward_2026_backtest.POOLS``.  The root cause was the
   same fact stored in two modules, so a test that only checks ``ARMS`` in
   isolation would not have caught the drift.
"""

import inspect
import json

import pytest

from research.experiments import random_portfolio_baseline as rpb
from research.experiments.forward_2026_backtest import POOLS
from research.market_regime.trade_outcome_ranking_evaluation import published_top_k


def write_manifest(tmp_path, model_id, top_k):
    path = tmp_path / "trade_outcome_ranking_{}_manifest.json".format(model_id)
    path.write_text(
        json.dumps({"selected_top_k_validation_only": top_k, "execution": {}}),
        encoding="utf-8",
    )
    return path


def test_the_width_is_read_from_the_manifest(tmp_path):
    write_manifest(tmp_path, "hs300_pit_outcome", 0.05)

    assert published_top_k(tmp_path, "hs300_pit_outcome") == 0.05


def test_the_manifest_wins_over_the_selector_registry(tmp_path):
    """They answer different questions; conflating them reintroduces the ratchet.

    ``resolve_baseline_top_k`` is the frozen incumbent a *selector* must beat.
    Here the manifest disagrees with it on purpose: a replica must follow the
    manifest, so the two must not be the same lookup.
    """

    from research.market_regime.trade_outcome_ranking_evaluation import PUBLISHED_TOP_K

    assert PUBLISHED_TOP_K["hs300_pit_outcome"] == 0.05
    write_manifest(tmp_path, "hs300_pit_outcome", 0.03)

    assert published_top_k(tmp_path, "hs300_pit_outcome") == 0.03


def test_a_missing_manifest_raises_and_names_the_file(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        published_top_k(tmp_path, "zz1000_pit_outcome")

    assert "zz1000_pit_outcome" in str(excinfo.value)
    assert "manifest" in str(excinfo.value)


def test_arms_point_at_the_current_pools_not_the_retired_static_ones():
    assert rpb.ARMS == {
        "hs300": "hs300_pit_outcome",
        "zz1000": "zz1000_pit_outcome",
    }


def test_arms_agrees_with_the_forward_backtest_pools():
    """The drift was two copies of one fact disagreeing, so compare the copies."""

    assert rpb.ARMS == {pool: spec["model_id"] for pool, spec in POOLS.items()}


def test_the_default_width_is_unset_rather_than_a_constant():
    """A default here cannot be right for every pool, so there must not be one."""

    parameter = inspect.signature(rpb.run_random_portfolio_baseline).parameters["top_k"]

    assert parameter.default is None
