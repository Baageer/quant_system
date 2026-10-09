"""E5b's cap rule: quota arithmetic, backfill, and the width it must preserve.

The full experiment needs the price panel and is not re-run here.  What is pinned
is the rule itself, because two of its properties are load-bearing for the
reading and neither is visible in the output when it breaks: the cap has to
**bind** (floor, not ceil) and it has to **keep the basket the same width**
(backfill), or the arm stops being an industry intervention and becomes an
exposure one.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.industry_cap_ab import capped_selection

SCORE = "entry_score"


def _frame(industries_by_rank):
    """Scores frame with descending score in the given rank order."""
    symbols = ["%06d" % (i + 1) for i in range(len(industries_by_rank))]
    size = len(symbols)
    return pd.DataFrame({
        "date": pd.Timestamp("2024-01-02"),
        "symbol": symbols,
        SCORE: [float(size - i) for i in range(size)],
    }), {symbols[i]: name for i, name in enumerate(industries_by_rank)}


def _selected(frame, top_k):
    counts = frame.groupby("date")[SCORE].transform("size")
    ranks = frame.groupby("date")[SCORE].rank(method="first", ascending=False)
    return (ranks <= np.ceil(counts * top_k)).astype(bool)


def test_quota_uses_floor_so_the_cap_actually_binds():
    """n = 13 at 30% must be 3, not 4.  ceil(3.9) = 4 would be 30.8% of the book,
    i.e. the cap would not bind on exactly the days it exists for."""
    frame, mapping = _frame(["A"] * 13)
    selected = _selected(frame, 1.0)
    capped, diagnostics = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    assert diagnostics["quota_min"] == 3 and diagnostics["quota_max"] == 3
    assert 13 * 0.30 < 4  # the ceil that would have been wrong
    assert int(capped.sum()) == 3


def test_backfill_keeps_the_basket_the_same_width():
    """The excluded names are replaced from further down the ranked list, so the
    book is not quietly shrunk.  Without this the arm would confound the industry
    effect with an exposure effect and nothing could separate them."""
    frame, mapping = _frame(["A"] * 5 + ["B", "B", "C", "C", "D", "D", "E"])
    selected = _selected(frame, 3 / 12)
    assert int(selected.sum()) == 3
    capped, diagnostics = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    assert int(capped.sum()) == 3, "basket shrank instead of backfilling"
    assert diagnostics["days_short_of_target"] == 0
    kept = sorted(frame.loc[np.asarray(capped), "symbol"])
    assert kept == ["000001", "000006", "000008"]  # A from the top, then B, then C


def test_no_industry_exceeds_the_quota_and_order_is_preference():
    """Greedy in score order is what makes the rule deterministic and keeps the
    highest-scored eligible names -- ties in the basket are broken by the model's
    own ranking, not by the cap."""
    frame, mapping = _frame(["A", "B", "A", "A", "B", "C", "A", "C", "B", "D"])
    selected = _selected(frame, 6 / 10)
    capped, _ = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    n = int(selected.sum())
    quota = max(int(np.floor(0.30 * n)), 1)
    industries = [mapping[s] for s in frame.loc[np.asarray(capped), "symbol"]]
    counts = pd.Series(industries).value_counts()
    assert counts.max() <= quota
    # Rank 1 ("A") is eligible and must survive; it is the model's top pick.
    assert "000001" in set(frame.loc[np.asarray(capped), "symbol"])


def test_shortfall_is_counted_when_the_universe_cannot_fill_the_basket():
    """A day too concentrated to supply n eligible names yields a thinner book.
    That is allowed, but it must be visible -- a silent shrink is a different arm."""
    frame, mapping = _frame(["A"] * 8)
    selected = _selected(frame, 0.5)
    capped, diagnostics = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    assert int(selected.sum()) == 4
    assert int(capped.sum()) == 1, "a single-industry day cannot satisfy any quota above 1"
    assert diagnostics["days_short_of_target"] == 1
    assert diagnostics["worst_shortfall"] == 3


def test_share_of_one_is_a_no_op():
    """The upper end of the sensitivity grid has to leave the basket alone, or the
    grid's own reference point is not the incumbent."""
    frame, mapping = _frame(["A", "B", "A", "B", "C", "A"])
    selected = _selected(frame, 0.5)
    capped, _ = capped_selection(selected, frame, mapping, 1.0, score_column=SCORE)
    assert np.array_equal(np.asarray(capped, dtype=bool), np.asarray(selected, dtype=bool))


def test_cap_is_aligned_to_the_scores_frame_not_to_its_own_order():
    """``apply_rank_selection`` reads the selection positionally, so a filter that
    returns rows in a different order would be silently scoring the wrong names."""
    frame, mapping = _frame(["A", "A", "B", "C", "B", "A"])
    selected = _selected(frame, 0.5)
    capped, _ = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    assert len(capped) == len(selected)
    assert capped.index.equals(frame.index)
    # Backfill means the cap *may* admit a name ranked below the cut -- here rank 4
    # ("C") replaces a second "A".  So the invariant is "never wider than the
    # published basket", not "only ever removes".  Asserting the wrong one is how
    # a backfill gets deleted by someone trying to make a test pass.
    assert int(capped.sum()) <= int(selected.sum())
    # rank order is A A B C B A with n = 3 and quota = 1, so the walk keeps the
    # top A (rank 1), then the B at rank 3, then reaches past the cut to the C at
    # rank 4.  Rank 5's B is already at quota.  That rank-4 admission is backfill.
    kept = sorted(frame.loc[np.asarray(capped, dtype=bool), "symbol"])
    assert kept == ["000001", "000003", "000004"]


def test_backfill_depth_reports_how_far_the_cap_had_to_reach():
    """A cap that fills the book four baskets deep is not the same model spread
    out, and nothing else in the output would show that."""
    frame, mapping = _frame(["A", "A", "A", "A", "B", "C"])
    selected = _selected(frame, 0.5)  # n = 3, quota = 1
    capped, diagnostics = capped_selection(selected, frame, mapping, 0.30, score_column=SCORE)
    # ranks 1,2,3 are A A A; quota 1 keeps rank 1, then B (rank 5) and C (rank 6).
    assert diagnostics["backfill_depth_max"] == pytest.approx(6 / 3)
    assert int(capped.sum()) == 3

    flat_frame, flat_mapping = _frame(["A", "B", "C", "D", "E", "F"])
    _, flat_diagnostics = capped_selection(
        _selected(flat_frame, 0.5), flat_frame, flat_mapping, 0.30, score_column=SCORE)
    assert flat_diagnostics["backfill_depth_max"] == pytest.approx(1.0)


def test_rejects_a_share_outside_the_unit_interval():
    frame, mapping = _frame(["A", "B"])
    with pytest.raises(ValueError):
        capped_selection(_selected(frame, 0.5), frame, mapping, 0.0, score_column=SCORE)
