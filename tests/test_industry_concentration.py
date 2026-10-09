"""E5 step 0: the coverage bracket, and the join that would silently break it.

The measurement itself needs the full price panel and is not re-run here.  What
*is* pinned is the part that decides the verdict and could be wrong without
anyone noticing: :func:`concentration_bounds` reads the pre-registered trigger on
a bracket instead of on the held book, and the bracket is only sound because an
unlabelled name can never *lower* the largest industry weight.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments.industry_concentration import (
    CONCENTRATION_TRIGGER,
    UNKNOWN,
    concentration_bounds,
    concentration_metrics,
    held_percentile_vs_baskets,
    industry_of,
    load_industry_map,
    pad_symbol,
)


def test_pad_symbol_restores_the_leading_zeros_the_predictions_dropped():
    """The published scores store the code as an int, so joining raw loses every
    Shenzhen name beginning with 0 -- most of them.  That reads as a data gap."""
    assert pad_symbol(1) == "000001"
    assert pad_symbol("000001.SZ") == "000001"
    assert pad_symbol("600519.SH") == "600519"
    assert pad_symbol(" 300750 ") == "300750"


def test_blank_industry_is_dropped_at_load_so_it_can_surface_as_unknown(tmp_path):
    """A blank is absent information.  Folding it into a bucket at load time would
    hide it; dropping the *name* would shrink the denominator and raise every
    weight.  Only the label is dropped, so industry_of can report UNKNOWN."""
    path = tmp_path / "stock_basic.csv"
    path.write_text(
        "symbol,industry\n000001.SZ,银行\n600519.SH,\n300750.SZ,电气设备\n",
        encoding="utf-8")
    mapping = load_industry_map(path)
    assert mapping == {"000001": "银行", "300750": "电气设备"}
    assert industry_of(1, mapping) == "银行"
    assert industry_of("600519.SH", mapping) == UNKNOWN
    assert industry_of("999999.SZ", mapping) == UNKNOWN


def test_bounds_bracket_the_maximum_for_every_assignment_of_the_blanks():
    """This is the whole reason the coverage gate does not have to stop the run.

    lower = best case for the portfolio (blanks all in one fresh industry),
    upper = worst case (blanks all piled into the largest labelled industry).
    Both ends are attainable, so the bracket is tight -- and the true maximum for
    any relabelling sits inside it, which is what lets a verdict read on `lower`
    stand without repairing the data.
    """
    weights = pd.DataFrame(
        {"银行": [0.30, 0.10], "证券": [0.10, 0.20], UNKNOWN: [0.60, 0.70]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    bounds = concentration_bounds(weights)
    assert bounds["lower"].tolist() == pytest.approx([0.30, 0.20])
    assert bounds["upper"].tolist() == pytest.approx([0.90, 0.90])
    # Every concrete relabelling of the blank mass lands in the bracket, and the
    # ends themselves are reached by the two extreme ones.
    for share_to_bank in np.linspace(0.0, 1.0, 11):
        for date, unknown in zip(weights.index, weights[UNKNOWN]):
            bank = weights.loc[date, "银行"]
            true_max = max(bank + unknown * share_to_bank,
                           weights.loc[date, "证券"])
            assert bounds.loc[date, "lower"] <= true_max <= bounds.loc[date, "upper"]


def test_bounds_are_decisive_only_where_both_ends_agree_on_the_trigger():
    """A day is decisive when the blanks cannot move it across 20%.  Straddling
    days stay indecisive on purpose: they are the ones that would have to be
    resolved by guessing, and guessing is how the close gets decided by the
    treatment rather than by the portfolio."""
    below = CONCENTRATION_TRIGGER - 0.05
    above = CONCENTRATION_TRIGGER + 0.05
    weights = pd.DataFrame(
        {
            "银行": [below, above, above, 0.10],
            "证券": [0.10, 0.10, 0.10, 0.10],
            UNKNOWN: [0.01, 0.01, 0.50, 0.50],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    bounds = concentration_bounds(weights)
    # settled below (upper still under), settled above (lower already over),
    # and two that straddle.
    assert bounds["decisive"].tolist() == [True, True, True, False]


def test_bounds_without_an_unknown_column_bracket_to_a_point():
    """ZZ1000 has no blanks at all, so its bracket must collapse -- otherwise the
    three-valued rule would report 'indeterminate' for a pool with perfect
    coverage and quietly keep E5 open forever."""
    weights = pd.DataFrame({"银行": [0.55], "证券": [0.45]},
                           index=pd.to_datetime(["2024-05-27"]))
    bounds = concentration_bounds(weights)
    assert bounds["lower"].iloc[0] == bounds["upper"].iloc[0] == 0.55
    assert bool(bounds["decisive"].iloc[0])


def test_bounds_take_the_maximum_over_labelled_columns_not_the_unknown_bucket():
    """UNKNOWN is usually the single largest column when a pool has blanks.  If it
    were allowed to win the max, the 'lower' bound would be the blank share and
    the bracket would be nonsense."""
    weights = pd.DataFrame({"银行": [0.25], UNKNOWN: [0.75]},
                           index=pd.to_datetime(["2024-01-02"]))
    bounds = concentration_bounds(weights)
    assert bounds["lower"].iloc[0] == 0.25


def test_baskets_are_drawn_as_wide_as_the_book_was_that_day():
    """A fixed-cap basket against a narrow book is a width difference, not a tilt:
    three names in three industries is already 33%, and scoring that against a
    20-name basket reports it as extreme concentration.  The null has to be the
    same size as the thing it is a null for."""
    from research.experiments.industry_concentration import permuted_basket_weights

    scores = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 40 + ["2024-01-03"] * 40),
        "symbol": ["%06d" % i for i in range(40)] * 2,
    })
    mapping = {"%06d" % i: "行业%d" % (i % 10) for i in range(40)}
    held_counts = pd.Series(
        [3, 19], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    baskets = permuted_basket_weights(
        scores, mapping, 0.5, held_counts=held_counts, draws=20, seed=7)
    # Each (draw, date) row is a composition over that day's width, so its
    # smallest attainable max weight is 1/width and it is never finer than that.
    for date, width in held_counts.items():
        day = baskets.xs(date, level="date")
        assert (day.sum(axis=1) - 1.0).abs().max() < 1e-12
        assert day.max(axis=1).min() >= (1.0 / width) - 1e-12
    narrow = baskets.xs(pd.Timestamp("2024-01-02"), level="date").max(axis=1)
    wide = baskets.xs(pd.Timestamp("2024-01-03"), level="date").max(axis=1)
    assert narrow.min() > wide.min()


def test_baskets_fall_back_to_the_cap_only_when_no_counts_are_given():
    from research.experiments.industry_concentration import permuted_basket_weights

    scores = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 40),
        "symbol": ["%06d" % i for i in range(40)],
    })
    mapping = {"%06d" % i: "行业%d" % (i % 10) for i in range(40)}
    with_counts = permuted_basket_weights(
        scores, mapping, 0.5, held_counts=pd.Series([4], index=scores["date"].unique()[:1]),
        draws=5, seed=7)
    assert len(with_counts) == 5
    without = permuted_basket_weights(scores, mapping, 0.5, draws=5, seed=7,
                                      max_positions=20)
    assert len(without) == 5


def _metrics_from_max(values, dates):
    return pd.DataFrame({"max_industry_weight": values}, index=pd.to_datetime(dates))


def test_percentile_ranks_the_held_book_inside_its_own_day_not_against_a_max_of_maxes():
    """The basket side is a max over draws, so it climbs with the draw count and
    the held book would look tame beside it.  Ranking per day removes that."""
    dates = ["2024-01-02", "2024-01-03"]
    held = _metrics_from_max([0.50, 0.10], dates)
    basket = pd.DataFrame(
        {"max_industry_weight": [0.10, 0.20, 0.30, 0.40, 0.60, 0.90]},
        index=pd.MultiIndex.from_tuples(
            [(0, pd.Timestamp("2024-01-02")), (1, pd.Timestamp("2024-01-02")),
             (2, pd.Timestamp("2024-01-02")), (0, pd.Timestamp("2024-01-03")),
             (1, pd.Timestamp("2024-01-03")), (2, pd.Timestamp("2024-01-03"))],
            names=["draw", "date"]))
    percentile = held_percentile_vs_baskets(held, basket)
    # day 1: held 0.50 beats all of 0.10/0.20/0.30 -> 3/3
    # day 2: held 0.10 beats none of 0.40/0.60/0.90 -> 0/3
    assert percentile.tolist() == pytest.approx([1.0, 0.0])


def test_percentile_refuses_mismatched_dates_rather_than_dropping_them():
    """A silent inner join here would rank the held book on whichever days the
    basket happened to cover and still print a plausible median."""
    held = _metrics_from_max([0.50], ["2024-01-02"])
    basket = pd.DataFrame(
        {"max_industry_weight": [0.20]},
        index=pd.MultiIndex.from_tuples(
            [(0, pd.Timestamp("2024-01-09"))], names=["draw", "date"]))
    with pytest.raises(ValueError) as excinfo:
        held_percentile_vs_baskets(held, basket)
    assert "have no basket" in str(excinfo.value)


def test_metrics_refuse_a_frame_that_does_not_partition():
    """A silent renormalisation upstream would make every weight look plausible
    and the trigger meaningless, so the partition is asserted rather than assumed."""
    weights = pd.DataFrame({"银行": [0.60], "证券": [0.20]},
                           index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError) as excinfo:
        concentration_metrics(weights)
    assert "partition" in str(excinfo.value)


def test_metrics_carry_the_hold_count_beside_the_weight():
    """A 100% industry weight in a one-name book is arithmetic, not a tilt.  The
    count travels with the weight so that cannot be misread."""
    weights = pd.DataFrame({"银行": [1.0, 0.30], "证券": [0.0, 0.70]},
                           index=pd.to_datetime(["2024-03-04", "2024-03-05"]))
    n_hold = pd.Series([1, 20], index=weights.index)
    metrics = concentration_metrics(weights, n_hold)
    assert metrics["max_industry_weight"].tolist() == [1.0, 0.70]
    assert metrics["n_hold"].tolist() == [1, 20]
