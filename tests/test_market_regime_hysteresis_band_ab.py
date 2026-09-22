import numpy as np
import pandas as pd

from research.market_regime.hysteresis_band_ab import (
    BASELINE_ARM,
    apply_band_selection,
    band_specs,
    select_rank_band,
    yearly_returns,
)
from research.market_regime.tradability_ranking_evaluation import select_top_k


def make_scores():
    dates = pd.to_datetime(["2021-01-01"] * 4 + ["2021-01-04"] * 4)
    return pd.DataFrame({
        "date": dates,
        "symbol": list("ABCDEFGH"),
        "entry_score": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.3, 0.2],
    })


def test_baseline_band_reproduces_select_top_k_exactly():
    """entry_k == exit_k must be bit-identical to v1's selection -- this is the parity guarantee."""
    scores = make_scores()

    for top_k in (0.125, 0.25, 0.5, 1.0):
        entry, held = select_rank_band(scores, top_k, top_k)
        reference = select_top_k(scores, top_k, "entry_score")

        assert entry.tolist() == reference.tolist()
        assert held.tolist() == reference.tolist()


def test_hold_band_contains_the_entry_band():
    scores = make_scores()

    entry, held = select_rank_band(scores, 0.125, 0.5)

    assert entry.sum() < held.sum()
    assert bool((entry & ~held).sum() == 0)  # every entry name is also a hold name


def test_exit_band_narrower_than_entry_is_rejected():
    scores = make_scores()

    try:
        select_rank_band(scores, 0.5, 0.125)
    except ValueError as error:
        assert "exit_k" in str(error)
    else:
        raise AssertionError("expected ValueError for exit_k < entry_k")


def test_out_of_range_bands_are_rejected():
    scores = make_scores()

    for entry_k, exit_k in ((0.0, 0.5), (0.5, 0.0), (0.5, 1.5)):
        try:
            select_rank_band(scores, entry_k, exit_k)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for entry={} exit={}".format(entry_k, exit_k))


def make_raw_market_data(dates, symbols):
    return {
        symbol: pd.DataFrame({"close": 10.0, "high": 10.0, "low": 10.0, "volume": 1.0}, index=dates)
        for symbol in symbols
    }


def test_band_selection_retains_a_name_it_would_not_re_enter():
    """The whole mechanism: a band name has hold_trade True but entry_trade False."""
    dates = pd.bdate_range("2021-01-04", periods=2)
    scores = pd.DataFrame({
        "date": list(dates) * 4,
        "symbol": [symbol for symbol in "ABCD" for _ in dates],
        # A tops the narrow band; A and B both sit inside the wider hold band.
        "entry_score": [0.9, 0.9, 0.8, 0.8, 0.2, 0.2, 0.1, 0.1],
    })
    raw = make_raw_market_data(dates, "ABCD")

    entry, held = select_rank_band(scores, 0.25, 0.5)
    prepared = apply_band_selection(raw, scores, entry, held)

    second = dates[1]
    # A: still in both bands -> can be bought, and is retained.
    assert bool(prepared["A"].loc[second, "entry_trade"])
    assert bool(prepared["A"].loc[second, "hold_trade"])
    # B: inside the hold band only -> retained but never (re-)entered.
    assert not bool(prepared["B"].loc[second, "entry_trade"])
    assert bool(prepared["B"].loc[second, "hold_trade"])
    # C: outside both -> neither entered nor retained.
    assert not bool(prepared["C"].loc[second, "entry_trade"])
    assert not bool(prepared["C"].loc[second, "hold_trade"])
    # Signals are shifted one session, so nothing fires on the first day.
    assert not prepared["A"]["entry_trade"].iloc[0]


def test_band_specs_lead_with_baseline_and_skip_exit_below_entry():
    specs = band_specs([0.05, 0.10], [0.10, 0.20])

    assert specs[0]["arm"] == BASELINE_ARM
    assert (specs[0]["entry_k"], specs[0]["exit_k"]) == (0.10, 0.10)
    # exit < entry is dropped, and the (0.10, 0.10) pair is not duplicated.
    assert [(spec["entry_k"], spec["exit_k"]) for spec in specs[1:]] == [
        (0.05, 0.10), (0.05, 0.20), (0.10, 0.20),
    ]


def test_yearly_returns_anchors_the_first_day_on_initial_capital():
    dates = pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-04"])
    equity = pd.Series([100.0, 110.0, 121.0], index=dates)

    yearly = yearly_returns(equity, 100.0)

    assert np.isclose(yearly.loc[2021], 0.10)  # 110 from an anchored 100
    assert np.isclose(yearly.loc[2022], 0.10)  # 121 from 110
