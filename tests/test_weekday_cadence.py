"""The weekday-shifted cadence must be v1 on a different weekday, nothing else.

``make_weekday_rank_strategy`` is a line-for-line copy of
``make_rank_strategy`` with the "is this a rebalance session?" test changed.  The
whole family is only interpretable if that claim holds, so the first test here
pins it directly: ``weekday=0`` must emit the *identical* signal dict on every
session.  The rest pin the calendar semantics the family relies on -- one strike
per ISO week, on the first session on or after the target weekday.
"""

import pandas as pd
import pytest

from research.market_regime.tradability_ranking_evaluation import make_rank_strategy
from research.market_regime.turnover_optimization_ab import make_weekday_rank_strategy


SYMBOLS = ("AAA", "BBB")


def make_frames(dates, buyable=("AAA", "BBB")):
    """One frame per symbol with the columns the rank strategies read."""

    frames = {}
    for position, symbol in enumerate(SYMBOLS):
        frames[symbol] = pd.DataFrame(
            {
                "close": [10.0 + position] * len(dates),
                "rank_score": [1.0 if symbol in buyable else 0.0] * len(dates),
                "entry_trade": [True] * len(dates),
                "hold_trade": [False] * len(dates),
            },
            index=pd.DatetimeIndex(dates),
        )
    return frames


def drive(strategy, dates, frames, stateless=False):
    """Feed the strategy a session at a time, applying its own signals to the book.

    Both strategies are driven with the identical position book so any signal
    difference is the cadence test and not a divergence in bookkeeping.

    ``stateless`` presents an always-empty book instead.  A rebalance session with
    a full book legitimately emits nothing (no free slots, no unheld candidates),
    so an empty signal dict is not evidence of a non-rebalance session -- but with
    an empty book and a buyable name every strike does emit, which is what makes
    the strike dates observable.
    """

    positions = {}
    log = []
    for date in dates:
        relevant = {symbol: frames[symbol] for symbol in SYMBOLS}
        signals = strategy(date, relevant, {} if stateless else positions)
        log.append((date, signals))
        for symbol, signal in signals.items():
            if signal["action"] == "buy":
                positions[symbol] = positions.get(symbol, 0) + int(signal["shares"])
            else:
                positions.pop(symbol, None)
    return log


@pytest.fixture(scope="module")
def sessions():
    # Six Mon-Fri weeks plus the Monday that opens the seventh.
    return list(pd.bdate_range("2024-01-01", periods=31))


def test_weekday_zero_is_byte_identical_to_production(sessions):
    frames = make_frames(sessions)
    production = drive(make_rank_strategy(3, 10000.0, min_holding_days=5), sessions, frames)
    shifted = drive(make_weekday_rank_strategy(3, 10000.0, 5, weekday=0), sessions, frames)
    assert shifted == production


def test_each_target_strikes_once_per_iso_week(sessions):
    """With an always-empty book and an always-buyable name, a strike is visible."""

    frames = make_frames(sessions)
    by_week = {}
    for date in sessions:
        by_week.setdefault(pd.Timestamp(date).isocalendar()[1], []).append(pd.Timestamp(date))
    # The final week is a lone Monday, so Thursday and Friday targets must skip it.
    assert len(by_week[max(by_week)]) == 1, "fixture should end on a Monday-only week"

    for weekday in range(5):
        strategy = make_weekday_rank_strategy(3, 10000.0, 5, weekday=weekday)
        fired = [d for d, signals in drive(strategy, sessions, frames, stateless=True) if signals]
        struck_weeks = pd.Series([pd.Timestamp(d).isocalendar()[1] for d in fired])
        # One strike per week at most, and no week struck twice.
        assert struck_weeks.is_unique, "weekday={} struck a week twice".format(weekday)
        # A week is struck iff it has a session on or after the target.  On an
        # ordinary Mon-Fri week that is every target; the lone-Monday week drops
        # out for Thu/Fri.  The weekday shift moves *when*, not how often.
        expected = {week for week, days in by_week.items()
                    if any(day.weekday() >= weekday for day in days)}
        assert set(struck_weeks) == expected


def test_fires_on_the_first_session_on_or_after_the_target(sessions):
    frames = make_frames(sessions)
    by_week = {}
    for date in sessions:
        by_week.setdefault(pd.Timestamp(date).isocalendar()[1], []).append(pd.Timestamp(date))

    for weekday in range(5):
        strategy = make_weekday_rank_strategy(3, 10000.0, 5, weekday=weekday)
        fired = {pd.Timestamp(d) for d, signals in drive(strategy, sessions, frames, stateless=True) if signals}
        for week, days in by_week.items():
            eligible = [d for d in days if d.weekday() >= weekday]
            if eligible:
                assert fired & set(days) == {eligible[0]}, (
                    "weekday={} week {} should strike {} only".format(weekday, week, eligible[0])
                )
            else:
                assert not (fired & set(days)), "weekday={} week {} should skip".format(weekday, week)


def test_monday_only_week_is_skipped_by_late_targets():
    """A week whose only session is Monday cannot strike Thu/Fri -- and is skipped."""

    dates = list(pd.bdate_range("2024-01-01", periods=1)) + list(pd.bdate_range("2024-01-08", periods=5))
    frames = make_frames(dates)
    strategy = make_weekday_rank_strategy(3, 10000.0, 5, weekday=4)
    fired = [d for d, signals in drive(strategy, dates, frames, stateless=True) if signals]
    assert pd.Timestamp("2024-01-01") not in fired          # Monday-only week
    assert pd.Timestamp("2024-01-12") in fired              # the Fri of the next week


def test_reported_strike_count_matches_the_strategy(sessions):
    """The report's rebalance count must be the strategy's own firing, not a guess.

    ``weekday_strike_dates`` duplicates the rebalance test to predict how many
    rebalances a target will produce; if it drifts from the strategy the report
    silently compares cadences while claiming to compare weekdays.
    """

    from research.experiments.validation_null_phase_check import weekday_strike_dates

    frames = make_frames(sessions)
    for weekday in range(5):
        strategy = make_weekday_rank_strategy(3, 10000.0, 5, weekday=weekday)
        fired = [d for d, signals in drive(strategy, sessions, frames, stateless=True) if signals]
        assert fired == weekday_strike_dates(sessions, weekday)
