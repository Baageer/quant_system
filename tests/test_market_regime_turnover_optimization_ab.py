import pandas as pd

from research.market_regime.turnover_optimization_ab import (
    BASELINE_ARM,
    build_arm_specs,
    make_periodic_rank_strategy,
    phase_offsets,
)


def test_phase_offsets_cover_the_cadence_evenly():
    assert phase_offsets(5, 5) == (0, 1, 2, 3, 4)
    assert phase_offsets(10, 5) == (0, 2, 4, 6, 8)
    assert phase_offsets(20, 5) == (0, 4, 8, 12, 16)


def test_phase_offsets_handle_degenerate_cadences():
    # A one-session cadence rebalances every session, so phase is meaningless.
    assert phase_offsets(1, 5) == (0,)
    # Asking for more phases than the cadence allows is clamped, not an error.
    assert phase_offsets(3, 10) == (0, 1, 2)


def make_market_data(symbols=("A",), sessions=12, price=10.0):
    dates = pd.bdate_range("2021-01-04", periods=sessions)
    data = {
        symbol: pd.DataFrame(
            {"hold_trade": False, "entry_trade": True, "rank_score": 1.0, "close": price}, index=dates
        )
        for symbol in symbols
    }
    return data, dates


def fired_sessions(strategy, data, dates):
    return [index for index, date in enumerate(dates) if strategy(date, data, {})]


def test_periodic_strategy_rebalances_on_the_cadence_and_nowhere_else():
    data, dates = make_market_data(sessions=12)
    strategy = make_periodic_rank_strategy(
        max_positions=1, target_notional=10000.0, min_holding_days=0, rebalance_days=5, offset=0
    )

    assert fired_sessions(strategy, data, dates) == [0, 5, 10]


def test_offset_shifts_the_rebalance_phase():
    data, dates = make_market_data(sessions=12)

    def sessions_for(offset):
        strategy = make_periodic_rank_strategy(
            max_positions=1, target_notional=10000.0, min_holding_days=0, rebalance_days=5, offset=offset
        )
        return fired_sessions(strategy, data, dates)

    assert sessions_for(0) == [0, 5, 10]
    assert sessions_for(3) == [3, 8]


def test_unit_cadence_matches_acting_every_session():
    data, dates = make_market_data(sessions=6)
    strategy = make_periodic_rank_strategy(
        max_positions=1, target_notional=10000.0, min_holding_days=0, rebalance_days=1, offset=0
    )

    assert fired_sessions(strategy, data, dates) == list(range(6))


def test_hold_trade_keeps_a_still_selected_name_held():
    """A held name that is still selected (`hold_trade`) is neither sold nor re-bought."""
    data, dates = make_market_data(symbols=("A",), sessions=6)
    data["A"]["hold_trade"] = True
    strategy = make_periodic_rank_strategy(
        max_positions=2, target_notional=10000.0, min_holding_days=0, rebalance_days=1, offset=0
    )

    signals = [strategy(date, data, {"A": 1000}) for date in dates]

    # No exit fires (hold_trade), and A still occupies a slot so it is not
    # re-bought either -- the book is simply left alone.
    assert signals == [{}] * 6


def test_arm_specs_lead_with_the_iso_week_baseline():
    specs = build_arm_specs([10, 20], [5, 10], phases=3)

    assert specs[0]["arm"] == BASELINE_ARM
    assert specs[0]["rebalance_days"] is None
    assert specs[0]["offsets"] == (0,)


def test_arm_specs_expand_every_hold_and_cadence_pair():
    specs = build_arm_specs([10, 20], [5, 10], phases=3)
    periodic = {(spec["min_holding_days"], spec["rebalance_days"]) for spec in specs[1:]}

    assert periodic == {(10, 5), (10, 10), (20, 5), (20, 10)}
    assert all(len(spec["offsets"]) == 3 for spec in specs[1:])
