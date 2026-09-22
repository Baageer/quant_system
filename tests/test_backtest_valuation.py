"""Held names with no quote on a session must not vanish from the book.

``load_local_price_history`` drops NaN-close rows, so a halted name has no row
at all rather than a NaN row.  Valuing the book from that session's execution
cross-section therefore deleted the position for the day: a one-day phantom
loss the size of the name's weight, which understates Sharpe and overstates max
drawdown.  Execution must still see the gap -- you cannot trade a halted name --
so only the valuation marks are carried forward.
"""

import pandas as pd
import pytest

from backtest.engine import BacktestEngine


def make_daily_data(close_values, dates=None):
    index = dates if dates is not None else pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    return pd.DataFrame(
        {
            "open": close_values,
            "close": close_values,
            "high": close_values,
            "low": close_values,
            "volume": [1000] * len(close_values),
            "amount": [10000] * len(close_values),
            "pct_change": [0.0] * len(close_values),
        },
        index=index,
    )


def make_engine():
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.min_commission = 0.0
    return engine


def buy_once(symbol, shares):
    state = {"bought": False}

    def strategy_func(date, _data, _positions):
        if not state["bought"]:
            state["bought"] = True
            return {symbol: {"action": "buy", "shares": shares}}
        return {}

    return strategy_func


def test_halted_holding_is_marked_at_its_last_close():
    """AAA halts on day 3 (no bar at all); the book must not dip that day."""

    full_index = pd.date_range("2024-01-01", periods=5, freq="D")
    # AAA is bought on day 1 at 10, rallies to 12, then simply has no row on the
    # halt day -- exactly as load_local_price_history produces it.
    aaa = make_daily_data([10.0, 12.0, 12.0, 12.0], dates=full_index.drop([full_index[2]]))
    # BBB trades every session at a constant price, so the only possible source
    # of movement in the book is AAA falling out of the valuation.
    bbb = make_daily_data([20.0] * 5)

    engine = make_engine()
    results = engine.run(
        {"AAA": aaa, "BBB": bbb}, buy_once("AAA", 1000), show_progress=False
    )

    values = results["portfolio_value"].astype(float)
    assert len(values) == 5
    # Day 1 buys at 10; from day 2 the mark is 1000 AAA @ 12 + 90k cash = 102k,
    # held right across the halt day rather than dropping to the 90k cash
    # balance and jumping back the next session.
    assert values.iloc[1:].nunique() == 1
    assert values.iloc[2] == pytest.approx(102000.0)
    assert values.iloc[2] == values.iloc[1] == values.iloc[3]


def test_halt_does_not_block_a_later_exit():
    """The carried-forward mark is valuation-only; execution still needs a bar."""

    full_index = pd.date_range("2024-01-01", periods=4, freq="D")
    aaa = make_daily_data([10.0, 10.0, 12.0], dates=full_index.drop([full_index[1]]))
    engine = make_engine()
    engine.positions["AAA"] = 100
    engine.last_buy_dates["AAA"] = full_index[0]

    # On the halt day the name has no row, so a sell must be rejected rather
    # than filled at the stale mark.
    assert engine._can_execute_trade({"AAA": aaa}, "AAA", "sell", full_index[1]) is False
    # Once it trades again the exit goes through.
    assert engine._can_execute_trade({"AAA": aaa}, "AAA", "sell", full_index[2]) is True
