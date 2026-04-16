import pandas as pd
import pytest

from backtest.engine import BacktestEngine


def make_daily_data(close_values, high_values=None, low_values=None):
    index = pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    if high_values is None:
        high_values = close_values
    if low_values is None:
        low_values = close_values

    return pd.DataFrame(
        {
            "open": close_values,
            "close": close_values,
            "high": high_values,
            "low": low_values,
            "volume": [1000] * len(close_values),
            "amount": [10000] * len(close_values),
            "pct_change": [0.0] * len(close_values),
        },
        index=index,
    )


def test_buy_order_is_rounded_to_lot_size():
    data = {"AAA": make_daily_data([10.0, 10.0])}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.min_commission = 0.0

    state = {"bought": False}

    def strategy_func(date, _data, _positions):
        if not state["bought"]:
            state["bought"] = True
            return {"AAA": {"action": "buy", "shares": 155}}
        return {}

    engine.run(data, strategy_func, show_progress=False)
    trades = engine.get_trades()

    assert len(trades) == 1
    assert trades.iloc[0]["action"] == "buy"
    assert trades.iloc[0]["shares"] == 100


def test_t1_blocks_same_day_sell():
    data = {"AAA": make_daily_data([10.0, 10.0])}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    d1, d2 = data["AAA"].index[0], data["AAA"].index[1]

    engine.positions["AAA"] = 100
    engine.last_buy_dates["AAA"] = d1

    assert engine._can_execute_trade(data, "AAA", "sell", d1) is False
    assert engine._can_execute_trade(data, "AAA", "sell", d2) is True


def test_limit_pct_by_prefix_applies_board_rules():
    growth_board = make_daily_data(
        close_values=[100.0, 115.0],
        high_values=[100.0, 115.0],
        low_values=[100.0, 115.0],
    )
    main_board = make_daily_data(
        close_values=[100.0, 115.0],
        high_values=[100.0, 115.0],
        low_values=[100.0, 115.0],
    )
    data = {"300001": growth_board, "600001": main_board}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    d2 = growth_board.index[1]

    assert engine._can_execute_trade(data, "300001", "buy", d2) is True
    assert engine._can_execute_trade(data, "600001", "buy", d2) is False


def test_rejected_buy_due_to_limit_up_is_recorded():
    data = {
        "600001": make_daily_data(
            close_values=[10.0, 11.0],
            high_values=[10.0, 11.0],
            low_values=[10.0, 11.0],
        )
    }
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    d2 = data["600001"].index[1]
    prices = {"600001": 11.0}

    engine._execute_trades({"600001": {"action": "buy", "shares": 100}}, prices, data, d2)
    trades = engine.get_trades()

    assert len(trades) == 1
    rejected = trades.iloc[0]
    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"] == "limit_up"
    assert rejected["action"] == "buy"
    assert rejected["requested_shares"] == 100
    assert rejected["shares"] == 0
    assert engine.positions.get("600001", 0) == 0


def test_sell_charges_commission_and_stamp_duty():
    data = {"AAA": make_daily_data([10.0, 10.0])}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0001, slippage=0.0)
    engine.min_commission = 5.0
    engine.stamp_duty_rate = 0.001

    d1, d2 = data["AAA"].index[0], data["AAA"].index[1]
    prices = {"AAA": 10.0}

    engine._execute_trades({"AAA": {"action": "buy", "shares": 100}}, prices, data, d1)
    engine._execute_trades({"AAA": {"action": "sell", "shares": 100}}, prices, data, d2)

    trades = engine.get_trades()
    assert set(trades["status"]) == {"filled"}
    sell_trade = trades[trades["action"] == "sell"].iloc[0]

    assert sell_trade["trade_value"] == pytest.approx(1000.0)
    assert sell_trade["commission"] == pytest.approx(5.0)
    assert sell_trade["stamp_duty"] == pytest.approx(1.0)
    assert sell_trade["revenue"] == pytest.approx(994.0)
