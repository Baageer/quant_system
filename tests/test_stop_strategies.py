import pandas as pd

from backtest.engine import BacktestEngine
from signals.stop.stop_loss import StopLossStrategy, stop_loss_signal
from signals.stop.stop_profit import StopProfitStrategy, stop_profit_signal


def make_stop_test_data():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 104.0, 103.0],
            "high": [101.0, 103.0, 106.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=index,
    )


def test_stop_profit_signal_uses_atr_threshold():
    data = make_stop_test_data()

    signals = stop_profit_signal(
        data,
        entry_price=100.0,
        profit_type="atr",
        atr_window=2,
        atr_multiplier=1.0,
    )

    expected = pd.Series([0, 0, 1, 0], index=data.index)
    pd.testing.assert_series_equal(signals, expected)


def test_stop_loss_signal_uses_atr_threshold():
    data = make_stop_test_data()
    data["close"] = [100.0, 99.0, 96.0, 95.0]
    data["high"] = [101.0, 102.0, 100.0, 98.0]
    data["low"] = [99.0, 98.0, 96.0, 95.0]

    signals = stop_loss_signal(
        data,
        entry_price=100.0,
        loss_type="atr",
        atr_window=2,
        atr_multiplier=1.0,
    )

    expected = pd.Series([0, 0, 1, 1], index=data.index)
    pd.testing.assert_series_equal(signals, expected)


def test_engine_applies_atr_stop_profit():
    data = {"AAA": make_stop_test_data()}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.set_stop_strategies(
        stop_profit_strategy=StopProfitStrategy(
            profit_type="atr",
            atr_window=2,
            atr_multiplier=1.0,
        )
    )

    state = {"bought": False}

    def strategy_func(date, data, positions):
        if not state["bought"]:
            state["bought"] = True
            return {"AAA": {"action": "buy", "shares": 1}}
        return {}

    engine.run(data, strategy_func, show_progress=False)

    trades = engine.get_trades()
    assert list(trades["action"]) == ["buy", "sell"]
    assert trades.iloc[1]["reason"] == "stop_profit"
    assert trades.iloc[1]["date"] == data["AAA"].index[2]


def test_engine_applies_atr_stop_loss():
    loss_data = make_stop_test_data()
    loss_data["close"] = [100.0, 99.0, 96.0, 95.0]
    loss_data["high"] = [101.0, 102.0, 100.0, 98.0]
    loss_data["low"] = [99.0, 98.0, 96.0, 95.0]

    data = {"AAA": loss_data}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.set_stop_strategies(
        stop_loss_strategy=StopLossStrategy(
            loss_type="atr",
            atr_window=2,
            atr_multiplier=1.0,
        )
    )

    state = {"bought": False}

    def strategy_func(date, data, positions):
        if not state["bought"]:
            state["bought"] = True
            return {"AAA": {"action": "buy", "shares": 1}}
        return {}

    engine.run(data, strategy_func, show_progress=False)

    trades = engine.get_trades()
    assert list(trades["action"]) == ["buy", "sell"]
    assert trades.iloc[1]["reason"] == "stop_loss"
    assert trades.iloc[1]["date"] == data["AAA"].index[2]


def test_engine_applies_holding_day_stop_loss():
    data = {"AAA": make_stop_test_data()}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.set_stop_strategies(
        stop_loss_strategy=StopLossStrategy(
            loss_type="holding_day",
            holding_day=2,
        )
    )

    state = {"bought": False}

    def strategy_func(date, data, positions):
        if not state["bought"]:
            state["bought"] = True
            return {"AAA": {"action": "buy", "shares": 1}}
        return {}

    engine.run(data, strategy_func, show_progress=False)

    trades = engine.get_trades()
    assert list(trades["action"]) == ["buy", "sell"]
    assert trades.iloc[1]["reason"] == "stop_loss"
    assert trades.iloc[1]["date"] == data["AAA"].index[2]


def test_engine_applies_holding_day_stop_profit():
    data = {"AAA": make_stop_test_data()}
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0, slippage=0.0)
    engine.set_stop_strategies(
        stop_profit_strategy=StopProfitStrategy(
            profit_type="holding_day",
            holding_day=2,
        )
    )

    state = {"bought": False}

    def strategy_func(date, data, positions):
        if not state["bought"]:
            state["bought"] = True
            return {"AAA": {"action": "buy", "shares": 1}}
        return {}

    engine.run(data, strategy_func, show_progress=False)

    trades = engine.get_trades()
    assert list(trades["action"]) == ["buy", "sell"]
    assert trades.iloc[1]["reason"] == "stop_profit"
    assert trades.iloc[1]["date"] == data["AAA"].index[2]
