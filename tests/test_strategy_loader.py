from signals.strategy_loader import StrategyLoader


class TestStrategyLoader:
    def test_build_timing_strategies(self):
        loader = StrategyLoader("./config/strategies.yaml")

        strategies, strategy_infos = loader.build_timing_strategies(["ma_cross"])

        assert len(strategies) == 1
        assert len(strategy_infos) == 1
        assert strategies[0].__class__.__name__ == "MACrossStrategy"
        assert strategy_infos[0]["module"] == "signals.timing.ma_cross"

    def test_build_stop_strategies(self):
        loader = StrategyLoader("./config/strategies.yaml")

        stop_loss_strategy, stop_loss_info, stop_profit_strategy, stop_profit_info = (
            loader.build_stop_strategies()
        )

        assert stop_loss_strategy is not None
        assert stop_profit_strategy is not None
        assert stop_loss_info["module"] == "signals.stop.stop_loss"
        assert stop_profit_info["module"] == "signals.stop.stop_profit"

    def test_missing_strategy_raises_helpful_error(self):
        loader = StrategyLoader("./config/strategies.yaml")

        try:
            loader.get_strategy("not_exists")
        except ValueError as exc:
            assert "not_exists" in str(exc)
            assert "timing_strategies" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing strategy")
