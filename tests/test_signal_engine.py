import pandas as pd
import pytest

from signals.signal_engine import SignalEngine


@pytest.fixture
def sample_data():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame({"close": [10, 11, 12, 11]}, index=index)


class TestSignalEngineRegistration:
    def test_generate_registered_timing_signal(self, sample_data):
        engine = SignalEngine()

        def mock_strategy(data):
            return pd.Series([0, 1, 1, -1], index=data.index)

        engine.register_timing_strategy("mock", mock_strategy)
        signal = engine.generate_timing_signals(sample_data, "mock")

        expected = pd.Series([0, 1, 1, -1], index=sample_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_generate_missing_timing_signal_raises(self, sample_data):
        engine = SignalEngine()

        with pytest.raises(ValueError, match="missing"):
            engine.generate_timing_signals(sample_data, "missing")

    def test_generate_registered_ranking_signal(self):
        engine = SignalEngine()
        factor_data = {
            "factor_a": pd.DataFrame({"score": [1, 2]}, index=["AAA", "BBB"]),
        }

        def mock_ranking(data):
            return pd.DataFrame({"rank": [2, 1]}, index=["AAA", "BBB"])

        engine.register_ranking_strategy("mock_rank", mock_ranking)
        signal = engine.generate_ranking_signals(factor_data, "mock_rank")

        expected = pd.DataFrame({"rank": [2, 1]}, index=["AAA", "BBB"])
        pd.testing.assert_frame_equal(signal, expected)


class TestSignalEngineCombineSignals:
    def test_combine_signals_uses_equal_weights_by_default(self):
        engine = SignalEngine()
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        signals = [
            pd.Series([1, 1, -1], index=index),
            pd.Series([1, -1, -1], index=index),
        ]

        combined = engine.combine_signals(signals)

        expected = pd.Series([1.0, 0.0, -1.0], index=index)
        pd.testing.assert_series_equal(combined, expected)

    def test_combine_signals_uses_custom_weights(self):
        engine = SignalEngine()
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        signals = [
            pd.Series([1, 0, -1], index=index),
            pd.Series([0, 1, -1], index=index),
        ]

        combined = engine.combine_signals(signals, weights=[0.7, 0.3])

        expected = pd.Series([0.7, 0.3, -1.0], index=index)
        pd.testing.assert_series_equal(combined, expected)

    def test_combine_signals_rejects_empty_signal_list(self):
        engine = SignalEngine()

        with pytest.raises(ValueError, match="must not be empty"):
            engine.combine_signals([])

    def test_combine_signals_rejects_weight_length_mismatch(self):
        engine = SignalEngine()
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        signals = [
            pd.Series([1, -1], index=index),
            pd.Series([1, 1], index=index),
        ]

        with pytest.raises(ValueError, match="weights length must match"):
            engine.combine_signals(signals, weights=[1.0])
