"""Tests for parallel signal generation in the backtest service."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pandas.testing as pdt
import yaml

from backtest.backtest_service import BacktestRequest, LoadedMarketData, generate_signals


def _write_config_files(tmp_path: Path) -> Tuple[Path, Path]:
    settings_path = tmp_path / "settings.yaml"
    strategy_path = tmp_path / "strategies.yaml"
    settings_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": "test",
                    "cache_dir": str(tmp_path / "raw"),
                    "processed_dir": str(tmp_path / "processed"),
                },
                "indicator_cache": {"enabled": False},
                "backtest": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    strategy_path.write_text(
        yaml.safe_dump(
            {
                "timing_strategies": {
                    "ma_cross": {
                        "name": "MA Cross",
                        "module": "signals.timing.ma_cross",
                        "class": "MACrossStrategy",
                        "params": {"short_window": 3, "long_window": 8},
                        "min_data_length": 8,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return settings_path, strategy_path


def _make_price_data(offset: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=80, freq="D")
    close = 100 + offset + np.linspace(-3, 6, len(index)) + np.sin(np.arange(len(index)) / 2)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(len(index), 1000),
        },
        index=index,
    )


def test_generate_signals_parallel_matches_serial_and_preserves_order(tmp_path):
    settings_path, strategy_path = _write_config_files(tmp_path)
    loaded_data = LoadedMarketData(
        data={
            "000003.SZ": _make_price_data(3),
            "000001.SZ": _make_price_data(1),
            "000002.SZ": _make_price_data(2),
        },
        symbols=["000003.SZ", "000001.SZ", "000002.SZ"],
        source="test",
        adjust_mode_label="raw",
    )
    request = BacktestRequest(
        strategy_names=["ma_cross"],
        config_path=str(settings_path),
        strategy_config_path=str(strategy_path),
        signal_workers=1,
    )

    serial_result = generate_signals(loaded_data, request)

    progress_updates = []
    request.signal_workers = 2
    parallel_result = generate_signals(
        loaded_data,
        request,
        progress_callback=lambda completed, total, message: progress_updates.append(
            (completed, total, message)
        ),
    )

    assert list(parallel_result) == list(serial_result)
    assert progress_updates[0][0] == 0
    assert progress_updates[-1][0] == len(loaded_data.data)
    for symbol in serial_result:
        pdt.assert_frame_equal(parallel_result[symbol], serial_result[symbol])
