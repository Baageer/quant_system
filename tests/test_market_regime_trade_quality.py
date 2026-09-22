from pathlib import Path

import pandas as pd
import pytest

from research.market_regime.trade_quality import (
    calculate_core_regime_metrics,
    export_regime_quality_report,
    normalize_daily_positions,
)


def make_backtest_inputs():
    dates = pd.bdate_range("2024-01-02", periods=6)
    results = pd.DataFrame(
        {
            "portfolio_value": [100.0] * 6,
            "positions": [
                {"AAA": 10},
                {},
                {"AAA": 10},
                {},
                {"AAA": 10},
                {"AAA": 10},
            ],
        },
        index=dates,
    )
    market_data = {"AAA": pd.DataFrame({"close": [10.0] * 6}, index=dates)}
    daily_states = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * 6,
            "segment_id": [0, 0, 1, 1, 2, 2],
            "regime": ["up", "up", "down", "down", "sideways", "sideways"],
            "open_segment": [False, False, False, False, True, True],
            "regime_confidence": [0.8] * 6,
        }
    )
    trades = pd.DataFrame(
        {
            "date": [dates[4], dates[5]],
            "symbol": ["AAA", "AAA"],
            "action": ["buy", "sell"],
            "status": ["filled", "filled"],
            "trade_value": [100.0, 100.0],
        }
    )
    return results, market_data, daily_states, trades


def test_normalize_daily_positions_expands_eod_position_snapshots():
    results, market_data, _, _ = make_backtest_inputs()

    positions = normalize_daily_positions(results, market_data)

    assert positions["date"].tolist() == [results.index[0], results.index[2], results.index[4], results.index[5]]
    assert positions["market_value"].tolist() == [100.0, 100.0, 100.0, 100.0]
    assert positions["weight"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_core_metrics_calculate_up_coverage_down_exposure_and_sideways_turnover():
    results, market_data, daily_states, trades = make_backtest_inputs()
    positions = normalize_daily_positions(results, market_data)

    metrics = calculate_core_regime_metrics(results, positions, daily_states, trades)

    assert metrics["up_coverage"].iloc[0]["exposure_rate"] == pytest.approx(0.5)
    assert metrics["down_exposure"].iloc[0]["exposure_rate"] == pytest.approx(0.5)
    sideways = metrics["sideways_turnover"].iloc[0]
    assert sideways["trade_count"] == 2
    assert sideways["turnover_value"] == pytest.approx(200.0)
    assert sideways["turnover_rate"] == pytest.approx(2.0)
    summary = dict(zip(metrics["summary"]["metric"], metrics["summary"]["value"]))
    assert summary["up_coverage_rate"] == pytest.approx(0.5)
    assert summary["down_exposure_rate"] == pytest.approx(0.5)


def test_export_regime_quality_report_writes_csv_tables_and_manifest(tmpdir):
    results, market_data, daily_states, trades = make_backtest_inputs()
    positions = normalize_daily_positions(results, market_data)
    metrics = calculate_core_regime_metrics(results, positions, daily_states, trades)
    report = {
        "daily_positions": positions,
        "segments": pd.DataFrame({"label_version": ["market_regime_v1_1"]}),
        "daily_states": daily_states,
        "pivots": pd.DataFrame(),
        "label_parameters": {"min_segment_length": 20},
        "skipped_symbols": [],
    }
    report.update(metrics)

    paths = export_regime_quality_report(report, Path(str(tmpdir)), "demo")

    assert paths["daily_positions"].is_file()
    assert paths["up_coverage"].is_file()
    assert paths["manifest"].is_file()
    assert "market_regime_v1_1" in paths["manifest"].read_text(encoding="utf-8")
