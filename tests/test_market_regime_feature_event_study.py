import numpy as np
import pandas as pd

from research.market_regime.feature_event_study import (
    _event_window_observations,
    audit_feature_causality,
    build_causal_feature_panel,
    summarize_event_windows_compact,
    summarize_turning_point_features,
)


def make_ohlcv(periods=180):
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    close = pd.Series(10 + np.linspace(0, 8, periods) + np.sin(np.arange(periods) / 5), index=dates)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1000 + np.arange(periods) * 3,
        },
        index=dates,
    )


def test_causal_panel_has_expected_research_features_and_index():
    panel = build_causal_feature_panel({"AAA": make_ohlcv()})

    assert panel.index.names == ["date", "symbol"]
    assert {"ret_60d", "sma_gap_5_20", "di_spread_14", "volatility_60", "volume_ratio_60"}.issubset(panel.columns)
    assert not {"regime", "segment_id", "is_pivot"}.intersection(panel.columns)


def test_causal_panel_can_limit_features_in_generation_order():
    full_panel = build_causal_feature_panel({"AAA": make_ohlcv()})
    limited_panel = build_causal_feature_panel({"AAA": make_ohlcv()}, feature_limit=12)

    assert len(limited_panel.columns) == 12
    assert list(limited_panel.columns) == list(full_panel.columns[:12])


def test_causal_panel_is_unchanged_when_future_rows_change():
    original = make_ohlcv()
    revised = original.copy()
    revised.loc[revised.index[-10:], ["high", "low", "close", "volume"]] *= 3

    original_panel = build_causal_feature_panel({"AAA": original}).xs("AAA", level="symbol")
    revised_panel = build_causal_feature_panel({"AAA": revised}).xs("AAA", level="symbol")

    pd.testing.assert_frame_equal(original_panel.iloc[:-10], revised_panel.iloc[:-10])
    assert audit_feature_causality(original).empty


def test_summary_compares_event_and_matched_control_paths():
    observations = pd.DataFrame(
        {
            "transition": ["down_to_up"] * 4,
            "sample_group": ["event", "event", "control", "control"],
            "feature": ["rsi_14"] * 4,
            "relative_day": [-1, -1, -1, -1],
            "value": [60.0, 70.0, 40.0, 50.0],
        }
    )

    summary, effects = summarize_turning_point_features(observations)

    assert set(summary["sample_group"]) == {"event", "control"}
    assert effects.iloc[0]["mean_difference"] == 20.0
    assert effects.iloc[0]["standardized_effect"] > 0


def test_compact_window_summary_matches_legacy_observation_summary():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    panel = pd.concat(
        [
            pd.DataFrame({"feature": np.arange(6, dtype=float)}, index=dates).assign(symbol="AAA"),
            pd.DataFrame({"feature": np.arange(10, 16, dtype=float)}, index=dates).assign(symbol="BBB"),
        ]
    ).set_index("symbol", append=True)
    panel.index.names = ["date", "symbol"]
    events = pd.DataFrame(
        {
            "event_id": ["one", "two"],
            "symbol": ["AAA", "BBB"],
            "transition": ["down_to_up", "down_to_up"],
            "event_date": [dates[3], dates[3]],
        }
    )
    controls = events.copy()
    controls["control_date"] = [dates[2], dates[2]]

    raw = pd.concat(
        [
            _event_window_observations(panel, events, ["feature"], 1, 0, "event"),
            _event_window_observations(panel, controls, ["feature"], 1, 0, "control"),
        ],
        ignore_index=True,
    )
    raw_summary, raw_effects = summarize_turning_point_features(raw)
    compact_summary, compact_effects = summarize_event_windows_compact(panel, events, controls, ["feature"], 1, 0)

    pd.testing.assert_frame_equal(
        raw_summary.sort_values(list(raw_summary.columns[:4])).reset_index(drop=True),
        compact_summary.sort_values(list(compact_summary.columns[:4])).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        raw_effects.sort_values(list(raw_effects.columns[:3])).reset_index(drop=True),
        compact_effects.sort_values(list(compact_effects.columns[:3])).reset_index(drop=True),
    )
