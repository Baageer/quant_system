from pathlib import Path

import pandas as pd
import pytest

from event_study_utils import (
    find_event_window_files,
    get_event_window,
    get_symbol_event_catalog,
    load_event_study_data,
)


def test_find_event_window_files_returns_sorted_matches(tmp_path: Path):
    output_dir = tmp_path / "pattern_event_study"
    output_dir.mkdir()

    (output_dir / "event_windows_b.csv").write_text("symbol\n000001\n", encoding="utf-8")
    (output_dir / "event_windows_a.csv").write_text("symbol\n000002\n", encoding="utf-8")
    (output_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    files = find_event_window_files(output_dir)

    assert files == [
        str(output_dir / "event_windows_a.csv"),
        str(output_dir / "event_windows_b.csv"),
    ]


def test_load_event_study_data_preserves_symbol_and_merges_meta(tmp_path: Path):
    event_windows_file = tmp_path / "event_windows_demo.csv"
    event_meta_file = tmp_path / "event_meta_demo.csv"

    event_windows_df = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "code": ["000001", "000001", "000001"],
            "open": [10.0, 10.2, 10.5],
            "close": [10.1, 10.4, 10.8],
            "high": [10.2, 10.5, 10.9],
            "low": [9.9, 10.1, 10.4],
            "volume": [1000, 1200, 1500],
            "relative_day": [-1, 0, 1],
            "symbol": ["000001", "000001", "000001"],
            "event_date": ["2024-01-03", "2024-01-03", "2024-01-03"],
            "event_close": [10.4, 10.4, 10.4],
            "close_to_event": [-0.0288, 0.0, 0.0385],
        }
    )
    event_meta_df = pd.DataFrame(
        {
            "symbol": ["000001"],
            "event_date": ["2024-01-03"],
            "event_close": [10.4],
            "bandwidth": [0.0521],
            "condition_threshold": [0.0412],
        }
    )

    event_windows_df.to_csv(event_windows_file, index=False, encoding="utf-8-sig")
    event_meta_df.to_csv(event_meta_file, index=False, encoding="utf-8-sig")

    event_windows, event_meta, event_catalog = load_event_study_data(event_windows_file)

    assert event_windows["symbol"].tolist() == ["000001", "000001", "000001"]
    assert event_windows["code"].tolist() == ["000001", "000001", "000001"]
    assert event_meta is not None
    assert event_catalog.loc[0, "symbol"] == "000001"
    assert event_catalog.loc[0, "bandwidth"] == pytest.approx(0.0521)
    assert event_catalog.loc[0, "condition_threshold"] == pytest.approx(0.0412)


def test_load_event_study_data_raises_for_missing_required_columns(tmp_path: Path):
    event_windows_file = tmp_path / "event_windows_invalid.csv"

    pd.DataFrame(
        {
            "trade_date": ["2024-01-03"],
            "open": [10.2],
            "close": [10.4],
        }
    ).to_csv(event_windows_file, index=False)

    with pytest.raises(ValueError, match="缺少必要字段"):
        load_event_study_data(event_windows_file)


def test_get_event_window_filters_visible_days_and_symbol_catalog(tmp_path: Path):
    event_windows_file = tmp_path / "event_windows_multi.csv"

    pd.DataFrame(
        {
            "trade_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-02-01",
                "2024-02-02",
            ],
            "open": [10.0, 10.2, 10.5, 20.0, 20.5],
            "close": [10.1, 10.4, 10.8, 20.3, 20.7],
            "high": [10.2, 10.5, 10.9, 20.4, 20.8],
            "low": [9.9, 10.1, 10.4, 19.8, 20.2],
            "volume": [1000, 1200, 1500, 2000, 2200],
            "relative_day": [-1, 0, 1, 0, 1],
            "symbol": ["000001", "000001", "000001", "000002", "000002"],
            "event_date": [
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
                "2024-02-01",
                "2024-02-01",
            ],
            "event_close": [10.4, 10.4, 10.4, 20.3, 20.3],
        }
    ).to_csv(event_windows_file, index=False)

    event_windows, _, event_catalog = load_event_study_data(event_windows_file)
    symbol_catalog = get_symbol_event_catalog(event_catalog, "000001")
    window = get_event_window(event_windows, "000001", pd.Timestamp("2024-01-03"), visible_days=0)

    assert len(symbol_catalog) == 1
    assert symbol_catalog.loc[0, "event_date"] == pd.Timestamp("2024-01-03")
    assert window["relative_day"].tolist() == [0]
    assert window.iloc[0]["close"] == pytest.approx(10.4)
