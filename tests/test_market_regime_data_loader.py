from pathlib import Path

import pandas as pd
import pytest

from research.market_regime.data_loader import (
    find_local_price_files,
    load_hs300_symbols,
    load_local_price_history,
    to_tushare_symbol,
)


def write_price_file(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_to_tushare_symbol_maps_shanghai_and_shenzhen_codes():
    assert to_tushare_symbol("600000") == "600000.SH"
    assert to_tushare_symbol("000001") == "000001.SZ"
    assert to_tushare_symbol("000001.SZ") == "000001.SZ"


def test_load_hs300_symbols_uses_non_empty_lines(tmpdir):
    stock_file = Path(str(tmpdir)) / "HS300.txt"
    stock_file.write_text("600000\n\n000001\n", encoding="utf-8")

    assert load_hs300_symbols(stock_file) == ["600000", "000001"]


def test_load_local_price_history_merges_snapshots_and_keeps_latest_date_value(tmpdir):
    tmp_path = Path(str(tmpdir))
    cache_dir = tmp_path / "price_history" / "raw_hfq_pct"
    cache_dir.mkdir(parents=True)
    write_price_file(
        cache_dir / "600000.SH_20050101_20241231.csv",
        [{"date": "2024-01-02", "close": 10.0}, {"date": "2024-01-03", "close": 11.0}],
    )
    write_price_file(
        cache_dir / "600000.SH_20050101_20251231.csv",
        [{"date": "2024-01-03", "close": 12.0}, {"date": "2024-01-04", "close": 13.0}],
    )

    files = find_local_price_files("600000", tmp_path / "price_history")
    data = load_local_price_history("600000", tmp_path / "price_history")

    assert len(files) == 2
    assert data.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert data.loc[pd.Timestamp("2024-01-03"), "close"] == pytest.approx(12.0)
