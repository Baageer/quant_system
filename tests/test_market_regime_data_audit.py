import json
from pathlib import Path

import pandas as pd

from research.market_regime.data_audit import audit_hs300_data, write_audit_report


def _write_snapshot(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_hs300_data_audit_reports_pool_cache_and_price_issues(tmpdir):
    root = Path(str(tmpdir))
    stock_file = root / "HS300.txt"
    stock_file.write_text("600000\n000001\n300001\n600000\nINVALID\n", encoding="utf-8")
    cache_dir = root / "price_history" / "raw_hfq_pct"
    cache_dir.mkdir(parents=True)
    _write_snapshot(
        cache_dir / "600000.SH_1.csv",
        [
            {"date": "2024-01-02", "close": 10.0},
            {"date": "2024-01-04", "close": 12.0},
            {"date": "2024-01-05", "close": -1.0},
        ],
    )
    _write_snapshot(
        cache_dir / "600000.SH_2.csv",
        [{"date": "2024-01-02", "close": 11.0}],
    )
    _write_snapshot(
        cache_dir / "000001.SZ_1.csv",
        [
            {"date": "2024-01-02", "close": 8.0},
            {"date": "2024-01-03", "close": 9.0},
            {"date": "2024-01-04", "close": 10.0},
        ],
    )

    report = audit_hs300_data(stock_file, root / "price_history", calendar_coverage=0.5)

    summary = report["summary"]
    first_symbol = report["symbols"].set_index("symbol").loc["600000"]
    issues = set(report["issues"]["issue"])
    assert summary["stock_pool_unique_symbol_count"] == 3
    assert summary["missing_cache_symbol_count"] == 1
    assert first_symbol["duplicate_raw_date_count"] == 1
    assert first_symbol["conflicting_close_date_count"] == 1
    assert first_symbol["non_positive_close_count"] == 1
    assert first_symbol["missing_reference_trading_day_count"] == 1
    assert {"invalid_symbol", "duplicate_symbol", "missing_cache", "missing_reference_trading_days"}.issubset(issues)


def test_write_audit_report_exports_json_and_csv_files(tmpdir):
    root = Path(str(tmpdir))
    stock_file = root / "HS300.txt"
    stock_file.write_text("600000\n", encoding="utf-8")
    cache_dir = root / "price_history" / "raw_hfq_pct"
    cache_dir.mkdir(parents=True)
    _write_snapshot(cache_dir / "600000.SH_1.csv", [{"date": "2024-01-02", "close": 10.0}])

    report = audit_hs300_data(stock_file, root / "price_history")
    paths = write_audit_report(report, root / "reports")

    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["stock_pool_entry_count"] == 1
