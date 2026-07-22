"""HS300 股票池及本地行情缓存的数据质量审计。"""

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from .data_loader import DEFAULT_PRICE_HISTORY_DIR, find_local_price_files, to_tushare_symbol


DEFAULT_STOCK_FILE = Path(__file__).resolve().parents[2] / "data" / "HS300.txt"
_SYMBOL_PATTERN = re.compile(r"^\d{6}(?:\.(?:SH|SZ))?$")
_DATE_SAMPLE_SIZE = 10


def audit_hs300_data(
    stock_file=None,
    price_history_dir=None,
    adjustment="raw_hfq_pct",
    calendar_coverage=0.8,
):
    """审计股票池和本地行情快照，返回摘要、逐标的结果和问题明细。

    参考交易日由至少 ``calendar_coverage`` 比例的可用标的共同覆盖的日期组成。
    它只用于发现股票在自身有效历史区间内缺少的市场交易日，不会把停牌日误判为
    数据文件的日期格式问题。
    """

    if not 0 < calendar_coverage <= 1:
        raise ValueError("calendar_coverage 必须在 (0, 1] 内。")

    stock_path = Path(stock_file) if stock_file else DEFAULT_STOCK_FILE
    history_dir = Path(price_history_dir) if price_history_dir else DEFAULT_PRICE_HISTORY_DIR
    pool = _read_stock_pool(stock_path)
    valid_symbols = [symbol for symbol in pool["symbols"] if _SYMBOL_PATTERN.fullmatch(symbol)]
    unique_symbols = list(dict.fromkeys(valid_symbols))
    issues = list(pool["issues"])
    symbol_rows = []
    valid_dates_by_symbol = {}

    for symbol in unique_symbols:
        row, symbol_issues, valid_dates = _audit_symbol(symbol, history_dir, adjustment)
        symbol_rows.append(row)
        issues.extend(symbol_issues)
        if valid_dates:
            valid_dates_by_symbol[symbol] = valid_dates

    reference_dates, required_coverage = _reference_calendar(valid_dates_by_symbol, calendar_coverage)
    for row in symbol_rows:
        symbol = row["symbol"]
        dates = valid_dates_by_symbol.get(symbol, set())
        missing_dates = _missing_reference_dates(dates, reference_dates)
        row["missing_reference_trading_day_count"] = len(missing_dates)
        row["missing_reference_trading_day_samples"] = _format_dates(missing_dates)
        if missing_dates:
            if row["status"] == "ok":
                row["status"] = "warning"
            issues.append(
                _issue(
                    symbol,
                    "missing_reference_trading_days",
                    "warning",
                    len(missing_dates),
                    "样本有效区间内缺少参考交易日：{}".format(_format_dates(missing_dates)),
                )
            )

    symbol_table = pd.DataFrame(symbol_rows)
    issue_table = pd.DataFrame(
        issues,
        columns=["symbol", "issue", "severity", "count", "detail"],
    )
    summary = _build_summary(
        stock_path,
        history_dir,
        adjustment,
        calendar_coverage,
        pool,
        unique_symbols,
        symbol_table,
        issue_table,
        reference_dates,
        required_coverage,
    )
    return {"summary": summary, "symbols": symbol_table, "issues": issue_table}


def write_audit_report(report, output_dir="output"):
    """将审计结果写为 JSON 摘要和两个 CSV 明细文件。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "hs300_data_audit_summary.json",
        "symbols": directory / "hs300_data_audit_symbols.csv",
        "issues": directory / "hs300_data_audit_issues.csv",
    }
    paths["summary"].write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["symbols"].to_csv(paths["symbols"], index=False, encoding="utf-8-sig")
    report["issues"].to_csv(paths["issues"], index=False, encoding="utf-8-sig")
    return paths


def _read_stock_pool(stock_path):
    if not stock_path.is_file():
        raise FileNotFoundError("未找到股票池文件：{}".format(stock_path))

    symbols = []
    issues = []
    for line_number, line in enumerate(stock_path.read_text(encoding="utf-8").splitlines(), start=1):
        symbol = line.strip().upper()
        if not symbol:
            continue
        symbols.append(symbol)
        if not _SYMBOL_PATTERN.fullmatch(symbol):
            issues.append(_issue(symbol, "invalid_symbol", "error", 1, "第 {} 行不是六位 A 股代码。".format(line_number)))

    duplicates = [symbol for symbol, count in Counter(symbols).items() if count > 1]
    for symbol in duplicates:
        issues.append(_issue(symbol, "duplicate_symbol", "warning", Counter(symbols)[symbol], "股票池中重复出现。"))

    return {"symbols": symbols, "issues": issues, "duplicate_symbol_count": len(duplicates)}


def _audit_symbol(symbol, history_dir, adjustment):
    files = find_local_price_files(symbol, history_dir, adjustment)
    row = _empty_symbol_row(symbol, len(files))
    if not files:
        row["status"] = "missing_cache"
        return row, [_issue(symbol, "missing_cache", "error", 1, "未找到本地行情快照。")], set()

    frames = []
    issues = []
    for file_path in files:
        try:
            frame = pd.read_csv(file_path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            issues.append(_issue(symbol, "unreadable_snapshot", "error", 1, "{}: {}".format(file_path.name, error)))
            continue
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
        frame["_source_file"] = file_path.name
        frames.append(frame)

    if not frames:
        row["status"] = "invalid_data"
        return row, issues, set()

    raw_data = pd.concat(frames, ignore_index=True)
    row["raw_row_count"] = len(raw_data)
    missing_columns = [column for column in ("date", "close") if column not in raw_data.columns]
    row["missing_required_columns"] = ",".join(missing_columns)
    if missing_columns:
        row["status"] = "invalid_data"
        issues.append(_issue(symbol, "missing_required_columns", "error", len(missing_columns), row["missing_required_columns"]))
        return row, issues, set()

    dates = pd.to_datetime(raw_data["date"], errors="coerce")
    close = pd.to_numeric(raw_data["close"], errors="coerce")
    valid_date_mask = dates.notnull()
    valid_close_mask = close.notnull()
    valid_data_mask = valid_date_mask & valid_close_mask & (close > 0)
    valid_dates = set(dates[valid_data_mask].dt.normalize())

    row["missing_date_count"] = int((~valid_date_mask).sum())
    row["duplicate_raw_date_count"] = int(dates[valid_date_mask].duplicated().sum())
    row["missing_close_count"] = int((~valid_close_mask).sum())
    row["non_positive_close_count"] = int((valid_close_mask & (close <= 0)).sum())
    row["conflicting_close_date_count"] = _conflicting_close_dates(dates, close)
    row["valid_row_count"] = int(valid_data_mask.sum())
    row["trading_day_count"] = len(valid_dates)
    row["start_date"] = _format_date(min(valid_dates)) if valid_dates else ""
    row["end_date"] = _format_date(max(valid_dates)) if valid_dates else ""

    quality_counts = {
        "missing_date": row["missing_date_count"],
        "duplicate_raw_date": row["duplicate_raw_date_count"],
        "missing_close": row["missing_close_count"],
        "non_positive_close": row["non_positive_close_count"],
        "conflicting_close_date": row["conflicting_close_date_count"],
    }
    for issue_name, count in quality_counts.items():
        if count:
            severity = "info" if issue_name == "duplicate_raw_date" else "warning"
            detail = "重叠快照日期会由加载器保留最新记录。" if severity == "info" else "请在生成标签前处理或确认。"
            issues.append(_issue(symbol, issue_name, severity, count, detail))

    if not valid_dates:
        row["status"] = "invalid_data"
        issues.append(_issue(symbol, "no_valid_price_rows", "error", 1, "没有日期和正收盘价均有效的记录。"))
    elif any(issue["severity"] in {"warning", "error"} for issue in issues):
        row["status"] = "warning"
    return row, issues, valid_dates


def _empty_symbol_row(symbol, snapshot_file_count):
    return {
        "symbol": symbol,
        "tushare_symbol": to_tushare_symbol(symbol),
        "status": "ok",
        "snapshot_file_count": snapshot_file_count,
        "raw_row_count": 0,
        "valid_row_count": 0,
        "trading_day_count": 0,
        "start_date": "",
        "end_date": "",
        "missing_required_columns": "",
        "missing_date_count": 0,
        "duplicate_raw_date_count": 0,
        "missing_close_count": 0,
        "non_positive_close_count": 0,
        "conflicting_close_date_count": 0,
        "missing_reference_trading_day_count": 0,
        "missing_reference_trading_day_samples": "",
    }


def _conflicting_close_dates(dates, close):
    valid = pd.DataFrame({"date": dates, "close": close}).dropna()
    if valid.empty:
        return 0
    return int(valid.groupby("date")["close"].nunique().gt(1).sum())


def _reference_calendar(valid_dates_by_symbol, calendar_coverage):
    if not valid_dates_by_symbol:
        return set(), 0
    required_coverage = max(1, int(len(valid_dates_by_symbol) * calendar_coverage + 0.999999))
    date_counts = Counter(date_value for dates in valid_dates_by_symbol.values() for date_value in dates)
    return {date_value for date_value, count in date_counts.items() if count >= required_coverage}, required_coverage


def _missing_reference_dates(dates, reference_dates):
    if not dates or not reference_dates:
        return []
    start_date, end_date = min(dates), max(dates)
    return sorted(date_value for date_value in reference_dates if start_date <= date_value <= end_date and date_value not in dates)


def _build_summary(
    stock_path,
    history_dir,
    adjustment,
    calendar_coverage,
    pool,
    unique_symbols,
    symbol_table,
    issue_table,
    reference_dates,
    required_coverage,
):
    return {
        "audit_date": date.today().isoformat(),
        "stock_file": str(stock_path),
        "price_history_dir": str(history_dir),
        "adjustment": adjustment,
        "calendar_coverage": calendar_coverage,
        "stock_pool_entry_count": len(pool["symbols"]),
        "stock_pool_unique_symbol_count": len(unique_symbols),
        "stock_pool_duplicate_symbol_count": pool["duplicate_symbol_count"],
        "missing_cache_symbol_count": int(symbol_table["status"].eq("missing_cache").sum()) if not symbol_table.empty else 0,
        "invalid_data_symbol_count": int(symbol_table["status"].eq("invalid_data").sum()) if not symbol_table.empty else 0,
        "warning_symbol_count": int(symbol_table["status"].eq("warning").sum()) if not symbol_table.empty else 0,
        "issue_count": int(len(issue_table)),
        "reference_trading_day_count": len(reference_dates),
        "reference_calendar_required_symbol_count": required_coverage,
        "reference_calendar_start_date": _format_date(min(reference_dates)) if reference_dates else "",
        "reference_calendar_end_date": _format_date(max(reference_dates)) if reference_dates else "",
    }


def _issue(symbol, issue, severity, count, detail):
    return {"symbol": symbol, "issue": issue, "severity": severity, "count": count, "detail": detail}


def _format_dates(dates):
    return ",".join(_format_date(date_value) for date_value in dates[:_DATE_SAMPLE_SIZE])


def _format_date(date_value):
    return pd.Timestamp(date_value).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="审计 HS300 股票池及本地行情缓存")
    parser.add_argument("--stock-file", default=str(DEFAULT_STOCK_FILE), help="股票池文本文件")
    parser.add_argument("--price-history-dir", default=str(DEFAULT_PRICE_HISTORY_DIR), help="本地 price_history 目录")
    parser.add_argument("--adjustment", default="raw_hfq_pct", help="缓存复权口径目录")
    parser.add_argument("--calendar-coverage", type=float, default=0.8, help="参考交易日最低覆盖比例")
    parser.add_argument("--output-dir", default="output", help="报告输出目录")
    args = parser.parse_args()

    report = audit_hs300_data(
        args.stock_file,
        args.price_history_dir,
        args.adjustment,
        args.calendar_coverage,
    )
    paths = write_audit_report(report, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("审计报告已写入：{}".format(", ".join(str(path) for path in paths.values())))


if __name__ == "__main__":
    main()
