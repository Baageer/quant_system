"""
Fetch sector/industry index-like codes from yfinance native interfaces.

This script uses yfinance domain APIs:
- yfinance.Sector(sector_key)
- yfinance.Industry(industry_key) (fallback to resolve missing industry symbol)

Outputs:
1. industry_sector_index_list.csv
2. industry_sector_index_codes.txt
3. industry_sector_index_code_name_map.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import re
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_SECTOR_KEYS = [
    "basic-materials",
    "communication-services",
    "consumer-cyclical",
    "consumer-defensive",
    "energy",
    "financial-services",
    "healthcare",
    "industrials",
    "real-estate",
    "technology",
    "utilities",
]


def resolve_repo_root() -> Path:
    """Resolve repository root from current script location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_values(raw_text: str, lower: bool = False, upper: bool = False) -> list[str]:
    values = [item.strip() for item in str(raw_text).split(",") if item.strip()]
    if lower:
        values = [item.lower() for item in values]
    if upper:
        values = [item.upper() for item in values]
    return values


def validate_date_ymd(date_text: str) -> str:
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {date_text}, expected format YYYY-MM-DD"
        ) from exc
    return date_text


def get_column_name(candidates: Iterable[str], columns: Iterable[str]) -> str:
    columns_set = set(columns)
    for name in candidates:
        if name in columns_set:
            return name
    raise KeyError(f"Unable to find any columns in candidates: {list(candidates)}")


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is not installed. Install it first: pip install pandas"
        ) from exc
    return pd


def require_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "yfinance is not installed. Install it first: pip install yfinance"
        ) from exc
    return yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch yfinance sector/industry index-like codes."
    )
    parser.add_argument(
        "--sector-keys",
        type=str,
        default=",".join(DEFAULT_SECTOR_KEYS),
        help="Comma-separated yfinance sector keys.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="both",
        choices=["sector", "industry", "both"],
        help="Fetch sector level, industry level, or both.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/yfinance/industry_sector_index",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--resolve-industry-symbol",
        action="store_true",
        help="When industry rows miss symbol, try yfinance.Industry(industry_key).symbol.",
    )
    parser.add_argument(
        "--keep-empty-code",
        action="store_true",
        help="Keep rows without resolved index_code.",
    )
    parser.add_argument(
        "--validate-online",
        action="store_true",
        help="Validate each resolved code by requesting recent history.",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="When used with --validate-online, only export validated rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry attempts per request.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=0.8,
        help="Base wait time before retry. Actual wait = base * attempt.",
    )
    parser.add_argument(
        "--download-daily",
        action="store_true",
        help="Download daily history for each resolved index_code.",
    )
    parser.add_argument(
        "--daily-start",
        type=validate_date_ymd,
        default=None,
        help="Daily history start date in YYYY-MM-DD. Required with --download-daily.",
    )
    parser.add_argument(
        "--daily-end",
        type=validate_date_ymd,
        default=None,
        help="Daily history end date in YYYY-MM-DD (inclusive). Required with --download-daily.",
    )
    parser.add_argument(
        "--daily-output-subdir",
        type=str,
        default="daily",
        help="Sub-directory under output dir for daily CSV files.",
    )
    parser.add_argument(
        "--skip-existing-daily",
        action="store_true",
        help="Skip daily download when target csv already exists.",
    )
    return parser.parse_args()


def safe_str(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def looks_like_ticker(text: str) -> bool:
    if not text:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9\.^=\-]{1,20}", text))


def payload_to_frame(pd, payload):
    if payload is None:
        return pd.DataFrame()
    if hasattr(payload, "copy") and hasattr(payload, "columns"):
        return payload.copy()
    if isinstance(payload, dict):
        if not payload:
            return pd.DataFrame()
        values = list(payload.values())
        if values and all(isinstance(v, dict) for v in values):
            frame = pd.DataFrame.from_dict(payload, orient="index").reset_index()
            frame = frame.rename(columns={"index": "key"})
            return frame
        if values and all(not isinstance(v, (dict, list, tuple)) for v in values):
            return pd.DataFrame([payload])
        if values and all(isinstance(v, (list, tuple)) for v in values):
            rows = []
            for key, seq in payload.items():
                rows.append({"key": key, "value": seq})
            return pd.DataFrame(rows)
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload)
    return pd.DataFrame()


def first_candidate_value(row, candidates: list[str]) -> str:
    for col in candidates:
        if col in row:
            val = safe_str(row[col])
            if val:
                return val
    return ""


def extract_symbol_from_table(pd, payload) -> str:
    frame = payload_to_frame(pd, payload)
    if frame.empty:
        return ""

    candidate_cols = [
        "symbol",
        "ticker",
        "code",
        "index_code",
        "ts_code",
        "Symbol",
        "Ticker",
    ]
    for col in candidate_cols:
        if col in frame.columns:
            for raw in frame[col].tolist():
                symbol = safe_str(raw).upper()
                if looks_like_ticker(symbol):
                    return symbol

    for idx_val in frame.index.tolist():
        symbol = safe_str(idx_val).upper()
        if looks_like_ticker(symbol):
            return symbol
    return ""


def fetch_sector_rows(
    pd,
    yf,
    sector_keys: list[str],
    include_sector: bool,
    include_industry: bool,
    resolve_industry_symbol: bool,
    sleep_seconds: float,
) -> tuple[object, list[str]]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    industry_cls = getattr(yf, "Industry", None)

    for idx, sector_key in enumerate(sector_keys):
        print(f"[{idx + 1}/{len(sector_keys)}] Fetching sector: {sector_key}")
        try:
            sector_obj = yf.Sector(sector_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sector={sector_key}\tinit_error={repr(exc)}")
            time.sleep(max(sleep_seconds, 0.0))
            continue

        sector_name = safe_str(getattr(sector_obj, "name", "")) or sector_key
        sector_symbol = safe_str(getattr(sector_obj, "symbol", "")).upper()
        if not sector_symbol:
            sector_symbol = extract_symbol_from_table(pd, getattr(sector_obj, "top_etfs", None))

        if include_sector:
            rows.append(
                {
                    "index_code": sector_symbol,
                    "industry_name": sector_name,
                    "level": "L1",
                    "src": "yfinance.Sector",
                    "category": "sector",
                    "sector_key": sector_key,
                    "industry_key": "",
                }
            )

        if include_industry:
            industries_payload = getattr(sector_obj, "industries", None)
            industries_df = payload_to_frame(pd, industries_payload)
            if industries_df.empty:
                errors.append(f"sector={sector_key}\tindustries_empty=True")
            else:
                for _, raw_row in industries_df.iterrows():
                    row_obj = raw_row.to_dict()
                    industry_key = first_candidate_value(
                        row_obj,
                        ["key", "industry_key", "industryKey", "slug", "id"],
                    )
                    industry_name = first_candidate_value(
                        row_obj,
                        ["name", "industry_name", "title", "industry"],
                    )
                    industry_symbol = first_candidate_value(
                        row_obj,
                        ["symbol", "ticker", "code", "index_code", "ts_code"],
                    ).upper()

                    if not industry_name:
                        industry_name = industry_key or sector_name

                    if (
                        resolve_industry_symbol
                        and not industry_symbol
                        and industry_key
                        and callable(industry_cls)
                    ):
                        try:
                            industry_obj = industry_cls(industry_key)
                            industry_symbol = safe_str(getattr(industry_obj, "symbol", "")).upper()
                            if not industry_symbol:
                                industry_symbol = extract_symbol_from_table(
                                    pd, getattr(industry_obj, "top_performing_companies", None)
                                )
                        except Exception as exc:  # noqa: BLE001
                            errors.append(
                                f"sector={sector_key}\tindustry_key={industry_key}\t"
                                f"resolve_symbol_error={repr(exc)}"
                            )

                    rows.append(
                        {
                            "index_code": industry_symbol,
                            "industry_name": industry_name,
                            "level": "L2",
                            "src": "yfinance.Sector.industries",
                            "category": "industry",
                            "sector_key": sector_key,
                            "industry_key": industry_key,
                        }
                    )

        time.sleep(max(sleep_seconds, 0.0))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, errors

    frame["index_code"] = frame["index_code"].astype(str).str.strip().str.upper()
    frame["industry_name"] = frame["industry_name"].astype(str).str.strip()
    frame["level"] = frame["level"].astype(str).str.strip()
    frame["src"] = frame["src"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    frame["sector_key"] = frame["sector_key"].astype(str).str.strip()
    frame["industry_key"] = frame["industry_key"].astype(str).str.strip()
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame, errors


def validate_codes_online(
    yf,
    data,
    sleep_seconds: float,
    max_retries: int,
    retry_wait_seconds: float,
):
    data = data.copy()
    data["is_valid"] = False
    data["error"] = ""

    total = len(data)
    for idx, row in data.iterrows():
        code = safe_str(row.get("index_code", "")).upper()
        if not code:
            data.at[idx, "error"] = "empty code"
            continue

        print(f"[{idx + 1}/{total}] Validating {code} ...")
        success = False
        last_error = ""

        for attempt in range(1, max(max_retries, 1) + 1):
            try:
                history = yf.Ticker(code).history(period="5d", auto_adjust=False)
                if history is not None and not history.empty:
                    success = True
                    last_error = ""
                    break
                last_error = "empty history"
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)

            if attempt < max(max_retries, 1):
                wait_seconds = max(retry_wait_seconds, 0.0) * attempt
                print(
                    f"  Retry {attempt}/{max(max_retries, 1) - 1} for {code} after "
                    f"{wait_seconds:.1f}s: {last_error}"
                )
                time.sleep(wait_seconds)

        data.at[idx, "is_valid"] = bool(success)
        data.at[idx, "error"] = "" if success else last_error
        time.sleep(max(sleep_seconds, 0.0))

    return data


def download_daily_history(
    pd,
    yf,
    data,
    output_dir: Path,
    daily_start: str,
    daily_end: str,
    sleep_seconds: float,
    max_retries: int,
    retry_wait_seconds: float,
    daily_output_subdir: str,
    skip_existing_daily: bool,
) -> tuple[int, int, int, Path]:
    data = data.copy()

    start_dt = datetime.strptime(daily_start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(daily_end, "%Y-%m-%d").date()
    end_exclusive = end_dt + timedelta(days=1)

    code_col = get_column_name(["index_code", "ts_code", "code"], data.columns)
    name_col = "industry_name" if "industry_name" in data.columns else "name"

    data[code_col] = data[code_col].astype(str).str.strip().str.upper()
    data = data[data[code_col] != ""].drop_duplicates(subset=[code_col]).reset_index(drop=True)

    daily_dir = output_dir / daily_output_subdir
    ensure_dir(daily_dir)

    total = len(data)
    success = 0
    skipped = 0
    failed_messages: list[str] = []
    start_compact = start_dt.strftime("%Y%m%d")
    end_compact = end_dt.strftime("%Y%m%d")

    for idx, row in data.iterrows():
        code = safe_str(row.get(code_col, "")).upper()
        name = safe_str(row.get(name_col, ""))
        csv_path = daily_dir / f"{code}_{start_compact}_{end_compact}.csv"

        if skip_existing_daily and csv_path.exists():
            skipped += 1
            print(f"[{idx + 1}/{total}] Skip existing daily csv: {code}")
            continue

        print(f"[{idx + 1}/{total}] Downloading daily history: {code} {name}")
        last_error = ""
        saved = False

        for attempt in range(1, max(max_retries, 1) + 1):
            try:
                history = yf.Ticker(code).history(
                    start=start_dt.isoformat(),
                    end=end_exclusive.isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )

                if history is None or history.empty:
                    last_error = "empty history"
                else:
                    hist_df = history.copy()
                    hist_df = hist_df.reset_index()

                    first_col = hist_df.columns[0]
                    hist_df = hist_df.rename(columns={first_col: "trade_date"})
                    hist_df["trade_date"] = pd.to_datetime(hist_df["trade_date"]).dt.strftime(
                        "%Y-%m-%d"
                    )
                    hist_df.insert(0, "index_code", code)
                    hist_df.insert(1, "industry_name", name)

                    hist_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    success += 1
                    saved = True
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)

            if attempt < max(max_retries, 1):
                wait_seconds = max(retry_wait_seconds, 0.0) * attempt
                print(
                    f"  Retry {attempt}/{max(max_retries, 1) - 1} for {code} after "
                    f"{wait_seconds:.1f}s: {last_error}"
                )
                time.sleep(wait_seconds)

        if not saved:
            failed_message = f"{code}\t{name}\t{last_error}"
            failed_messages.append(failed_message)
            print(f"  Failed: {failed_message}")

        time.sleep(max(sleep_seconds, 0.0))

    failed_path = daily_dir / "daily_history_failed.txt"
    failed_path.write_text("\n".join(failed_messages), encoding="utf-8")
    return total, success, skipped, failed_path


def write_outputs(data, output_dir: Path) -> None:
    code_col = get_column_name(["index_code", "ts_code", "code"], data.columns)
    name_col = get_column_name(["industry_name", "name"], data.columns)
    level_col = "level" if "level" in data.columns else "L1"
    src_col = "src" if "src" in data.columns else "yfinance"

    listing_csv_path = output_dir / "industry_sector_index_list.csv"
    data.to_csv(listing_csv_path, index=False, encoding="utf-8-sig")

    unique_codes = data[code_col].astype(str).str.strip()
    unique_codes = [code for code in unique_codes.drop_duplicates().tolist() if code]
    codes_txt_path = output_dir / "industry_sector_index_codes.txt"
    codes_txt_path.write_text("\n".join(unique_codes), encoding="utf-8")

    mapping_path = output_dir / "industry_sector_index_code_name_map.txt"
    mapping_lines = [
        f"{row[code_col]}\t{row[name_col]}\t{row[level_col]}\t{row[src_col]}"
        for _, row in data.iterrows()
        if safe_str(row.get(code_col, ""))
    ]
    mapping_path.write_text("\n".join(mapping_lines), encoding="utf-8")

    print("")
    print("Saved files:")
    print(f"- {listing_csv_path}")
    print(f"- {codes_txt_path}")
    print(f"- {mapping_path}")


def write_errors(errors: list[str], output_dir: Path) -> None:
    error_path = output_dir / "industry_sector_index_failed.txt"
    error_path.write_text("\n".join(errors), encoding="utf-8")
    print(f"- {error_path}")


def main() -> int:
    args = parse_args()

    sector_keys = parse_csv_values(args.sector_keys, lower=True)
    if not sector_keys:
        print("No valid --sector-keys provided.", file=sys.stderr)
        return 1

    include_sector = args.scope in {"sector", "both"}
    include_industry = args.scope in {"industry", "both"}

    if args.download_daily:
        if not args.daily_start or not args.daily_end:
            print(
                "When --download-daily is enabled, both --daily-start and --daily-end are required.",
                file=sys.stderr,
            )
            return 6
        start_dt = datetime.strptime(args.daily_start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(args.daily_end, "%Y-%m-%d").date()
        if start_dt > end_dt:
            print("--daily-start must be <= --daily-end.", file=sys.stderr)
            return 7

    try:
        pd = require_pandas()
        yf = require_yfinance()
    except Exception as exc:  # noqa: BLE001
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 2

    output_df, fetch_errors = fetch_sector_rows(
        pd=pd,
        yf=yf,
        sector_keys=sector_keys,
        include_sector=include_sector,
        include_industry=include_industry,
        resolve_industry_symbol=bool(args.resolve_industry_symbol),
        sleep_seconds=max(args.sleep_seconds, 0.0),
    )

    if output_df is None or output_df.empty:
        print("No sector/industry rows fetched from yfinance.", file=sys.stderr)
        repo_root = resolve_repo_root()
        output_dir = (repo_root / args.output_dir).resolve()
        ensure_dir(output_dir)
        write_errors(fetch_errors, output_dir)
        return 3

    if not args.keep_empty_code:
        output_df = output_df[output_df["index_code"].astype(str).str.strip() != ""].reset_index(drop=True)

    if output_df.empty:
        print("All rows have empty codes after filtering.", file=sys.stderr)
        return 4

    if args.validate_online:
        output_df = validate_codes_online(
            yf=yf,
            data=output_df,
            sleep_seconds=max(args.sleep_seconds, 0.0),
            max_retries=max(args.max_retries, 1),
            retry_wait_seconds=max(args.retry_wait_seconds, 0.0),
        )
        if args.only_valid:
            output_df = output_df[output_df["is_valid"]].reset_index(drop=True)

    repo_root = resolve_repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(output_dir)

    try:
        write_outputs(output_df, output_dir)
        write_errors(fetch_errors, output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write output files: {exc}", file=sys.stderr)
        return 5

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Total rows: {len(output_df)}")
    print(f"Unique codes: {output_df['index_code'].astype(str).str.strip().nunique()}")
    print(f"Categories: {sorted(output_df['category'].astype(str).str.strip().unique().tolist())}")
    if "is_valid" in output_df.columns:
        print(f"Valid rows: {int(output_df['is_valid'].sum())}")
        print(f"Invalid rows: {int((~output_df['is_valid']).sum())}")
    if fetch_errors:
        print(f"Fetch warnings/errors: {len(fetch_errors)} (see industry_sector_index_failed.txt)")

    if args.download_daily:
        print("")
        print(
            f"Downloading daily history for each index: {args.daily_start} -> {args.daily_end} "
            f"(inclusive)"
        )
        total, success, skipped, daily_failed_path = download_daily_history(
            pd=pd,
            yf=yf,
            data=output_df,
            output_dir=output_dir,
            daily_start=args.daily_start,
            daily_end=args.daily_end,
            sleep_seconds=max(args.sleep_seconds, 0.0),
            max_retries=max(args.max_retries, 1),
            retry_wait_seconds=max(args.retry_wait_seconds, 0.0),
            daily_output_subdir=args.daily_output_subdir,
            skip_existing_daily=bool(args.skip_existing_daily),
        )
        print("")
        print("Daily history summary:")
        print(f"- Total codes: {total}")
        print(f"- Success: {success}")
        print(f"- Skipped: {skipped}")
        print(f"- Failed: {total - success - skipped}")
        print(f"- Failed details: {daily_failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
