"""
Download industry sector index codes and daily history from AkShare.

Workflow:
1. Fetch all industry sector index codes from Eastmoney via AkShare.
2. Save code list to txt files.
3. Download daily kline data for each code and save as CSV.

Default date range: 20050101 - 20251231
Default output dir: data/raw/akshare/industry_sector_index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable


def resolve_repo_root() -> Path:
    """Resolve repository root from current script location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_column_name(candidates: Iterable[str], columns: Iterable[str]) -> str:
    """Return the first matched column name from candidates."""
    columns_set = set(columns)
    for name in candidates:
        if name in columns_set:
            return name
    raise KeyError(f"Unable to find any columns in candidates: {list(candidates)}")


def validate_date(date_text: str) -> str:
    """Validate date in YYYYMMDD format."""
    if len(date_text) != 8 or not date_text.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid date: {date_text}, expected YYYYMMDD")
    return date_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AkShare industry sector index codes and daily history."
    )
    parser.add_argument(
        "--start-date",
        type=validate_date,
        default="20050101",
        help="Start date in YYYYMMDD, default: 20050101",
    )
    parser.add_argument(
        "--end-date",
        type=validate_date,
        default="20251231",
        help="End date in YYYYMMDD, default: 20251231",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/akshare/industry_sector_index",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--adjust",
        type=str,
        default="",
        choices=["", "qfq", "hfq"],
        help="Adjust mode for kline data. Default is no adjust.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Sleep seconds between requests. Default: 0.5",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for each code. Default: 3",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
        help="Base wait time before retry. Actual wait = base * attempt.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if csv already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(output_dir)

    try:
        import akshare as ak
    except ModuleNotFoundError:
        print("akshare is not installed. Install it first: pip install akshare", file=sys.stderr)
        return 1

    print("Fetching industry sector index list from AkShare...")
    industry_df = ak.stock_board_industry_name_em()
    if industry_df is None or industry_df.empty:
        print("No industry sector index list returned from AkShare.", file=sys.stderr)
        return 2

    code_col = get_column_name(
        ["\u677f\u5757\u4ee3\u7801", "\u4ee3\u7801", "code"], industry_df.columns
    )
    name_col = get_column_name(
        ["\u677f\u5757\u540d\u79f0", "\u540d\u79f0", "name"], industry_df.columns
    )

    industry_df[code_col] = industry_df[code_col].astype(str).str.strip()
    industry_df[name_col] = industry_df[name_col].astype(str).str.strip()

    listing_csv_path = output_dir / "industry_sector_index_list.csv"
    industry_df.to_csv(listing_csv_path, index=False, encoding="utf-8-sig")

    codes = industry_df[code_col].tolist()
    codes_txt_path = output_dir / "industry_sector_index_codes.txt"
    codes_txt_path.write_text("\n".join(codes), encoding="utf-8")

    mapping_txt_path = output_dir / "industry_sector_index_code_name_map.txt"
    mapping_lines = [
        f"{code}\t{name}" for code, name in zip(industry_df[code_col], industry_df[name_col])
    ]
    mapping_txt_path.write_text("\n".join(mapping_lines), encoding="utf-8")

    print(f"Saved {len(codes)} industry codes to: {codes_txt_path}")
    print(f"Downloading daily history: {args.start_date} -> {args.end_date}")

    success_codes: list[str] = []
    failed_messages: list[str] = []

    for idx, row in industry_df.iterrows():
        code = str(row[code_col]).strip()
        name = str(row[name_col]).strip()
        csv_path = output_dir / f"{code}_{args.start_date}_{args.end_date}.csv"

        if args.skip_existing and csv_path.exists():
            print(f"[{idx + 1}/{len(industry_df)}] Skip existing: {code} {name}")
            success_codes.append(code)
            continue

        print(f"[{idx + 1}/{len(industry_df)}] Downloading: {code} {name}")
        last_exc: Exception | None = None

        for attempt in range(1, args.max_retries + 1):
            try:
                hist_df = None
                fetch_error: Exception | None = None

                # Newer AkShare versions support BK code directly.
                # Some older versions only accept board name, so we fallback to name.
                for symbol in (code, name):
                    try:
                        hist_df = ak.stock_board_industry_hist_em(
                            symbol=symbol,
                            start_date=args.start_date,
                            end_date=args.end_date,
                            period="\u65e5k",
                            adjust=args.adjust,
                        )
                        time.sleep(3)
                        if hist_df is not None and not hist_df.empty:
                            break
                    except Exception as inner_exc:  # noqa: BLE001
                        fetch_error = inner_exc

                if hist_df is None or hist_df.empty:
                    if fetch_error is not None:
                        raise fetch_error
                    raise ValueError("empty result")

                hist_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                success_codes.append(code)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < args.max_retries:
                    wait_seconds = args.retry_wait_seconds * attempt
                    print(
                        f"  Retry {attempt}/{args.max_retries - 1} for {code} after "
                        f"{wait_seconds:.1f}s: {exc}"
                    )
                    time.sleep(wait_seconds)

        if last_exc is not None:
            message = f"{code}\t{name}\t{repr(last_exc)}"
            failed_messages.append(message)
            print(f"  Failed: {message}")

        time.sleep(max(args.sleep_seconds, 0.0))

    failed_path = output_dir / "industry_sector_index_failed.txt"
    failed_path.write_text("\n".join(failed_messages), encoding="utf-8")

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Total: {len(industry_df)}")
    print(f"Success: {len(success_codes)}")
    print(f"Failed: {len(failed_messages)}")
    print(f"Listing csv: {listing_csv_path}")
    print(f"Codes txt: {codes_txt_path}")
    print(f"Code-name map txt: {mapping_txt_path}")
    print(f"Failed txt: {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
