"""
Download THS industry board list and index data from AkShare.

Workflow:
1. Fetch all industry boards with `stock_board_industry_name_ths()`.
2. Save board list (name/code/link) to csv and txt artifacts.
3. Fetch board index history with `stock_board_industry_index_ths()` and save per-board csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import sys
import time
from pathlib import Path
from typing import Iterable


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_date_yyyymmdd(text: str) -> str:
    if len(text) != 8 or not text.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid date: {text}, expected YYYYMMDD")
    return text


def safe_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def pick_col(columns: Iterable[str], candidates: Iterable[str]) -> str:
    cols = {str(col).strip(): col for col in columns}
    for candidate in candidates:
        if candidate in cols:
            return cols[candidate]
    return ""


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch THS industry board list/index data via AkShare."
    )
    parser.add_argument(
        "--start-date",
        type=validate_date_yyyymmdd,
        default="20050101",
        help="Start date in YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        type=validate_date_yyyymmdd,
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/akshare/industry_sector_index",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3,
        help="Sleep seconds between board requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for each board.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=1.0,
        help="Base wait seconds before retry. Actual wait = base * attempt.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip board csv download if target file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start_date > args.end_date:
        print("--start-date must be <= --end-date", file=sys.stderr)
        return 1

    repo_root = resolve_repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(output_dir)

    try:
        import akshare as ak
    except ModuleNotFoundError:
        print("akshare is not installed. Install it first: pip install akshare", file=sys.stderr)
        return 2

    print("Fetching THS industry board list from AkShare...")
    industry_df = ak.stock_board_industry_name_ths()
    if industry_df is None or industry_df.empty:
        print("No industry board list returned by stock_board_industry_name_ths().", file=sys.stderr)
        return 3

    name_col = pick_col(
        industry_df.columns,
        ["name", "industry_name", "board_name", "\u540d\u79f0", "\u677f\u5757\u540d\u79f0"],
    )
    code_col = pick_col(
        industry_df.columns,
        ["code", "industry_code", "board_code", "\u4ee3\u7801", "\u677f\u5757\u4ee3\u7801"],
    )

    if not name_col:
        name_col = industry_df.columns[0]
    if not code_col:
        code_col = ""

    list_df = industry_df.copy()
    list_df.columns = [str(col).strip() for col in list_df.columns]

    if name_col != "industry_name":
        list_df["industry_name"] = list_df[name_col].map(safe_str)
    else:
        list_df["industry_name"] = list_df[name_col].map(safe_str)

    if code_col:
        list_df["index_code"] = list_df[code_col].map(safe_str)
    else:
        list_df["index_code"] = ""

    list_df["link"] = list_df["index_code"].map(
        lambda x: f"https://q.10jqka.com.cn/thshy/detail/code/{x}/" if x else ""
    )
    list_df["src"] = "ak.stock_board_industry_name_ths"

    list_df = list_df[list_df["industry_name"] != ""].copy()
    list_df = list_df.drop_duplicates(subset=["industry_name"]).reset_index(drop=True)
    list_df = list_df[["industry_name", "index_code", "link", "src"]]

    listing_csv_path = output_dir / "industry_sector_index_list.csv"
    list_df.to_csv(listing_csv_path, index=False, encoding="utf-8-sig")

    codes_txt_path = output_dir / "industry_sector_index_codes.txt"
    codes = [code for code in list_df["index_code"].tolist() if code]
    codes_txt_path.write_text("\n".join(codes), encoding="utf-8")

    mapping_txt_path = output_dir / "industry_sector_index_code_name_map.txt"
    mapping_lines = [
        f"{row['index_code']}\t{row['industry_name']}\t{row['link']}" for _, row in list_df.iterrows()
    ]
    mapping_txt_path.write_text("\n".join(mapping_lines), encoding="utf-8")

    print(f"List rows: {len(list_df)}")
    print(f"Saved list csv: {listing_csv_path}")
    print(f"Saved codes txt: {codes_txt_path}")
    print(f"Saved code-name-link txt: {mapping_txt_path}")
    print(f"Downloading THS index data: {args.start_date} -> {args.end_date}")

    success = 0
    skipped = 0
    failed_messages: list[str] = []

    total = len(list_df)
    for idx, row in list_df.iterrows():
        industry_name = safe_str(row.get("industry_name"))
        index_code = safe_str(row.get("index_code"))
        link = safe_str(row.get("link"))

        file_prefix = sanitize_filename(index_code if index_code else industry_name)
        csv_path = output_dir / f"{file_prefix}_{args.start_date}_{args.end_date}.csv"

        if args.skip_existing and csv_path.exists():
            skipped += 1
            print(f"[{idx + 1}/{total}] Skip existing: {industry_name}")
            continue

        print(f"[{idx + 1}/{total}] Downloading: {industry_name} ({index_code})")
        last_error: Exception | None = None

        for attempt in range(1, max(args.max_retries, 1) + 1):
            try:
                hist_df = ak.stock_board_industry_index_ths(
                    symbol=industry_name,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                if hist_df is None or hist_df.empty:
                    raise ValueError("empty dataframe")

                hist_out = hist_df.copy()
                hist_out.insert(0, "industry_name", industry_name)
                hist_out.insert(1, "index_code", index_code)
                hist_out.insert(2, "source_link", link)
                hist_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
                success += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max(args.max_retries, 1):
                    wait_seconds = max(args.retry_wait_seconds, 0.0) * attempt
                    print(
                        f"  Retry {attempt}/{max(args.max_retries, 1) - 1} after "
                        f"{wait_seconds:.1f}s: {exc}"
                    )
                    time.sleep(wait_seconds)

        if last_error is not None:
            failed_msg = f"{index_code}\t{industry_name}\t{link}\t{repr(last_error)}"
            failed_messages.append(failed_msg)
            print(f"  Failed: {failed_msg}")

        time.sleep(max(args.sleep_seconds, 0.0))

    failed_path = output_dir / "industry_sector_index_failed.txt"
    failed_path.write_text("\n".join(failed_messages), encoding="utf-8")

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Total boards: {total}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed_messages)}")
    print(f"Failed details: {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
