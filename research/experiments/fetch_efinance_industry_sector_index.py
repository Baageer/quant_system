"""
Fetch industry board index list from efinance and optionally download daily kline.

Default behavior:
1. Fetch and validate industry board list (code/name).
2. Save list artifacts to output directory.

Optional:
- Download daily kline for every board code with --download-daily.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable


INDUSTRY_BOARD_KEY = "行业板块"
INDUSTRY_BOARD_FS_EXPR = "m:90 t:2 f:!50"
DEFAULT_FIELDS = "f12,f14"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_date_yyyymmdd(text: str) -> str:
    if len(text) != 8 or not text.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid date: {text}, expected YYYYMMDD")
    return text


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is not installed. Install it first: pip install pandas"
        ) from exc
    return pd


def require_efinance():
    try:
        import efinance as ef  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "efinance is not installed. Install it first: pip install efinance"
        ) from exc
    return ef


def require_requests():
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "requests is not installed. Install it first: pip install requests"
        ) from exc
    return requests


def safe_str(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def pick_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns_set = set(columns)
    for name in candidates:
        if name in columns_set:
            return name
    return ""


def normalize_industry_list(pd, raw_df, source: str):
    if raw_df is None or getattr(raw_df, "empty", True):
        return pd.DataFrame(columns=["index_code", "industry_name", "src"])

    data = raw_df.copy()
    data.columns = [str(col).strip() for col in data.columns]

    code_col = pick_existing_column(
        data.columns,
        ["f12", "index_code", "code", "ticker", "stock_code", "board_code"],
    )
    name_col = pick_existing_column(
        data.columns,
        ["f14", "industry_name", "name", "board_name", "stock_name"],
    )

    if not code_col:
        code_col = data.columns[0]
    if not name_col:
        name_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]

    output = pd.DataFrame()
    output["index_code"] = data[code_col].astype(str).str.strip().str.upper()
    output["industry_name"] = data[name_col].astype(str).str.strip()
    output["src"] = source

    output = output[output["index_code"] != ""].copy()
    output = output[output["industry_name"] != ""].copy()
    output = output.drop_duplicates().reset_index(drop=True)
    return output


def fetch_list_via_efinance(pd, ef):
    # efinance uses http push2 endpoint internally for realtime list.
    # Some environments now require https, so we patch it at runtime.
    try:
        import efinance.shared as ef_shared  # type: ignore
    except Exception:
        ef_shared = None

    patched = False
    original_get = None
    if ef_shared is not None and hasattr(ef_shared, "session"):
        original_get = ef_shared.session.get

        def _patched_get(url, *args, **kwargs):
            if isinstance(url, str) and url.startswith("http://push2.eastmoney.com/"):
                url = "https://push2.eastmoney.com/" + url.split(
                    "http://push2.eastmoney.com/", 1
                )[1]
            return original_get(url, *args, **kwargs)

        ef_shared.session.get = _patched_get
        patched = True

    try:
        raw_df = ef.stock.get_realtime_quotes(fs=INDUSTRY_BOARD_KEY)
    finally:
        if patched and ef_shared is not None and original_get is not None:
            ef_shared.session.get = original_get

    return normalize_industry_list(
        pd,
        raw_df=raw_df,
        source="efinance.stock.get_realtime_quotes",
    )


def request_page_with_retry(
    session,
    url: str,
    headers: dict[str, str],
    params: dict[str, object],
    max_retries: int,
    retry_wait_seconds: float,
):
    last_error = ""
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            response = session.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not data:
                raise ValueError("response has no data field")
            diff = data.get("diff") or []
            total = int(data.get("total") or 0)
            return total, diff, ""
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < max(max_retries, 1):
                wait_seconds = max(retry_wait_seconds, 0.0) * attempt
                time.sleep(wait_seconds)
    return 0, [], last_error


def fetch_list_via_https_api(
    pd,
    requests,
    max_retries: int,
    retry_wait_seconds: float,
    sleep_seconds: float,
):
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    errors: list[str] = []
    for trust_env in (True, False):
        session = requests.Session()
        session.trust_env = trust_env

        first_params = {
            "pn": 1,
            "pz": 200,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            "fs": INDUSTRY_BOARD_FS_EXPR,
            "fields": DEFAULT_FIELDS,
        }
        total, first_rows, first_error = request_page_with_retry(
            session=session,
            url=url,
            headers=headers,
            params=first_params,
            max_retries=max_retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        if first_error:
            errors.append(f"trust_env={trust_env}: first_page_failed: {first_error}")
            continue

        rows = list(first_rows)
        page_size = max(len(first_rows), 1)
        pages = max(int(math.ceil(total / page_size)), 1)

        failed = False
        for page in range(2, pages + 1):
            params = dict(first_params)
            params["pn"] = page
            _, page_rows, page_error = request_page_with_retry(
                session=session,
                url=url,
                headers=headers,
                params=params,
                max_retries=max_retries,
                retry_wait_seconds=retry_wait_seconds,
            )
            if page_error:
                failed = True
                errors.append(f"trust_env={trust_env}: page={page} failed: {page_error}")
                break
            rows.extend(page_rows)
            time.sleep(max(sleep_seconds, 0.0))

        if failed:
            continue

        raw_df = pd.DataFrame(rows)
        output = normalize_industry_list(
            pd,
            raw_df=raw_df,
            source=f"https_api_trust_env_{trust_env}",
        )
        if not output.empty:
            return output, errors

    return pd.DataFrame(columns=["index_code", "industry_name", "src"]), errors


def validate_list(data):
    data = data.copy()
    data["is_bk_code"] = data["index_code"].astype(str).str.match(r"^BK\\d+$", na=False)
    data["name_not_empty"] = data["industry_name"].astype(str).str.strip() != ""
    return data


def write_list_outputs(data, output_dir: Path) -> None:
    list_csv = output_dir / "industry_sector_index_list.csv"
    codes_txt = output_dir / "industry_sector_index_codes.txt"
    mapping_txt = output_dir / "industry_sector_index_code_name_map.txt"

    data.to_csv(list_csv, index=False, encoding="utf-8-sig")

    unique_codes = data["index_code"].astype(str).str.strip().drop_duplicates()
    unique_codes = [code for code in unique_codes.tolist() if code]
    codes_txt.write_text("\n".join(unique_codes), encoding="utf-8")

    lines = [
        f"{row['index_code']}\t{row['industry_name']}\t{row['src']}"
        for _, row in data.iterrows()
    ]
    mapping_txt.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Saved files:")
    print(f"- {list_csv}")
    print(f"- {codes_txt}")
    print(f"- {mapping_txt}")


def write_errors(errors: list[str], output_dir: Path) -> Path:
    error_path = output_dir / "industry_sector_index_failed.txt"
    error_path.write_text("\n".join(errors), encoding="utf-8")
    print(f"- {error_path}")
    return error_path


def find_col_name(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns_set = set(columns)
    for col in candidates:
        if col in columns_set:
            return col
    return ""


def download_daily_history(
    ef,
    data,
    output_dir: Path,
    daily_start: str,
    daily_end: str,
    daily_output_subdir: str,
    fqt: int,
    max_retries: int,
    retry_wait_seconds: float,
    sleep_seconds: float,
    skip_existing_daily: bool,
) -> tuple[int, int, int, Path]:
    daily_dir = output_dir / daily_output_subdir
    ensure_dir(daily_dir)

    data = data.copy()
    data = data.drop_duplicates(subset=["index_code"]).reset_index(drop=True)

    total = len(data)
    success = 0
    skipped = 0
    failed_messages: list[str] = []

    for idx, row in data.iterrows():
        code = safe_str(row.get("index_code", "")).upper()
        name = safe_str(row.get("industry_name", ""))
        if not code:
            continue

        csv_path = daily_dir / f"{code}_{daily_start}_{daily_end}.csv"
        if skip_existing_daily and csv_path.exists():
            skipped += 1
            print(f"[{idx + 1}/{total}] Skip existing daily csv: {code}")
            continue

        print(f"[{idx + 1}/{total}] Downloading daily history: {code} {name}")
        saved = False
        last_error = ""

        for attempt in range(1, max(max_retries, 1) + 1):
            try:
                hist_df = ef.stock.get_quote_history(
                    code,
                    beg=daily_start,
                    end=daily_end,
                    klt=101,
                    fqt=fqt,
                    suppress_error=True,
                )
                if hist_df is None or hist_df.empty:
                    last_error = "empty history"
                else:
                    date_col = find_col_name(
                        hist_df.columns,
                        ["date", "trade_date", "datetime", "日期"],
                    )
                    if date_col:
                        hist_df[date_col] = hist_df[date_col].astype(str).str.strip()
                    hist_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    success += 1
                    saved = True
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)

            if attempt < max(max_retries, 1):
                wait_seconds = max(retry_wait_seconds, 0.0) * attempt
                time.sleep(wait_seconds)

        if not saved:
            msg = f"{code}\t{name}\t{last_error}"
            failed_messages.append(msg)
            print(f"  Failed: {msg}")

        time.sleep(max(sleep_seconds, 0.0))

    failed_path = daily_dir / "daily_history_failed.txt"
    failed_path.write_text("\n".join(failed_messages), encoding="utf-8")
    return total, success, skipped, failed_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch efinance industry board list and optionally daily kline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/efinance/industry_sector_index",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--list-source",
        type=str,
        default="auto",
        choices=["auto", "efinance", "https"],
        help="List source mode. auto = efinance first, then https fallback.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per request/download.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=1.0,
        help="Base wait seconds before retry. Actual wait = base * attempt.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep seconds between requests/tasks.",
    )
    parser.add_argument(
        "--download-daily",
        action="store_true",
        help="Download daily history for each index code after list fetch.",
    )
    parser.add_argument(
        "--daily-start",
        type=validate_date_yyyymmdd,
        default="20200101",
        help="Daily history start date in YYYYMMDD.",
    )
    parser.add_argument(
        "--daily-end",
        type=validate_date_yyyymmdd,
        default="20241231",
        help="Daily history end date in YYYYMMDD.",
    )
    parser.add_argument(
        "--daily-output-subdir",
        type=str,
        default="kline_daily",
        help="Sub-directory under output dir for daily CSV files.",
    )
    parser.add_argument(
        "--fqt",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Adjust mode for daily history. 0=no adjust, 1=forward, 2=backward.",
    )
    parser.add_argument(
        "--skip-existing-daily",
        action="store_true",
        help="Skip daily download when target csv already exists.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Rows to print for quick list validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pd = require_pandas()
        ef = require_efinance()
        requests = require_requests()
    except Exception as exc:  # noqa: BLE001
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 1

    repo_root = resolve_repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(output_dir)

    fetch_errors: list[str] = []
    data = pd.DataFrame(columns=["index_code", "industry_name", "src"])

    if args.list_source in {"auto", "efinance"}:
        try:
            data = fetch_list_via_efinance(pd=pd, ef=ef)
            if data.empty:
                fetch_errors.append("efinance list returned empty dataframe")
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append(f"efinance list fetch error: {repr(exc)}")

    if data.empty and args.list_source in {"auto", "https"}:
        https_data, https_errors = fetch_list_via_https_api(
            pd=pd,
            requests=requests,
            max_retries=max(args.max_retries, 1),
            retry_wait_seconds=max(args.retry_wait_seconds, 0.0),
            sleep_seconds=max(args.sleep_seconds, 0.0),
        )
        fetch_errors.extend(https_errors)
        data = https_data

    if data.empty:
        print("No industry list rows fetched.", file=sys.stderr)
        write_errors(fetch_errors, output_dir)
        return 2

    data = data.drop_duplicates(subset=["index_code"]).reset_index(drop=True)
    data = validate_list(data)

    write_list_outputs(data, output_dir)
    write_errors(fetch_errors, output_dir)

    total = len(data)
    bk_count = int(data["is_bk_code"].sum()) if "is_bk_code" in data.columns else 0
    empty_name_count = int((~data["name_not_empty"]).sum()) if "name_not_empty" in data.columns else 0
    unique_codes = data["index_code"].astype(str).str.strip().nunique()

    print("")
    print("List summary:")
    print(f"- Total rows: {total}")
    print(f"- Unique codes: {unique_codes}")
    print(f"- BK-like codes: {bk_count}")
    print(f"- Empty names: {empty_name_count}")
    print(f"- Source tags: {sorted(data['src'].astype(str).unique().tolist())}")
    print("")
    print(f"List sample (top {max(args.sample_size, 1)}):")
    print(data.head(max(args.sample_size, 1)).to_string(index=False))

    if args.download_daily:
        print("")
        print(f"Downloading daily history for each index: {args.daily_start} -> {args.daily_end}")
        total_d, success_d, skipped_d, daily_failed_path = download_daily_history(
            ef=ef,
            data=data,
            output_dir=output_dir,
            daily_start=args.daily_start,
            daily_end=args.daily_end,
            daily_output_subdir=args.daily_output_subdir,
            fqt=args.fqt,
            max_retries=max(args.max_retries, 1),
            retry_wait_seconds=max(args.retry_wait_seconds, 0.0),
            sleep_seconds=max(args.sleep_seconds, 0.0),
            skip_existing_daily=bool(args.skip_existing_daily),
        )
        print("")
        print("Daily history summary:")
        print(f"- Total codes: {total_d}")
        print(f"- Success: {success_d}")
        print(f"- Skipped: {skipped_d}")
        print(f"- Failed: {total_d - success_d - skipped_d}")
        print(f"- Failed details: {daily_failed_path}")

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
