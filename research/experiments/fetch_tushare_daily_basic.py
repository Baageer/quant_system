"""Fetch tushare ``daily_basic`` daily valuation/liquidity panels, one CSV per symbol.

``daily_basic`` carries the point-in-time cross-section the frozen 28 features are
missing entirely: market cap (``total_mv`` / ``circ_mv``), turnover
(``turnover_rate`` / ``turnover_rate_f``) and valuation (``pe_ttm`` / ``pb`` /
``ps_ttm`` / ``dv_ttm``).  It is end-of-day data computed from the *latest
disclosed* financials, so joining it on the signal date ``t`` introduces no
restatement look-ahead.

This intentionally talks to the tushare HTTP endpoint with the standard library
only -- the ``tushare`` / ``requests`` packages are not installed in this
environment, and the payload is a plain JSON POST.

Cached layout mirrors the price-history convention::

    data/raw/tushare/daily_basic/600519.SH.csv

Example::

    python -m research.experiments.fetch_tushare_daily_basic --limit 5   # smoke
    python -m research.experiments.fetch_tushare_daily_basic             # full universe
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from data.data_api import DataAPI
from research.market_regime.tradability_labels import normalize_a_share_symbols


API_URL = "https://api.tushare.pro"
DEFAULT_OUTPUT_DIR = "data/raw/tushare/daily_basic"
DEFAULT_DATASET_DIR = "output/datasets"
DEFAULT_DATASET_ID = "excess_hs300_v1"
DEFAULT_START_DATE = "20050101"
DEFAULT_END_DATE = "20251231"
SPLITS = ("train", "validation", "test")

DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
    "total_share,float_share,free_share,total_mv,circ_mv"
)


def read_token(env_file: Path = Path(".env")) -> str:
    """Read TS_TOKEN from the repo .env without requiring python-dotenv."""

    if not env_file.exists():
        raise FileNotFoundError("No .env found at {}; set TS_TOKEN there".format(env_file))
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*TS_TOKEN\s*=\s*[\"']?([^\"'\n]+)", line)
        if match:
            return match.group(1).strip()
    raise ValueError("TS_TOKEN not found in {}".format(env_file))


def call_api(api_name: str, token: str, params: Dict[str, object], fields: str = "",
             retries: int = 4, backoff: float = 5.0, timeout: int = 60) -> pd.DataFrame:
    """POST one tushare pro query and return its payload as a DataFrame."""

    payload = json.dumps({"api_name": api_name, "token": token, "params": params, "fields": fields})
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                API_URL, data=payload.encode("utf-8"), headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = "network: {}".format(exc)
        else:
            if body.get("code") == 0:
                data = body.get("data") or {}
                fields_out = data.get("fields") or []
                items = data.get("items") or []
                return pd.DataFrame(items, columns=fields_out)
            last_error = "api code={} msg={}".format(body.get("code"), body.get("msg"))
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError("tushare {} failed after {} retries: {}".format(api_name, retries, last_error))


def dataset_symbols(dataset_dir: Path, dataset_id: str) -> List[str]:
    """Union of six-digit symbols present across the dataset's three splits."""

    symbols = set()
    for split in SPLITS:
        path = dataset_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
        if not path.exists():
            raise FileNotFoundError("Missing sample split: {}".format(path))
        frame = pd.read_csv(path, usecols=["symbol"], dtype={"symbol": str}, low_memory=False)
        symbols.update(normalize_a_share_symbols(frame["symbol"]).unique())
    return sorted(symbols)


def fetch_symbol(
    ts_code: str, token: str, output_dir: Path, start_date: str, end_date: str,
    force: bool = False, sleep: float = 0.35,
) -> Dict[str, object]:
    """Fetch (or reuse) one symbol's daily_basic history."""

    path = output_dir / "{}.csv".format(ts_code)
    if path.exists() and not force:
        existing = pd.read_csv(path, low_memory=False)
        if not existing.empty:
            return {"ts_code": ts_code, "status": "cached", "rows": int(len(existing)), "path": str(path)}

    frame = call_api(
        "daily_basic", token,
        {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
        fields=DAILY_BASIC_FIELDS,
    )
    if sleep:
        time.sleep(sleep)
    if frame.empty:
        return {"ts_code": ts_code, "status": "empty", "rows": 0, "path": None}

    frame = frame.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {"ts_code": ts_code, "status": "fetched", "rows": int(len(frame)), "path": str(path)}


def fetch_universe(
    symbols: Sequence[str], token: str, output_dir: Path, start_date: str, end_date: str,
    force: bool = False, sleep: float = 0.35, limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch every symbol, reporting per-symbol status instead of aborting on one failure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(symbols)[: int(limit)] if limit else list(symbols)
    records = []
    for symbol in tqdm(selected, desc="Fetching daily_basic", unit="symbol"):
        ts_code = DataAPI._normalize_symbol_for_tushare(symbol)
        try:
            records.append(fetch_symbol(ts_code, token, output_dir, start_date, end_date, force=force, sleep=sleep))
        except Exception as exc:  # keep going; one bad symbol must not kill the run
            records.append({"ts_code": ts_code, "status": "failed", "rows": 0, "path": None, "error": str(exc)[:200]})
    return pd.DataFrame(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--sleep", type=float, default=0.35, help="seconds between API calls")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N symbols (smoke test)")
    parser.add_argument("--force", action="store_true", help="refetch symbols that already have a cache file")
    parser.add_argument("--symbols", default=None, help="comma-separated six-digit symbols instead of the dataset universe")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    token = read_token(Path(args.env_file))
    if args.symbols:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    else:
        symbols = dataset_symbols(Path(args.dataset_dir), args.dataset_id)

    report = fetch_universe(
        symbols, token, Path(args.output_dir), args.start_date, args.end_date,
        force=args.force, sleep=args.sleep, limit=args.limit,
    )
    counts = report["status"].value_counts().to_dict()
    print("daily_basic fetch done: {} symbols -> {}".format(len(report), counts))
    failures = report[report["status"] == "failed"]
    if not failures.empty:
        print("Failures:\n{}".format(failures[["ts_code", "error"]].to_string(index=False)))
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
