"""Fetch the ``ts_code -> name`` mapping, which the repo otherwise does not keep.

Every other stage of the pipeline works in bare six-digit codes and the price
cache carries no ``name`` column, so a pick list rendered without this lookup
shows ``600519`` where a human needs ``贵州茅台``.  ``data/data_get_stocklist.py``
queries ``stock_basic`` with the ``name`` field and then throws it away, so there
is nothing on disk to reuse.

One call returns the full listed universe (plus delisted rows under
``list_status='L,D,P'``), which matters here: the names a 2026 selection needs
include ones delisted since.  The frame is small, so it is cached whole::

    data/raw/tushare/stock_basic/stock_basic.csv

Example::

    python -m research.experiments.fetch_tushare_stock_basic
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from research.experiments.fetch_tushare_daily_basic import call_api, read_token

DEFAULT_OUTPUT = "data/raw/tushare/stock_basic/stock_basic.csv"
FIELDS = "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status"


def fetch_stock_basic(token: str, list_status: str = "L,D,P") -> pd.DataFrame:
    frame = call_api("stock_basic", token, {"list_status": list_status}, FIELDS, retries=5)
    if frame.empty:
        raise RuntimeError("stock_basic returned no rows")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.zfill(6)
    frame["name"] = frame["name"].astype(str).str.strip()
    return frame.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def load_stock_names(path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Cached lookup for consumers.  Returns an empty frame if never fetched.

    Callers are expected to fall back to bare codes rather than fail: the name
    table is presentation only and must never gate the selection itself.
    """

    table = Path(path)
    if not table.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    frame = pd.read_csv(table, dtype={"symbol": str}, low_memory=False)
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--list-status", default="L,D,P",
                        help="L=listed, D=delisted, P=paused; a 2026 pick list may cite any of them.")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    token = read_token(Path(args.env_file))
    frame = fetch_stock_basic(token, args.list_status)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    named = frame["name"].notna().sum()
    print("stock_basic: {} rows -> {}  ({} with a name)".format(len(frame), output, named))
    print(frame.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
