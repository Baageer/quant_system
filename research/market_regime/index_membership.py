"""Point-in-time index membership, replacing the static ``data/HS300.txt`` snapshot.

The pipeline's universe is frozen once, at ``feature_event_study`` time, from a
file that holds *today's* constituents, and is then carried downstream purely as
the ``symbol`` column.  Every stage therefore sees a survivorship/look-ahead
universe.  This module is the missing piece: it turns the monthly
``index_weight`` snapshots fetched by
``research.experiments.fetch_tushare_index_weight`` into a membership test that
any stage can apply to its own ``(date, symbol)`` rows.

Causality and granularity
-------------------------
Membership at date ``t`` is the constituent set of the **most recent snapshot
with ``trade_date`` <= ``t``** (``lag_days=0``).  Snapshots are month-end, so a
semi-annual adjustment effective mid-June is only reflected from the 30-June
snapshot onward.  That makes the universe **stale by up to one month**, never
look-ahead: we under-react to additions and keep removals a little too long.
Tightening it to the exact effective date needs the adjustment announcements
(``index_member`` returns empty for this token), so month-end granularity is the
honest ceiling here -- document it, do not hide it.

Example::

    from research.market_regime.index_membership import membership_mask
    mask = membership_mask(scores, "hs300")      # scores has date + symbol columns
    pit = scores[mask]
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERSHIP_DIR = REPO_ROOT / "data" / "raw" / "tushare" / "index_weight"

#: pool name used across the research chain -> tushare index code
INDEX_CODE_BY_POOL: Dict[str, str] = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
}

#: pool name -> the static snapshot file that is currently used as the universe
STATIC_POOL_FILE: Dict[str, str] = {
    "hs300": "data/HS300.txt",
    "zz500": "data/ZZ_500.txt",
    "zz1000": "data/ZZ_1000.txt",
}


def membership_path(index_code: str, membership_dir: Optional[Path] = None) -> Path:
    return Path(membership_dir or DEFAULT_MEMBERSHIP_DIR) / "{}.csv".format(index_code)


def load_index_weight(index_code: str, membership_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the cached monthly snapshots as ``(trade_date, symbol, weight)``."""

    path = membership_path(index_code, membership_dir)
    if not path.exists():
        raise FileNotFoundError(
            "缺少 {} 的成分权重缓存：{}\n"
            "先跑：python -m research.experiments.fetch_tushare_index_weight --indexes {}".format(
                index_code, path, index_code)
        )
    frame = pd.read_csv(path, dtype={"con_code": str}, low_memory=False)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    # "000001.SZ" -> "000001"; the research chain carries bare six-digit codes
    frame["symbol"] = frame["con_code"].astype(str).str.split(".").str[0].str.zfill(6)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["symbol"])
    return (frame[["trade_date", "symbol", "weight"]]
            .drop_duplicates(["trade_date", "symbol"], keep="last")
            .sort_values(["trade_date", "symbol"])
            .reset_index(drop=True))


def membership_mask(
    frame: pd.DataFrame,
    pool: str,
    membership_dir: Optional[Path] = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    lag_days: int = 0,
) -> pd.Series:
    """Boolean mask: is each ``(date, symbol)`` row an index member as of that date?

    Parameters
    ----------
    frame : DataFrame with a date column and a bare six-digit symbol column.
    pool : one of :data:`INDEX_CODE_BY_POOL` (``"hs300"``, ``"zz500"``, ``"zz1000"``).
    lag_days : extra calendar days to push the snapshot back, if you want to be
        stricter than "most recent month-end snapshot".
    """

    if pool not in INDEX_CODE_BY_POOL:
        raise KeyError("unknown pool {!r}; expected one of {}".format(pool, sorted(INDEX_CODE_BY_POOL)))
    weights = load_index_weight(INDEX_CODE_BY_POOL[pool], membership_dir)

    dates = pd.to_datetime(frame[date_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    symbols = frame[symbol_col].astype(str).str.zfill(6).to_numpy()

    snapshot_dates = np.sort(weights["trade_date"].unique())
    if lag_days:
        dates = dates - np.timedelta64(int(lag_days), "D")
    # side="right" then -1 -> most recent snapshot at or before the row's date
    position = np.searchsorted(snapshot_dates, dates, side="right") - 1
    in_history = position >= 0
    snapshot_for_row = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    snapshot_for_row[in_history] = snapshot_dates[position[in_history]]

    members = pd.MultiIndex.from_frame(weights[["trade_date", "symbol"]])
    rows = pd.MultiIndex.from_arrays([snapshot_for_row, symbols])
    mask = rows.isin(members) & in_history
    return pd.Series(mask, index=frame.index, name="is_index_member")


def membership_intervals(index_code: str, membership_dir: Optional[Path] = None) -> pd.DataFrame:
    """Reconstruct ``(symbol, start_date, end_date)`` membership spells from snapshots.

    ``end_date`` is the last snapshot on which the symbol was still present, so an
    open spell runs to the final snapshot.  Useful for auditing how much the
    universe actually churns, and for eyeballing known index events.
    """

    weights = load_index_weight(index_code, membership_dir)
    snapshots = np.sort(weights["trade_date"].unique())
    weights = weights.assign(_snap=weights["trade_date"].map({d: i for i, d in enumerate(snapshots)}))
    weights = weights.sort_values(["symbol", "_snap"])
    weights["_gap"] = weights.groupby("symbol")["_snap"].diff().fillna(1) != 1
    weights["_spell"] = weights.groupby("symbol")["_gap"].cumsum()
    grouped = weights.groupby(["symbol", "_spell"])["_snap"].agg(["min", "max"]).reset_index()
    grouped["start_date"] = snapshots[grouped["min"].astype(int)]
    grouped["end_date"] = snapshots[grouped["max"].astype(int)]
    return grouped[["symbol", "start_date", "end_date"]].sort_values(["start_date", "symbol"]).reset_index(drop=True)


def pool_diff_by_date(
    pool: str,
    membership_dir: Optional[Path] = None,
    static_pool_file: Optional[str] = None,
) -> pd.DataFrame:
    """Per-snapshot disagreement between the static pool file and true membership."""

    weights = load_index_weight(INDEX_CODE_BY_POOL[pool], membership_dir)
    path = Path(static_pool_file or STATIC_POOL_FILE[pool])
    static = {line.strip().zfill(6) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    rows = []
    for snapshot, group in weights.groupby("trade_date"):
        actual = set(group["symbol"])
        rows.append({
            "trade_date": snapshot,
            "index_members": len(actual),
            "static_pool": len(static),
            "in_both": len(actual & static),
            "index_only": len(actual - static),
            "static_only": len(static - actual),
            "jaccard": len(actual & static) / len(actual | static) if (actual | static) else np.nan,
        })
    return pd.DataFrame(rows)
