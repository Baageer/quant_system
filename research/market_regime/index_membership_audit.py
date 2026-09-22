"""How wrong is the static pool?  Point-in-time membership vs ``data/HS300.txt``.

Answers, per pool:

* how much the index actually churns (entries/exits per year, distinct names ever);
* on an average date in the backtest windows, how many names the static pool
  carries that were **not** index members then (look-ahead / survivorship), and
  how many real members the static pool **misses** (missed opportunity).

Neither direction is a pure bias -- the missed names offset the smuggled ones --
which is exactly why the net effect has to be measured rather than asserted.

Example::

    python -m research.market_regime.index_membership_audit
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.index_membership import (
    INDEX_CODE_BY_POOL,
    STATIC_POOL_FILE,
    load_index_weight,
    membership_intervals,
    pool_diff_by_date,
)
from research.market_regime.output_paths import DATASET_DIR

WINDOWS = {
    "validation": ("2019-07-17", "2021-12-02"),
    "test": ("2022-07-20", "2025-12-02"),
}


def churn_stats(index_code: str, membership_dir: Optional[Path] = None) -> dict:
    intervals = membership_intervals(index_code, membership_dir)
    weights = load_index_weight(index_code, membership_dir)
    snapshots = np.sort(weights["trade_date"].unique())
    years = (pd.Timestamp(snapshots[-1]) - pd.Timestamp(snapshots[0])).days / 365.25
    spell_days = (intervals["end_date"] - intervals["start_date"]).dt.days
    return {
        "snapshots": len(snapshots),
        "first": str(pd.Timestamp(snapshots[0]).date()),
        "last": str(pd.Timestamp(snapshots[-1]).date()),
        "distinct_symbols_ever": int(intervals["symbol"].nunique()),
        "spells": int(len(intervals)),
        "entries_per_year": float(len(intervals) / years) if years else np.nan,
        "median_spell_days": float(spell_days.median()),
    }


def window_overlap(pool: str, membership_dir: Optional[Path] = None) -> pd.DataFrame:
    diff = pool_diff_by_date(pool, membership_dir)
    diff["trade_date"] = pd.to_datetime(diff["trade_date"])
    rows = []
    for name, (start, end) in WINDOWS.items():
        sub = diff[(diff["trade_date"] >= start) & (diff["trade_date"] <= end)]
        if sub.empty:
            continue
        rows.append({
            "pool": pool,
            "window": name,
            "snapshots": len(sub),
            "mean_index_only": float(sub["index_only"].mean()),
            "mean_static_only": float(sub["static_only"].mean()),
            "mean_in_both": float(sub["in_both"].mean()),
            "mean_jaccard": float(sub["jaccard"].mean()),
            "worst_jaccard": float(sub["jaccard"].min()),
        })
    return pd.DataFrame(rows)


def run(
    pools: Sequence[str] = ("hs300", "zz500", "zz1000"),
    membership_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    churn = pd.DataFrame([{"pool": p, "index_code": INDEX_CODE_BY_POOL[p],
                           "static_file": STATIC_POOL_FILE[p],
                           **churn_stats(INDEX_CODE_BY_POOL[p], membership_dir)}
                          for p in pools])

    print("\n=== index churn (from monthly snapshots) ===")
    print(churn.to_string(index=False))

    overlaps = pd.concat([window_overlap(p, membership_dir) for p in pools], ignore_index=True)
    print("\n=== static pool vs true membership, averaged over each backtest window ===")
    print("index_only   = real members the static pool is MISSING (missed opportunity)")
    print("static_only  = static-pool names that were NOT members then (look-ahead)")
    display = overlaps.copy()
    for column in ("mean_jaccard", "worst_jaccard"):
        display[column] = display[column].map(lambda v: "{:.3f}".format(v))
    for column in ("mean_index_only", "mean_static_only", "mean_in_both"):
        display[column] = display[column].map(lambda v: "{:.1f}".format(v))
    print(display.to_string(index=False))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        churn.to_csv(output_dir / "index_membership_churn.csv", index=False, encoding="utf-8-sig")
        overlaps.to_csv(output_dir / "index_membership_window_overlap.csv", index=False, encoding="utf-8-sig")
    return overlaps


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz500,zz1000")
    parser.add_argument("--membership-dir", default=None)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    args = parser.parse_args(argv)
    run(
        tuple(p.strip() for p in args.pools.split(",") if p.strip()),
        Path(args.membership_dir) if args.membership_dir else None,
        Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
