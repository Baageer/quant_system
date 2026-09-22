"""Merge 2026 PIT membership snapshots onto the historical ones.

The trap this exists to avoid
-----------------------------
``index_membership.membership_mask`` resolves each row's universe with
``searchsorted(snapshot_dates, date, side="right") - 1`` -- the most recent
snapshot **at or before** the row's date (``index_membership.py:109-113``).
That is the correct rule, and it has a nasty corollary: if the snapshot file
simply *stops* at 2025-12-31, every 2026 row silently resolves to the 2025-12-31
list and nothing raises.  The forward window would then be evaluated on a stale
universe whose error grows all year, with no symptom.

So the 2026 snapshots fetched by ``fetch_tushare_forward_2026`` are merged into
a directory that also carries the full history, and that directory is passed to
``membership_mask`` explicitly.  The historical
``data/raw/tushare/index_weight/`` is never written.

The merge is a pure concat + dedupe on ``(index_code, con_code, trade_date)``;
the cache and the 2026 fetch have disjoint date ranges, so no row is actually
in conflict, and ``keep="last"`` makes the fresh 2026 fetch win if one ever is.

Verification (printed, and in the manifest): the membership in force at each
2026 month-end must **differ** from the 2025-12-31 set.  If every month matched,
the snapshots would not be doing anything and the stale-fallback trap would be
back through a different door.

Example::

    python -m research.experiments.merge_forward_2026_membership
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from research.market_regime.index_membership import INDEX_CODE_BY_POOL, load_index_weight

CACHE_DIR = Path("data/raw/tushare/index_weight")
FORWARD_DIR = Path("data/raw/tushare/forward_2026")
OUTPUT_DIR = FORWARD_DIR / "index_weight_merged"
DEFAULT_INDEXES = "000300.SH,000852.SH,000905.SH"


def merge_one(index_code: str, cache_dir: Path, forward_dir: Path, output_dir: Path) -> Dict[str, object]:
    """Concat history + 2026 for one index and report snapshot coverage."""

    history_path = cache_dir / "{}.csv".format(index_code)
    forward_path = forward_dir / "index_weight" / "{}.csv".format(index_code)
    if not history_path.exists():
        raise FileNotFoundError("missing historical membership: {}".format(history_path))

    frames = [pd.read_csv(history_path, dtype=str, low_memory=False)]
    history_snapshots = int(frames[0]["trade_date"].nunique()) if not frames[0].empty else 0
    forward_snapshots = 0
    if forward_path.exists():
        forward = pd.read_csv(forward_path, dtype=str, low_memory=False)
        forward_snapshots = int(forward["trade_date"].nunique()) if not forward.empty else 0
        frames.append(forward)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["index_code", "con_code", "trade_date"], keep="last")
    combined = combined.sort_values(["trade_date", "con_code"])

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "{}.csv".format(index_code)
    combined.to_csv(target, index=False, encoding="utf-8-sig")

    weights = load_index_weight(index_code, output_dir)
    snapshots = sorted(weights["trade_date"].unique())
    return {
        "index_code": index_code,
        "path": str(target),
        "history_snapshots": history_snapshots,
        "forward_snapshots": forward_snapshots,
        "merged_snapshots": len(snapshots),
        "first": str(pd.Timestamp(snapshots[0]).date()),
        "last": str(pd.Timestamp(snapshots[-1]).date()),
        "rows": int(len(combined)),
    }


def stale_fallback_check(index_code: str, output_dir: Path) -> List[dict]:
    """Per-snapshot membership disagreement against the 2025-12-31 list.

    This is the assertion that the 2026 snapshots actually take effect: with the
    merged directory in hand, every 2026 month-end must resolve to its own list,
    not to the year-end-2025 one.
    """

    weights = load_index_weight(index_code, output_dir)
    members = {snapshot: set(group["symbol"]) for snapshot, group in weights.groupby("trade_date")}
    baseline_date = max(d for d in members if d < pd.Timestamp("2026-01-01"))
    baseline = members[baseline_date]

    rows = []
    for snapshot in sorted(d for d in members if d >= pd.Timestamp("2026-01-01")):
        current = members[snapshot]
        rows.append({
            "index_code": index_code,
            "snapshot": str(snapshot.date()),
            "members": len(current),
            "vs_baseline_identical": current == baseline,
            "added": len(current - baseline),
            "removed": len(baseline - current),
        })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--forward-dir", default=str(FORWARD_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--indexes", default=DEFAULT_INDEXES)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache_dir, forward_dir = Path(args.cache_dir), Path(args.forward_dir)
    output_dir = Path(args.output_dir) if args.output_dir else forward_dir / "index_weight_merged"
    indexes = [c.strip() for c in args.indexes.split(",") if c.strip()]

    summary, checks, missing = [], [], []
    for index_code in indexes:
        try:
            summary.append(merge_one(index_code, cache_dir, forward_dir, output_dir))
        except FileNotFoundError as exc:
            missing.append({"index_code": index_code, "reason": str(exc)})
            continue
        checks += stale_fallback_check(index_code, output_dir)

    print("\n=== merged membership directories ===")
    print(pd.DataFrame(summary).to_string(index=False))

    check_frame = pd.DataFrame(checks)
    if not check_frame.empty:
        print("\n=== 2026 snapshots vs the 2025-12-31 list ===")
        print(check_frame.to_string(index=False))
        identical = check_frame.loc[check_frame["vs_baseline_identical"], "snapshot"].tolist()
        print("\nsnapshots still identical to 2025-12-31: {}".format(identical or "none"))
        # A snapshot that resolves to 2025-12-31 for *every* 2026 date is the trap.
        if len(identical) == len(check_frame):
            print("FAIL: no 2026 snapshot changes the universe -- membership is stale", flush=True)
            return 2
    if missing:
        print("\nskipped (no historical membership cached): {}".format(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    (forward_dir / "membership_merge_manifest.json").write_text(json.dumps({
        "output_dir": str(output_dir),
        "indexes": summary,
        "stale_fallback_checks": checks,
        "skipped": missing,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmanifest -> {}".format(forward_dir / "membership_merge_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
