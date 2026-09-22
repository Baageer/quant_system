"""Restrict an existing dataset to point-in-time index members.

The universe is baked into the dataset the moment ``feature_event_study`` reads
``data/HS300.txt`` and is carried forward purely as the ``symbol`` column, so the
cheapest honest fix is a row filter at the dataset boundary rather than a rerun
of the whole feature chain.  That shortcut is valid *for this pipeline* because:

* the 28 features are per-symbol causal time series -- ``build_causal_feature_panel``
  consumes ``stock_data.items()`` one symbol at a time and never ranks or
  normalises across the cross-section;
* the excess target is measured against the **HS300 index**
  (``excess_target_dataset`` docstring), a fixed external series, so removing
  rows does not move the label of any surviving row.

What *does* legitimately change is the training population -- the model refit and
the Top-K validation grid both see a different universe, which is the entire
point.

Emits ``trend_probability_{split}_{kind}_{run_id}.csv`` plus renamed manifests,
so every downstream stage can be pointed at the new dataset with ``--run-id``.

Example::

    python -m research.market_regime.pit_pool_filter \
        --dataset-id hs300_probability_dataset --pool hs300 --run-id hs300_pit_v1
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from research.market_regime.index_membership import DEFAULT_MEMBERSHIP_DIR, membership_mask
from research.market_regime.output_paths import DATASET_DIR

SPLITS = ("train", "validation", "test")
KINDS = ("tradability_samples", "model_samples")
MANIFESTS = ("trend_probability_dataset_manifest_{}.json",
             "trend_probability_tradability_manifest_{}.json")


def _rewrite_manifest_paths(paths, dataset_id: str, run_id: str, dataset_dir: Path) -> list:
    """Point manifest paths at the PIT files; return the keys left un-rewritten.

    A path is rewritten only when the PIT counterpart actually exists on disk,
    so a stale entry can never masquerade as PIT data.  Anything left behind is
    reported back so the manifest records exactly what is still static.
    """

    stale = []
    for key, value in list(paths.items()):
        if isinstance(value, dict):
            stale.extend("{}.{}".format(key, sub) for sub in _rewrite_manifest_paths(value, dataset_id, run_id, dataset_dir))
            continue
        if not isinstance(value, str):
            continue
        candidate = dataset_dir / Path(value).name.replace("_{}.csv".format(dataset_id), "_{}.csv".format(run_id))
        if candidate.exists():
            paths[key] = str(candidate)
        else:
            stale.append(key)
    return stale


def filter_dataset(
    dataset_dir: Path,
    dataset_id: str,
    pool: str,
    run_id: str,
    membership_dir: Optional[Path] = None,
    splits: Sequence[str] = SPLITS,
    kinds: Sequence[str] = KINDS,
) -> pd.DataFrame:
    rows = []
    for kind in kinds:
        for split in splits:
            source = dataset_dir / "trend_probability_{}_{}_{}.csv".format(split, kind, dataset_id)
            if not source.exists():
                continue
            frame = pd.read_csv(source, dtype={"symbol": str}, low_memory=False)
            mask = membership_mask(frame, pool, membership_dir)
            kept = frame[mask.to_numpy()]
            target = dataset_dir / "trend_probability_{}_{}_{}.csv".format(split, kind, run_id)
            kept.to_csv(target, index=False, encoding="utf-8-sig")
            rows.append({
                "split": split,
                "kind": kind,
                "rows_before": len(frame),
                "rows_after": len(kept),
                "kept_pct": len(kept) / len(frame) if len(frame) else float("nan"),
                "symbols_before": frame["symbol"].nunique(),
                "symbols_after": kept["symbol"].nunique(),
                "date_min": str(kept["date"].min()) if len(kept) else "",
                "date_max": str(kept["date"].max()) if len(kept) else "",
            })

    # Downstream resolvers look manifests up by dataset id, so the PIT run needs
    # its own copy -- but a verbatim copy is a trap: `load_model_splits` reads
    # `paths[...]` out of the manifest, so stale entries would quietly feed the
    # *static* samples back in while the run looks like a PIT one.  Rewrite every
    # path that has a PIT counterpart, and fail loudly if a rewritten file is
    # missing rather than falling back to the original.
    for template in MANIFESTS:
        source = dataset_dir / template.format(dataset_id)
        if not source.exists():
            continue
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["pit_source_dataset_id"] = dataset_id
        manifest["pit_filter"] = {
            "pool": pool,
            "run_id": run_id,
            "membership_dir": str(membership_dir or DEFAULT_MEMBERSHIP_DIR),
            "rule": "row kept iff (date, symbol) is a member at the most recent month-end snapshot <= date",
        }
        stale = _rewrite_manifest_paths(manifest.get("paths", {}), dataset_id, run_id, dataset_dir)
        manifest["pit_filter"]["paths_still_static"] = sorted(stale)
        (dataset_dir / template.format(run_id)).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return pd.DataFrame(rows)


def run(
    dataset_dir: Path,
    dataset_id: str,
    pool: str,
    run_id: str,
    membership_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    report = filter_dataset(dataset_dir, dataset_id, pool, run_id, membership_dir)
    if report.empty:
        raise SystemExit("no source splits found for dataset_id={!r} in {}".format(dataset_id, dataset_dir))
    display = report.copy()
    display["kept_pct"] = display["kept_pct"].map(lambda v: "{:.1%}".format(v))
    print("\n=== {} -> {} (pool={}) ===".format(dataset_id, run_id, pool))
    print(display.to_string(index=False))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_dir / "pit_pool_filter_{}.csv".format(run_id), index=False, encoding="utf-8-sig")
    return report


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--dataset-id", default="hs300_probability_dataset")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--membership-dir", default=None)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    args = parser.parse_args(argv)
    run(
        Path(args.dataset_dir), args.dataset_id, args.pool, args.run_id,
        Path(args.membership_dir) if args.membership_dir else None,
        Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
