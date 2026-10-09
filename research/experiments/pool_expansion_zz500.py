"""E7: replicate the published pipeline on the 中证500 (zz500) pool.

Why this is not an A/B
----------------------
Every other experiment in this round pits two constructions of the *same* pool
against each other and reports a paired difference.  E7 does not.  It applies
the already-frozen published pipeline to a third pool and asks whether that
pool's own cross-section carries an edge.  There are no arms, no candidate
construction and nothing to re-tune, so the reading is taken off the pool's own
random-portfolio null rather than off a difference against another arm.

That shapes the criteria: they are about the *pool's soundness*, not about
beating some number.  Criterion 1 failing is a legitimate, informative outcome
-- it would say the published edge is not universal -- and the pre-registration
writes that down before any result exists.

Each criterion below is a pure function of artifacts already on disk, so
``--report-only`` recomputes them without touching the pipeline.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from research.market_regime.output_paths import DATASET_DIR

#: Null runs are named ``random_portfolio_{model_id}_{tag}_{draws,summary}.csv``.
#: The tag has to be passed in because each pool's published null uses its own.
DEFAULT_TAGS = {"test": "zz500_pit", "validation": "zz500_pit_val"}

#: Criterion 1's bar, from the pre-registration.
PERCENTILE_THRESHOLD = 0.95

#: The three sample splits criterion 2 requires to be present.  "Present" means
#: the PIT-filtered dataset has rows for them, not that each carries a null.
SAMPLE_SPLITS = ("train", "validation", "test")

#: The windows that get a random-portfolio null of their own.  A null is an
#: engine backtest of a randomly drawn basket, so only the out-of-sample
#: windows can have one -- "train" is a category error here, and the
#: pre-registration's "三段齐全，且每段都有自己的零假设" resolves to: all three
#: splits present, and *both* evaluation windows nulled.  HS300 still owes a
#: PIT validation null, which is the debt a new pool must not inherit.
NULL_SPLITS = ("validation", "test")


@dataclass
class Pool:
    """One pool's identity across the artifacts the criteria read."""

    name: str
    model_id: str
    dataset_id: str
    benchmark: str
    tags: Mapping[str, str] = field(default_factory=dict)


POOLS: Dict[str, Pool] = {
    "hs300": Pool(
        name="hs300",
        model_id="hs300_pit_outcome",
        dataset_id="hs300_pit_v1",
        benchmark="data/benchmarks/000300.csv",
        # HS300 has no validation null on disk: the pre-registration names that
        # gap as the debt a new pool must not inherit, so the tag is absent
        # rather than pointed at some other window's draws.
        tags={"test": "pit"},
    ),
    "zz1000": Pool(
        name="zz1000",
        model_id="zz1000_pit_outcome",
        dataset_id="zz1000_pit_v1",
        benchmark="data/benchmarks/000852.csv",
        tags={"test": "zz1000_pit", "validation": "zz1000_pit_val"},
    ),
    "zz500": Pool(
        name="zz500",
        model_id="zz500_pit_outcome",
        dataset_id="zz500_pit_v1",
        benchmark="data/benchmarks/000905.csv",
        tags=DEFAULT_TAGS,
    ),
}

#: The chain E7 replays, in order.  Recorded here so the archive carries the
#: exact commands rather than a prose description of them.
PIPELINE_STEPS: Sequence[Sequence[str]] = (
    ["-m", "research.experiments.run_zz1000_batches",
     "--stock-file", "data/ZZ_500.txt", "--run-prefix", "zz500_batch"],
    ["-m", "research.market_regime.probability_dataset",
     "--study-run-id", "zz500_batch_01", "zz500_batch_02", "zz500_batch_03",
     "zz500_batch_04", "zz500_batch_05",
     "--run-id", "zz500_probability_dataset", "--horizon", "10",
     "--min-event-confidence", "0.3", "--train-start", "2007-01-31"],
    ["-m", "research.market_regime.logistic_model",
     "--dataset-id", "zz500_probability_dataset", "--run-id", "zz500_logistic_v1"],
    ["-m", "research.market_regime.pit_pool_filter",
     "--dataset-id", "zz500_probability_dataset", "--pool", "zz500",
     "--run-id", "zz500_pit_v1"],
    ["-m", "research.market_regime.excess_target_dataset",
     "--dataset-id", "zz500_pit_v1", "--run-id", "excess_zz500_pit_v1",
     "--benchmark-file", "data/benchmarks/000905.csv"],
    # ``--top-k`` here only settles what the model's own manifest records.  The
    # width the null is drawn at comes from the *ranking* manifest, and the two
    # have disagreed before (zz1000's model manifest says 0.05 while its ranking
    # manifest says 0.10, and the null follows the ranking one); passing the
    # registered incumbent at both sites keeps the pair from diverging again.
    ["-m", "research.market_regime.trade_outcome_model",
     "--dataset-id", "excess_zz500_pit_v1", "--trend-model-id", "zz500_logistic_v1",
     "--run-id", "zz500_pit_outcome", "--horizon", "10", "--top-k", "0.10"],
    ["-m", "research.market_regime.trade_outcome_ranking_evaluation",
     "--model-id", "zz500_pit_outcome", "--baseline-top-k", "0.10",
     "--benchmark-file", "data/benchmarks/000905.csv"],
)


def null_summary_path(dataset_dir: Path, pool: Pool, split: str) -> Path:
    """Where ``random_portfolio_baseline`` writes a pool's null for one split."""

    tag = pool.tags.get(split)
    if tag is None:
        raise ValueError("pool {!r} has no null tag for split {!r}".format(pool.name, split))
    return Path(dataset_dir) / "random_portfolio_{}_{}_summary.json".format(pool.model_id, tag)


def read_null_summary(path: Path) -> Dict:
    """Load one null summary, failing loudly rather than returning an empty dict."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no null summary at {}".format(path))
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "random" not in summary or "total_return" not in summary["random"]:
        raise ValueError("{} has no random.total_return block".format(path))
    return summary


def criterion_1_null_percentile(summary: Mapping, threshold: float = PERCENTILE_THRESHOLD) -> Dict:
    """Where the pool's own return sits in its own random null.

    The percentile is read straight off the summary -- ``random_portfolio_baseline``
    already computed it -- rather than recomputed from the draws, so the number
    here is the published one and not a second implementation of it.

    Reported alongside it: the null's width (``seeds``, ``null_std``) and its
    ceiling (``null_max``), which together are what make the criterion's
    discriminating power visible.  E5b found "at or above the 95th percentile"
    can be trivially true for a strong arm, so a bare PASS is not enough -- the
    margin over the best draw is what says whether the pass is decisive.
    """

    block = summary["random"]["total_return"]
    percentile = float(block["percentile_of_actual"])
    return {
        "percentile_of_actual": percentile,
        "empirical_p_value": float(block.get("empirical_p_value", float("nan"))),
        "seeds": int(summary.get("seeds", 0)),
        "actual_total_return": float(summary["actual"]["total_return"]),
        "null_mean": float(block["mean"]),
        "null_std": float(block["std"]),
        # ``median`` and ``max`` are carried for the renderer and for the report:
        # the percentile alone does not say whether the actual arm cleared every
        # draw or merely most of them.
        "null_median": float(block["median"]),
        "null_max": float(block["max"]),
        "threshold": float(threshold),
        "passed": bool(percentile >= threshold),
    }


def pit_filter_path(dataset_dir: Path, pool: Pool) -> Path:
    """``pit_pool_filter``'s per-split audit table for one pool."""

    return Path(dataset_dir) / "pit_pool_filter_{}.csv".format(pool.dataset_id)


def read_pit_split_rows(path: Path) -> Dict[str, int]:
    """``{split: rows_after}`` from a PIT filter audit, de-duplicated by split."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no PIT filter audit at {}".format(path))
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = {"split", "rows_after"} - set(frame.columns)
    if missing:
        raise ValueError("{} is missing {}".format(path, sorted(missing)))
    # A pool's audit can carry several ``kind`` rows per split; summing would
    # double-count, so the split is present iff any one of its rows kept data.
    grouped = frame.groupby("split")["rows_after"].max()
    return {str(split): int(rows) for split, rows in grouped.items()}


def criterion_2_split_completeness(
    dataset_dir: Path,
    pool: Pool,
    sample_splits: Sequence[str] = SAMPLE_SPLITS,
    null_splits: Sequence[str] = NULL_SPLITS,
) -> Dict:
    """All three splits present, and *both* evaluation windows carrying a null.

    The second half is the point.  A pool can have three splits of samples and
    still leave one window un-nulled, which is exactly the debt HS300 carries on
    its PIT validation window; a null borrowed from another window does not
    discharge it, because the null depends on ``(universe, top_k, seed)``.
    """

    dataset_dir = Path(dataset_dir)
    splits = {}
    try:
        rows_by_split = read_pit_split_rows(pit_filter_path(dataset_dir, pool))
    except (FileNotFoundError, ValueError) as exc:
        rows_by_split = {}
        audit_error = str(exc)
    else:
        audit_error = None
    for split in sample_splits:
        splits[split] = {"rows_after": rows_by_split.get(split, 0),
                         "present": rows_by_split.get(split, 0) > 0}

    nulls = {}
    for split in null_splits:
        try:
            summary = read_null_summary(null_summary_path(dataset_dir, pool, split))
        except (FileNotFoundError, ValueError) as exc:
            nulls[split] = {"null": False, "detail": str(exc)}
            continue
        nulls[split] = {
            "null": True,
            "model_id": summary.get("model_id"),
            "top_k": summary.get("top_k"),
            "seeds": summary.get("seeds"),
            # Newer summaries say ``split_rows``; the published ones say
            # ``test_rows``.  Either is the row count the null was drawn
            # against, carried forward so the caliber check needs no second read.
            "rows": summary.get("split_rows", summary.get("test_rows")),
            "matches_pool": summary.get("model_id") == pool.model_id,
        }

    missing_splits = [split for split in sample_splits if not splits[split]["present"]]
    missing_nulls = [split for split in null_splits if not nulls[split]["null"]]
    mislabelled = [split for split in null_splits
                   if nulls[split]["null"] and not nulls[split]["matches_pool"]]
    return {
        "splits": splits,
        "nulls": nulls,
        "audit_error": audit_error,
        "missing_splits": missing_splits,
        "missing_nulls": missing_nulls,
        "mislabelled_nulls": mislabelled,
        "passed": not (missing_splits or missing_nulls or mislabelled),
    }


def criterion_3_separate_report(rows: Sequence[Mapping]) -> pd.DataFrame:
    """One row per pool, and never a pooled or averaged row.

    Three pools are not three independent trials of one hypothesis: the same
    names migrate between them (measured previously -- 13.4% of the zz500 test
    cross-section was once in HS300 and 41.3% once in ZZ1000), so any row that
    combines or averages them would be a significance claim the design cannot
    support.
    """

    frame = pd.DataFrame(list(rows))
    if "pool" not in frame.columns:
        raise ValueError("comparison rows need a 'pool' column")
    if frame["pool"].duplicated().any():
        raise ValueError("comparison rows must be one per pool")
    # Absent keys align to NaN across rows; in an all-text column that renders
    # as a literal "NaN" and reads like a value rather than a blank.
    if "error" in frame.columns:
        frame["error"] = frame["error"].fillna("")
    return frame.reset_index(drop=True)


def collect_pool_row(dataset_dir: Path, pool: Pool, split: str = "test") -> Dict:
    """Flatten one pool's headline numbers into a single comparison row."""

    summary = read_null_summary(null_summary_path(dataset_dir, pool, split))
    block = summary["random"]["total_return"]
    return {
        "pool": pool.name,
        "model_id": pool.model_id,
        "split": split,
        "top_k": summary.get("top_k"),
        "max_positions": summary.get("max_positions"),
        "seeds": summary.get("seeds"),
        "actual_total_return": float(summary["actual"]["total_return"]),
        "null_median": float(block["median"]),
        "null_mean": float(block["mean"]),
        "percentile_of_actual": float(block["percentile_of_actual"]),
        "empirical_p_value": float(block.get("empirical_p_value", float("nan"))),
        "actual_sharpe_zero_rate": float(summary["actual"].get("sharpe_zero_rate", float("nan"))),
        "actual_max_drawdown": float(summary["actual"].get("max_drawdown", float("nan"))),
    }


def print_pipeline() -> None:
    """The exact commands E7 replays, in order."""

    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print("[{}] PYTHONPATH=. python {}".format(index, " ".join(step)))


def run_pipeline(execute: bool = False) -> int:
    """Run the chain (or just show it).  ``execute`` is opt-in on purpose."""

    if not execute:
        print_pipeline()
        return 0
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print("[{}/{}] {}".format(index, len(PIPELINE_STEPS), " ".join(step)), flush=True)
        completed = subprocess.run([sys.executable] + list(step))
        if completed.returncode != 0:
            print("[stop] step {} exited {}".format(index, completed.returncode), flush=True)
            return completed.returncode
    return 0


def build_report(dataset_dir: Path, compare_split: str = "test") -> Dict:
    """All three criteria plus the side numbers that are reported but not judged."""

    dataset_dir = Path(dataset_dir)
    target = POOLS["zz500"]
    report: Dict = {"dataset_dir": str(dataset_dir), "pool": target.name}

    split_results = {}
    for split in NULL_SPLITS:
        path = null_summary_path(dataset_dir, target, split)
        if not path.exists():
            split_results[split] = {"missing": True, "path": str(path)}
            continue
        summary = read_null_summary(path)
        split_results[split] = criterion_1_null_percentile(summary)
        split_results[split]["path"] = str(path)
    report["splits"] = split_results
    # Criterion 1 is the pre-registered test-window percentile; validation is
    # carried above because criterion 2 requires its null to exist, not because
    # a validation pass would count toward criterion 1.
    report["criterion_1"] = split_results.get(compare_split, {"missing": True})
    report["criterion_2"] = criterion_2_split_completeness(dataset_dir, target)

    comparisons = []
    for pool in POOLS.values():
        try:
            comparisons.append(collect_pool_row(dataset_dir, pool, compare_split))
        except (FileNotFoundError, ValueError) as exc:
            comparisons.append({"pool": pool.name, "model_id": pool.model_id,
                                "split": compare_split, "error": str(exc)})
    report["comparison"] = criterion_3_separate_report(comparisons)
    return report


def print_report(report: Mapping) -> None:
    """Human-readable rendering; the JSON next to it stays authoritative."""

    print("== E7 zz500 pool expansion ==")
    for split in NULL_SPLITS:
        row = report["splits"].get(split, {})
        if row.get("missing"):
            print("  {:<11} MISSING NULL  {}".format(split, row.get("path", "")))
            continue
        print("  {:<11} actual {:+7.2f}%  null median {:+7.2f}%  percentile {:.1f}%  "
              "p={:.4f}  seeds={}".format(
                  split, 100.0 * row["actual_total_return"], 100.0 * row["null_median"],
                  100.0 * row["percentile_of_actual"], row["empirical_p_value"], row["seeds"]))
    c1 = report["criterion_1"]
    if not c1.get("missing"):
        print("criterion 1 (percentile >= {:.0f}%): {}".format(
            100.0 * c1["threshold"], "PASS" if c1["passed"] else "FAIL"))
        # What bounds this criterion's discriminating power is the *headroom* over
        # the null, not the percentile restated.  E5b's lesson was that "at or above
        # the 95th percentile" can be trivially true for a strong arm, so print the
        # best draw and the margin over it: a pass that clears every draw by 20pp
        # and one that squeaks past by 0.1pp are not the same result.
        print("  -- reported, not judged: best of {} draws {:+7.2f}%, actual clears it "
              "by {:.2f}pp".format(
                  c1["seeds"], 100.0 * c1["null_max"],
                  100.0 * (c1["actual_total_return"] - c1["null_max"])))
    c2 = report["criterion_2"]
    print("criterion 2 (three splits present, both evaluation windows nulled): {}".format(
        "PASS" if c2["passed"] else "FAIL"))
    for split, row in c2["splits"].items():
        print("  {:<11} rows_after={}".format(split, row["rows_after"]))
    for split, row in c2["nulls"].items():
        print("  null {:<6} {}".format(split, "present" if row["null"] else "MISSING"))
    if not c2["passed"]:
        print("  missing splits: {}".format(", ".join(c2["missing_splits"]) or "none"))
        print("  missing nulls:  {}".format(", ".join(c2["missing_nulls"]) or "none"))
        print("  mislabelled:    {}".format(", ".join(c2["mislabelled_nulls"]) or "none"))
        if c2.get("audit_error"):
            print("  audit:          {}".format(c2["audit_error"]))
    print("criterion 3 (reported per pool, never merged):")
    print(report["comparison"].to_string(index=False))


def write_report(report: Mapping, output: Path) -> Sequence[Path]:
    """Persist the report as JSON plus the criterion-3 table as CSV.

    Two files because they answer different questions: the JSON is the record of
    what was computed (including which fields were missing), the CSV is the
    cross-pool table a reader will actually put side by side.  The CSV is written
    from the already-built frame rather than re-read from disk, so it cannot
    disagree with the JSON it sits next to.
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    table_path = output.with_name(output.stem + "_table.csv")
    report["comparison"].to_csv(table_path, index=False, encoding="utf-8-sig")
    return [output, table_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--split", default="test", choices=list(NULL_SPLITS))
    parser.add_argument("--report-only", action="store_true",
                        help="Recompute the criteria from disk artifacts; run nothing")
    parser.add_argument("--execute", action="store_true",
                        help="Actually replay the pipeline before reporting")
    parser.add_argument("--print-pipeline", action="store_true",
                        help="Print the replay commands and exit")
    parser.add_argument("--output", default=None, help="Where to write the criteria JSON")
    args = parser.parse_args(argv)

    if args.print_pipeline:
        print_pipeline()
        return 0
    if args.execute:
        code = run_pipeline(execute=True)
        if code != 0:
            return code
    elif not args.report_only:
        # Default is the cheap, non-destructive path: nothing here is worth
        # starting a multi-hour pipeline by accident.
        return run_pipeline(execute=False)

    report = build_report(Path(args.dataset_dir), args.split)
    print_report(report)
    if args.output:
        for path in write_report(report, Path(args.output)):
            print("[write] {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
