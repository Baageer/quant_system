"""Append the two pre-gate signals to the frozen samples as two extra columns.

Pre-registration: ``docs/research/nine_experiments/alternative_data_main_ab_2026-09.md``.

Three rules from that document are load-bearing here, and each one exists because
the obvious alternative is silently wrong:

1. **Append, never re-select.**  The frozen feature set was chosen *competitively*
   (78 candidates down to at most 30, greedy in ranked order with a hard budget --
   ``probability_dataset.select_features``).  Feeding the new columns back into
   that selection would let them *displace* existing features, so the treatment arm
   would be "27 old + 2 new" rather than "28 old + 2 new" and the one-variable A/B
   would be dead on arrival.  ZZ1000 sits at exactly 30 columns, so its budget is
   provably spent and displacement is not a possibility to be weighed, only a
   certainty to be avoided.  The caller appends to ``feature_columns`` directly;
   this module never touches the selection.
2. **The shared columns must come through untouched.**  They are the *control*.
   The sample files are read as text (``dtype=str``) and written back as text, which
   makes the round-trip byte-identical -- verified, not assumed: reading them the
   ordinary way parses ``symbol`` as an integer, so ``000001`` comes back as ``1``
   and the file's join key is quietly destroyed.  ``verify_append_only`` re-checks
   this on the written file rather than trusting the construction.
3. **The clock is the gate's clock.**  The join reuses
   ``alternative_data_gate.load_signal_panel`` / ``attach_signal`` verbatim.  A
   second implementation of "what was knowable at t" would be a second answer to a
   question that already has one.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.alternative_data_gate import (
    Clock,
    attach_signal,
    load_signal_panel,
    to_symbols,
)


#: The gate's passing signals, in the order the pre-registration pins them
#: (``moneyflow`` first, ``stk_holdernumber`` second).  The order is part of the
#: registration because it fixes the appended column order.
SIGNALS: Sequence[Clock] = (
    Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
          form="level"),
    Clock(name="stk_holdernumber", date_column="ann_date", value_column="holder_num",
          form="pct_change", asof=True),
)

POOLS: Dict[str, Dict[str, str]] = {
    "hs300": {"source_id": "excess_hs300_pit_v1", "target_id": "excess_hs300_pit_e8_v1"},
    "zz1000": {"source_id": "excess_zz1000_pit_v1",
               "target_id": "excess_zz1000_pit_e8_v1"},
}

SPLITS: Sequence[str] = ("train", "validation", "test")

DEFAULT_DATASET_DIR = "D:/temp/quant_output/datasets"
DEFAULT_SIGNAL_DIR = "data/raw/tushare/alternative"

#: Reported alongside the values but never written as a feature: staleness is how
#: the join is audited, and a column of "how many days old was this" would be a
#: feature nobody registered.
AGE_COLUMN = "signal_age_days"

#: Rows per read/write pass over the sample files.  Sized to keep one text-typed
#: chunk plus its joined copy comfortably inside memory; the largest split has 63
#: columns, and a chunk of these costs a few hundred MB.
CHUNK_ROWS = 100_000


def signal_column(clock: Clock) -> str:
    """The appended column's name.  Must not collide with a frozen feature."""

    return "alt_{}_{}".format(clock.name, clock.form)


def sample_path(dataset_dir: Path, dataset_id: str, split: str) -> Path:
    return Path(dataset_dir) / "trend_probability_{}_tradability_samples_{}.csv".format(
        split, dataset_id)


#: Every text read in this module goes through these.  ``dtype=str`` keeps the
#: six-digit ``symbol`` from being parsed as an integer (``000001`` -> ``1``), and
#: the two NA flags keep pandas from inventing a float where the file has text.
#:
#: ``na_filter=False`` is the load-bearing one.  Left on, an empty field becomes
#: float ``NaN``, and ``NaN != NaN`` -- so a comparison over these frames reports
#: every blank cell in the file as a difference.  This is not a hypothetical: the
#: first run of the append-only check reported 1,475,014 differing cells on one
#: split, every one of them a blank paired with a blank, and zero real differences.
#: Turning NA inference off removes the float entirely, so ``==`` is exact text
#: equality and there is no missing-value special case left to get wrong.  It is
#: also strictly stronger than special-casing NaN: a literal ``NA`` that one side
#: reads as blank and the other as text is still a difference, which a NaN-aware
#: comparison would silently accept.
TEXT_READ_OPTIONS: Dict[str, object] = {
    "dtype": str,
    "low_memory": False,
    "keep_default_na": False,
    "na_filter": False,
}


def read_samples_as_text(path: Path) -> pd.DataFrame:
    """Read a sample file without letting pandas reinterpret anything.

    ``dtype=str`` is not belt-and-braces, it is the difference between preserving
    the file and corrupting it: ``symbol`` is a six-digit zero-padded string that
    pandas otherwise reads as an integer (``000001`` -> ``1``).
    """

    return pd.read_csv(path, **TEXT_READ_OPTIONS)


def render_values(values) -> np.ndarray:
    """Floats to text at full round-trip precision; missing stays an empty cell.

    The sample files leave missing numbers as empty fields, so the appended
    columns match rather than introducing ``nan`` as a third spelling of absent.

    Returns a positional array rather than a ``Series`` on purpose: callers assign
    it onto a frame whose row order has just been restored, and an index-aligned
    ``Series`` would silently reorder it back.
    """

    def one(value: object) -> str:
        if value is None or value is pd.NA:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return "" if math.isnan(number) else repr(number)

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.map(one).to_numpy()


def load_panels(signal_dir: Path,
                signals: Sequence[Clock] = SIGNALS) -> Dict[str, pd.DataFrame]:
    """Every signal's panel, loaded once and reused across splits.

    Separated from the join because the panels are the expensive half and the same
    three panels serve all three splits per pool.  Reloading per split would re-read
    the largest file on disk six times for no gain.
    """

    return {clock.name: load_signal_panel(signal_dir, clock) for clock in signals}


def attach_signals(raw: pd.DataFrame, panels: Mapping[str, pd.DataFrame],
                   signals: Sequence[Clock] = SIGNALS) -> pd.DataFrame:
    """Add one column per signal to a text-read sample frame, row-aligned.

    ``attach_signal`` sorts the rows it is given, so the original row position is
    carried through the join as a column and used to put the values back.  Any
    assignment by position would misalign the file the moment the sort reordered it.
    """

    keys = pd.DataFrame({
        "_row": np.arange(len(raw)),
        "date": pd.to_datetime(raw["date"], errors="coerce"),
        "symbol": to_symbols(raw["symbol"]),
    })
    for clock in signals:
        merged = attach_signal(keys, panels[clock.name], clock).sort_values("_row")
        if len(merged) != len(raw):
            raise ValueError("{}: join changed the row count".format(clock.name))
        raw[signal_column(clock)] = render_values(merged["value"].to_numpy())
    return raw


def read_header(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def iter_chunks(path: Path, chunk_rows: int):
    """Text-typed chunks, so nothing the file holds is reinterpreted on the way in.

    The build and the verification read through this same function on purpose.  If
    one side inferred NA values and the other did not, a source cell reading ``NA``
    would be written as an empty field and then reported as a difference -- a
    failure manufactured by the two readers disagreeing about what the file says.
    """

    return pd.read_csv(path, chunksize=chunk_rows, **TEXT_READ_OPTIONS)


def verify_append_only(source_path: Path, target_path: Path,
                       expected_new_columns: Sequence[str],
                       chunk_rows: int = CHUNK_ROWS) -> Dict[str, object]:
    """Prove the target is the source plus exactly the expected columns.

    Reads both as text and compares the shared columns character-for-character,
    which is the strongest available form of "the control did not move".  A float
    tolerance would be the wrong instrument here: the point is that nothing about
    the existing features changed, and text equality leaves no room to argue.

    "Character-for-character" is only true because of ``TEXT_READ_OPTIONS``; with
    pandas' NA inference on, an empty field arrives as float ``NaN`` and the ``==``
    below turns every blank cell into a reported difference.  See that constant.

    Streamed in lockstep chunks rather than read whole.  The largest split is
    543 MB of text and holding both sides at once costs several gigabytes for no
    extra assurance -- the comparison is row-local, so a chunk boundary cannot
    hide a difference inside it.  ``zip_longest`` rather than ``zip``: a target
    with extra rows is exactly the defect this is looking for, and ``zip`` would
    stop at the shorter side and report agreement.
    """

    source_columns = read_header(source_path)
    target_columns = read_header(target_path)
    new_columns = [column for column in target_columns if column not in set(source_columns)]
    missing = [column for column in source_columns if column not in set(target_columns)]

    report: Dict[str, object] = {
        "source_rows": 0,
        "target_rows": 0,
        "source_columns": len(source_columns),
        "new_columns": new_columns,
        "dropped_columns": missing,
        "unexpected_new_columns": sorted(set(new_columns) - set(expected_new_columns)),
        "identical_shared_columns": [],
        "differing_shared_columns": {},
        "column_order_preserved": target_columns[:len(source_columns)] == source_columns,
    }
    if missing or report["column_order_preserved"] is False:
        # Column i of one file no longer means what column i of the other means, so a
        # positional comparison would be measuring nothing.  Report and stop.
        report["ok"] = False
        return report

    differences = {column: 0 for column in source_columns}
    for source, target in itertools.zip_longest(iter_chunks(source_path, chunk_rows),
                                                iter_chunks(target_path, chunk_rows),
                                                fillvalue=None):
        if source is None or target is None:
            report["source_rows"] = "chunk count differs"
            report["target_rows"] = "chunk count differs"
            report["ok"] = False
            return report
        report["source_rows"] += int(len(source))
        report["target_rows"] += int(len(target))
        if len(source) != len(target):
            report["ok"] = False
            return report
        for column in source_columns:
            same = source[column].to_numpy() == target[column].to_numpy()
            differences[column] += int((~same).sum())

    report["identical_shared_columns"] = [c for c, count in differences.items() if not count]
    report["differing_shared_columns"] = {c: count for c, count in differences.items() if count}
    report["ok"] = (
        not report["differing_shared_columns"]
        and not missing
        and not report["unexpected_new_columns"]
        and sorted(new_columns) == sorted(expected_new_columns)
        and report["column_order_preserved"]
    )
    return report


def build_split(dataset_dir: Path, signal_dir: Path, source_id: str, target_id: str,
                split: str, signals: Sequence[Clock] = SIGNALS,
                panels: Optional[Mapping[str, pd.DataFrame]] = None,
                chunk_rows: int = CHUNK_ROWS) -> Dict[str, object]:
    """One split: read, append, write, then verify what was written.

    Written chunk by chunk.  The sample files run to hundreds of megabytes of text
    and the verification reads both sides, so a whole-frame build would hold the
    source, the target and the signal panels in memory simultaneously -- on a 16 GB
    machine that is the difference between finishing and swapping.  The join is
    row-local, so chunking it changes nothing about the result.
    """

    source_path = sample_path(dataset_dir, source_id, split)
    target_path = sample_path(dataset_dir, target_id, split)
    if not source_path.exists():
        raise FileNotFoundError("source samples missing: {}".format(source_path))

    panels = panels if panels is not None else load_panels(signal_dir, signals)
    expected = [signal_column(clock) for clock in signals]
    if target_path.exists():
        # Opened in append mode below, so a leftover target would be silently extended
        # and then verify against a source it is no longer a copy-plus-columns of.
        target_path.unlink()

    present = {column: 0 for column in expected}
    rows = 0
    wrote_header = False
    for chunk in iter_chunks(source_path, chunk_rows):
        chunk = attach_signals(chunk.reset_index(drop=True), panels, signals)
        for column in expected:
            # An empty cell is this file format's spelling of missing, and
            # ``to_numeric(errors="coerce")`` already lands it on NaN -- no
            # substitution step is needed, or wanted.
            present[column] += int(pd.to_numeric(chunk[column], errors="coerce").notna().sum())
        chunk.to_csv(target_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        rows += len(chunk)
    if not wrote_header:                     # an empty split still needs its header
        attach_signals(pd.DataFrame(columns=read_header(source_path)), panels,
                       signals).to_csv(target_path, index=False)

    report = verify_append_only(source_path, target_path, expected, chunk_rows=chunk_rows)
    report.update({"split": split, "source": str(source_path), "target": str(target_path)})
    for column in expected:
        report["{}_coverage".format(column)] = (present[column] / rows) if rows else float("nan")
    return report


def build_pool(dataset_dir: Path, signal_dir: Path, pool: str, splits: Sequence[str] = SPLITS,
               signals: Sequence[Clock] = SIGNALS, chunk_rows: int = CHUNK_ROWS) -> Dict[str, object]:
    spec = POOLS[pool]
    # Loaded once per pool: the panels do not depend on the split, and the largest
    # of them is by far the most expensive thing in this module.
    panels = load_panels(signal_dir, signals)
    reports = [build_split(dataset_dir, signal_dir, spec["source_id"], spec["target_id"],
                           split, signals, panels=panels, chunk_rows=chunk_rows)
               for split in splits]
    return {
        "pool": pool,
        "source_id": spec["source_id"],
        "target_id": spec["target_id"],
        "ok": all(report["ok"] for report in reports),
        "splits": reports,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--signal-dir", default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--pools", nargs="+", default=sorted(POOLS), choices=sorted(POOLS))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--output", default=None, help="JSON report path")
    args = parser.parse_args(argv)

    dataset_dir, signal_dir = Path(args.dataset_dir), Path(args.signal_dir)
    print("== E8 append-only feature build ==")
    print("  appended columns: {}".format(
        ", ".join(signal_column(clock) for clock in SIGNALS)))

    reports = []
    for pool in args.pools:
        report = build_pool(dataset_dir, signal_dir, pool, tuple(args.splits))
        reports.append(report)
        for split in report["splits"]:
            coverage = " ".join(
                "{}={:.1%}".format(key, value) for key, value in split.items()
                if key.endswith("_coverage"))
            print("  {:<8} {:<11} ok={} rows={} {}".format(
                pool, split["split"], split["ok"], split["source_rows"], coverage))
            if not split["ok"]:
                print("     differing: {}".format(split["differing_shared_columns"]))
                print("     unexpected new columns: {}".format(split["unexpected_new_columns"]))

    output = Path(args.output) if args.output else dataset_dir / "alternative_data_features_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"pools": reports}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("[write] {}".format(output))
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
