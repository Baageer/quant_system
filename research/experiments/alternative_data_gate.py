"""E8 pre-gate G1 (direction knowability) and G2 (non-redundancy).

Pre-registration: ``docs/research/03_factor_and_valuation/alternative_data_pregate_2026-09.md`` section 4.

The gate asks a narrower question than a feature study would.  Not "does this
signal have IC" -- the valuation study already showed a signal can carry the
strongest IC in the book and still be unusable -- but "**is its cross-sectional
direction knowable in advance and stable**".  A signal whose sign is only visible
after the fact is dropped, however good its pooled IC looks.

Three shapes of this module are deliberate:

* **The clock is applied here, not in the fetcher.**  ``margin_detail`` is used a
  day late because G0 *measured* that it publishes a day late; the stored CSV
  holds what the interface said.  The lag is a judgement about knowability, so it
  lives next to the judgement rather than in the raw file.
* **Test is joined but never judged.**  It is reported so a reader can see whether
  the validation sign held up, but no criterion reads it -- if test helped decide
  which signals proceed, the test numbers would no longer be clean.
* **No champion is picked.**  Every ``(signal, pool)`` is reported on its own; a
  signal that clears every gate is still not "the one".
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.feature_study_core import (
    daily_rank_ics,
    load_study_split,
    rank_corr_between,
    summarize_ic,
)


HORIZON = 10
LABEL = "future_excess_return_{}d".format(HORIZON)
#: House half-threshold, the same one ``surviving_candidates`` uses inside its
#: ``sign_consistent`` test; the pre-registration adopted it for the two halves.
MIN_ABS_IC = 0.01
#: The production pipeline's own de-duplication threshold
#: (``docs/research/02_model_and_target/future_trend_probability_model_plan.md``: ``correlation_threshold=0.85``).
MAX_REDUNDANCY = 0.85
SPLIT_MIN_ROWS = 20

DEFAULT_DATASET_DIR = "D:/temp/quant_output/datasets"
DEFAULT_SIGNAL_DIR = "data/raw/tushare/alternative"
#: The two frozen outcome models.  Their ``feature_columns`` define G2's target
#: set, and they are read from the manifest rather than hard-coded here so the
#: gate cannot drift away from the models that were actually published.
POOL_MODELS = {
    "hs300": {"model_id": "hs300_pit_outcome", "dataset_id": "excess_hs300_pit_v1"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "dataset_id": "excess_zz1000_pit_v1"},
}


@dataclass(frozen=True)
class Clock:
    """How one signal becomes knowable at ``t``, per the G0 verdict."""

    name: str
    date_column: str
    value_column: str
    form: str            # "level" for a flow, "pct_change" for a stock/level
    lag_days: int = 0    # 1 means "use the previous published observation"
    asof: bool = False   # latest announcement at or before t, not an exact join
    judged: bool = True

    @property
    def key(self) -> str:
        """Stable identifier for this signal *form*; level and pct differ."""

        return "{}|{}".format(self.name, self.form)

    @property
    def column(self) -> str:
        return "signal_{}".format(self.key)


#: Pinned in pre-registration section 4.0.2, written before any IC was computed.
CLOCKS: Sequence[Clock] = (
    Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount", form="level"),
    Clock(name="margin_detail", date_column="trade_date", value_column="rzye",
          form="pct_change", lag_days=1),
    Clock(name="top_list", date_column="trade_date", value_column="net_amount", form="level"),
    Clock(name="stk_holdernumber", date_column="ann_date", value_column="holder_num",
          form="pct_change", asof=True),
    # Reported but never judged: these are the size-proxy forms the pinned rule
    # rejected.  Computing them means a reader can see what was passed over, and
    # the judgement stays on the pinned forms whatever these show.
    Clock(name="margin_detail", date_column="trade_date", value_column="rzye",
          form="level", lag_days=1, judged=False),
    Clock(name="stk_holdernumber", date_column="ann_date", value_column="holder_num",
          form="level", asof=True, judged=False),
)


def to_symbols(ts_codes: pd.Series) -> pd.Series:
    """``600519.SH`` -> ``600519``, the six-digit key the sample files use.

    A plain ``zfill``-style normaliser leaves the exchange suffix attached, which
    would join to nothing at all rather than fail loudly.
    """

    return ts_codes.astype(str).str.strip().str.split(".").str[0].str.zfill(6)


#: Rows per read pass.  The panel is built from a 21.9 M row file once the training
#: window is included, and the vendor's row is several times wider than the three
#: columns that survive; reading it whole holds every intermediate column of every
#: row at once to produce a frame a fraction of that size.
CHUNK_ROWS = 500_000


def _reduce_rows(frame: pd.DataFrame, clock: Clock) -> pd.DataFrame:
    """The row-local half: symbol, date and value, and nothing else.

    Everything here maps a row to a row, so applying it chunk by chunk is the same
    computation as applying it to the whole file.
    """

    frame = frame.copy()
    frame["symbol"] = to_symbols(frame["ts_code"])
    frame["signal_date"] = pd.to_datetime(frame[clock.date_column].astype(str).str.strip(),
                                          format="%Y%m%d", errors="coerce")
    frame["value"] = pd.to_numeric(frame[clock.value_column], errors="coerce")
    frame = frame.loc[:, ["symbol", "signal_date", "value"]]
    return frame.dropna(subset=["signal_date"])


def _is_row_local(clock: Clock) -> bool:
    """Whether every step of the transform is a per-row map.

    False for the two shifts below, which need a symbol's whole history in hand.  A
    chunked read of those would compute a growth rate across an arbitrary boundary
    and report it as a fact about the company.
    """

    return clock.form != "pct_change" and not clock.lag_days


def load_signal_panel(signal_dir: Path, clock: Clock,
                      chunk_rows: Optional[int] = CHUNK_ROWS) -> pd.DataFrame:
    """One fetched signal, put on the clock the gate registered for it.

    Only three of the file's columns are read.  The fetch keeps the vendor's whole
    row -- ``moneyflow`` alone carries four amount columns the panel never looks at --
    and with the training window included the file is 21.9 M rows, so reading all of
    it as untyped objects costs several gigabytes to produce a three-column frame.
    Nothing below touches any other column, so this is not a narrowing of the
    contract, only of the read.

    Chunked for the same reason, and only where chunking is provably the same
    computation: a row-local clock reduces row by row, so a chunk boundary cannot
    change an answer.  The one cross-row step left is ``drop_duplicates``, and the
    chunked path does not reproduce it -- it **refuses** if the file has a repeated
    key, because ``keep="last"`` would have chosen a row the chunked path never held
    side by side, and quietly returning the first is exactly the kind of
    plausible-looking wrong answer this module exists to prevent.
    """

    path = Path(signal_dir) / "{}.csv.gz".format(clock.name)
    if not path.exists():
        raise FileNotFoundError("signal not fetched: {}".format(path))
    header = set(pd.read_csv(path, nrows=0).columns)
    usecols = ["ts_code", clock.date_column, clock.value_column]
    unknown = [column for column in usecols if column not in header]
    if unknown:
        # Named individually: the vendor's own error for a bad field list names none
        # of them, and "which column did I get wrong" is the whole question here.
        raise KeyError("{} is missing {}".format(path.name, unknown))

    if chunk_rows and _is_row_local(clock):
        frame = pd.concat([_reduce_rows(chunk, clock) for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=chunk_rows, low_memory=False)],
            ignore_index=True)
        repeated = frame.duplicated(["symbol", "signal_date"])
        if bool(repeated.any()):
            raise ValueError(
                "{} has {} repeated (symbol, date) keys; the chunked load cannot "
                "reproduce keep='last' across a chunk boundary".format(
                    path.name, int(repeated.sum())))
        frame = frame.sort_values(["symbol", "signal_date"])
    else:
        frame = _reduce_rows(pd.read_csv(path, usecols=usecols, low_memory=False), clock)
        frame = frame.sort_values(["symbol", "signal_date"])
        frame = frame.drop_duplicates(["symbol", "signal_date"], keep="last")

    if clock.form == "pct_change":
        previous = frame.groupby("symbol", sort=False)["value"].shift(1)
        # A zero or absent base has no defined growth rate; leaving the resulting
        # infinity in would let one newly-margin-traded name own a day's ranking.
        frame["value"] = np.where(previous > 0, frame["value"] / previous - 1.0, np.nan)
    frame["known_date"] = frame["signal_date"]
    if clock.lag_days:
        # An interface that publishes row d at d+1 is, at t, only known up to
        # t-1.  So the *value* moves forward while the date stays put: shifting
        # the date too would attribute row t back to t-1, which is look-ahead
        # wearing the costume of caution.
        frame["value"] = frame.groupby("symbol", sort=False)["value"].shift(clock.lag_days)

    panel = frame.loc[:, ["symbol", "known_date", "value"]].dropna(subset=["value"])
    return panel.sort_values(["symbol", "known_date"]).reset_index(drop=True)


def attach_signal(samples: pd.DataFrame, panel: pd.DataFrame, clock: Clock) -> pd.DataFrame:
    """Join one signal onto the sample rows without ever reading a future row."""

    # ``merge_asof`` wants the on-key sorted *globally*, and sorting by date first
    # leaves each ``by`` group increasing as well, so one ordering satisfies both.
    left = samples.sort_values(["date", "symbol"])
    if clock.asof:
        # The latest announcement at or before t.  No staleness cutoff: an
        # arbitrary one would be a free parameter, and "the most recent thing
        # disclosed by t" is exactly what a trader standing at t would hold.
        merged = pd.merge_asof(left, panel.sort_values(["known_date", "symbol"]),
                               left_on="date", right_on="known_date",
                               by="symbol", direction="backward")
        merged["signal_age_days"] = (merged["date"] - merged["known_date"]).dt.days
    else:
        merged = left.merge(panel.rename(columns={"known_date": "date"}),
                            on=["symbol", "date"], how="left")
    return merged


def evaluate_signal(panel: pd.DataFrame, feature_columns: Sequence[str], clock: Clock) -> Dict[str, object]:
    """G1 (two validation halves) and G2 (redundancy) for one signal on one pool."""

    judged = panel.dropna(subset=[LABEL])
    labelled = judged.dropna(subset=["value"])
    if labelled.empty:
        return {"signal": clock.name, "form": clock.form, "value_column": clock.value_column,
                "judged": clock.judged, "paired_rows": 0, "validation_rows": 0,
                "validation_label_rows": int(len(judged)), "validation_coverage": 0.0,
                "daily_rows_median": 0, "signal_age_days_median": None,
                "halves": {"first": summarize_ic(np.asarray([])),
                           "second": summarize_ic(np.asarray([]))},
                "validation": summarize_ic(np.asarray([])), "test": summarize_ic(np.asarray([])),
                "redundancy": {}, "criteria": {"1a": False, "1b": False}}

    # Test is computed only to be reported; no criterion below reads it.
    validation = labelled[labelled["source_split"] == "validation"]
    test = labelled[labelled["source_split"] == "test"]

    halves = {}
    if len(validation):
        median_date = validation["date"].median()
        for name, subset in (("first", validation[validation["date"] <= median_date]),
                             ("second", validation[validation["date"] > median_date])):
            halves[name] = summarize_ic(daily_rank_ics(subset["date"], subset[LABEL].to_numpy(),
                                                       subset["value"].to_numpy()))
    else:
        halves = {"first": summarize_ic(np.asarray([])), "second": summarize_ic(np.asarray([]))}

    def _summary(subset):
        if not len(subset):
            return summarize_ic(np.asarray([]))
        return summarize_ic(daily_rank_ics(subset["date"], subset[LABEL].to_numpy(),
                                           subset["value"].to_numpy()))

    first_ic = halves["first"]["mean_ic"]
    second_ic = halves["second"]["mean_ic"]
    same_sign = bool(np.isfinite(first_ic) and np.isfinite(second_ic)
                     and np.sign(first_ic) == np.sign(second_ic))
    both_large = bool(np.isfinite(first_ic) and np.isfinite(second_ic)
                      and abs(first_ic) >= MIN_ABS_IC and abs(second_ic) >= MIN_ABS_IC)

    # Redundancy is only worth its cost for the forms under judgement.
    redundancy = {}
    if clock.judged:
        for feature in feature_columns:
            if feature not in validation.columns:
                continue
            correlation = rank_corr_between(
                validation["date"], validation["value"].to_numpy(dtype=float),
                pd.to_numeric(validation[feature], errors="coerce").to_numpy(dtype=float),
                min_rows=SPLIT_MIN_ROWS)
            if np.isfinite(correlation):
                redundancy[feature] = float(correlation)

    # A signal covering 3% of names and one covering 95% are not the same claim,
    # so the join rate is reported next to every IC rather than left implicit.
    validation_labelled = judged[judged["source_split"] == "validation"]
    coverage = float(len(validation) / len(validation_labelled)) if len(validation_labelled) else 0.0

    age = panel["signal_age_days"].median() if "signal_age_days" in panel.columns else None
    return {
        "signal": clock.name,
        "form": clock.form,
        "value_column": clock.value_column,
        "judged": clock.judged,
        "paired_rows": int(len(labelled)),
        "validation_rows": int(len(validation)),
        "validation_label_rows": int(len(validation_labelled)),
        "validation_coverage": coverage,
        "daily_rows_median": float(validation.groupby("date").size().median()) if len(validation) else 0,
        "signal_age_days_median": None if age is None or not np.isfinite(age) else float(age),
        "halves": halves,
        "validation": _summary(validation),
        "test": _summary(test),
        "redundancy": redundancy,
        "criteria": {"1a": same_sign, "1b": both_large},
    }


def apply_g2(rows: Sequence[Dict]) -> None:
    """Attach the G2 verdict: max |rank corr| over the pool's own frozen features."""

    for row in rows:
        redundancy = row.get("redundancy") or {}
        if not redundancy:
            row["max_abs_rank_corr"] = None
            row["max_abs_rank_corr_feature"] = None
            row["criteria"]["2"] = False
            continue
        feature = max(redundancy, key=lambda name: abs(redundancy[name]))
        row["max_abs_rank_corr"] = abs(redundancy[feature])
        row["max_abs_rank_corr_feature"] = feature
        row["criteria"]["2"] = bool(abs(redundancy[feature]) < MAX_REDUNDANCY)


def apply_cross_pool(rows: Sequence[Dict]) -> None:
    """Criterion 1c needs a signal's sign to agree *across* pools, not just in time."""

    by_signal: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        if row["judged"]:
            by_signal.setdefault(row["signal"], {})[row["pool"]] = row

    for pool_rows in by_signal.values():
        passing = [row for row in pool_rows.values() if row["criteria"]["1a"] and row["criteria"]["1b"]]
        signs = {np.sign(row["validation"]["mean_ic"]) for row in passing}
        agree = len(pool_rows) == len(POOL_MODELS) and len(passing) == len(pool_rows) and len(signs) == 1
        for row in pool_rows.values():
            row["criteria"]["1c"] = bool(agree)

    for row in rows:
        row["criteria"].setdefault("1c", False)
        row["criteria"]["passes"] = bool(row["judged"] and row["criteria"]["1a"]
                                         and row["criteria"]["1b"] and row["criteria"]["1c"]
                                         and row["criteria"].get("2"))


def run_gate(signal_dir: Path, dataset_dir: Path, model_dir: Path,
             pool_models: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """Run G1 and G2 for every (signal, pool); test is joined but never judged."""

    pool_models = POOL_MODELS if pool_models is None else pool_models
    panels = {clock.key: load_signal_panel(signal_dir, clock) for clock in CLOCKS}

    rows: List[Dict] = []
    pool_features = {}
    for pool, config in pool_models.items():
        manifest_path = Path(model_dir) / "trade_outcome_manifest_{}.json".format(config["model_id"])
        feature_columns = list(json.loads(manifest_path.read_text(encoding="utf-8"))["feature_columns"])
        pool_features[pool] = feature_columns

        samples = load_study_split(Path(dataset_dir), config["dataset_id"], "validation", HORIZON)
        test_samples = load_study_split(Path(dataset_dir), config["dataset_id"], "test", HORIZON)
        samples["source_split"] = "validation"
        test_samples["source_split"] = "test"
        # Only the join keys, the label and the redundancy targets are needed; the
        # sample files carry every feature column across three splits' worth of rows.
        keep = ["date", "symbol", "source_split", LABEL, *feature_columns]
        combined = pd.concat([samples[keep], test_samples[keep]], ignore_index=True)

        for clock in CLOCKS:
            merged = attach_signal(combined, panels[clock.key], clock)
            row = evaluate_signal(merged, feature_columns, clock)
            row["pool"] = pool
            rows.append(row)
            if verbose:
                print("  {:<18} {:<11} {:<7} {:<9} val IC={:+.4f}  halves={:+.4f}/{:+.4f}".format(
                    row["signal"], row["form"], pool,
                    "判" if row["judged"] else "(照报)",
                    row["validation"]["mean_ic"], row["halves"]["first"]["mean_ic"],
                    row["halves"]["second"]["mean_ic"]))

    apply_g2(rows)
    apply_cross_pool(rows)
    return {"horizon": HORIZON, "label": LABEL, "min_abs_ic": MIN_ABS_IC,
            "max_redundancy": MAX_REDUNDANCY, "split_min_rows": SPLIT_MIN_ROWS,
            "pool_features": pool_features,
            "pool_feature_counts": {pool: len(cols) for pool, cols in pool_features.items()},
            "rows": rows}


def verdict_table(report: Dict) -> pd.DataFrame:
    """The gate's rows as a flat table, report-only forms flagged not dropped."""

    records = []
    for row in report["rows"]:
        records.append({
            "pool": row["pool"], "signal": row["signal"], "form": row["form"],
            "judged": row["judged"],
            "paired_rows": row["paired_rows"], "daily_rows_median": row["daily_rows_median"],
            "validation_coverage": round(row["validation_coverage"], 4),
            "signal_age_days_median": row["signal_age_days_median"],
            "validation_ic": row["validation"]["mean_ic"],
            "first_half_ic": row["halves"]["first"]["mean_ic"],
            "second_half_ic": row["halves"]["second"]["mean_ic"],
            "validation_days": row["validation"]["n_obs"],
            "test_ic": row["test"]["mean_ic"],
            "max_abs_rank_corr": row["max_abs_rank_corr"],
            "max_abs_rank_corr_feature": row["max_abs_rank_corr_feature"],
            "pass_1a": row["criteria"]["1a"], "pass_1b": row["criteria"]["1b"],
            "pass_1c": row["criteria"]["1c"], "pass_2": row["criteria"]["2"],
            "passes_gate": row["criteria"]["passes"],
        })
    return pd.DataFrame(records)


def write_report(report: Dict, output: Path) -> Sequence[Path]:
    """Persist the gate's JSON verdicts and its flat table side by side."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    table_path = output.with_name(output.stem + "_table.csv")
    verdict_table(report).to_csv(table_path, index=False, encoding="utf-8-sig")
    return [output, table_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal-dir", default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    print("== E8 pre-gate G1 (direction knowability) + G2 (non-redundancy) ==")
    report = run_gate(Path(args.signal_dir), Path(args.dataset_dir), Path(args.model_dir))
    print("-- judged verdicts --")
    for row in report["rows"]:
        if not row["judged"]:
            continue
        print("  {:<18} {:<7} 1a= {:<5} 1b= {:<5} 1c= {:<5} 2= {:<5} => {}".format(
            row["signal"], row["pool"], row["criteria"]["1a"], row["criteria"]["1b"],
            row["criteria"]["1c"], row["criteria"]["2"],
            "过门" if row["criteria"]["passes"] else "不过"))

    if args.output:
        for path in write_report(report, Path(args.output)):
            print("[write] {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
