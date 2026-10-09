"""Fetch the E8 pre-gate's candidate signals, one trade date at a time.

Pre-registration: ``docs/research/03_factor_and_valuation/alternative_data_pregate_2026-09.md``.

Two things about this fetcher are deliberate and load-bearing:

1. **Truncation is checked for, never assumed away.**  A range query comes back
   truncated at the interface's row cap, and a truncated range looks exactly like
   a successful one on the wire -- whole months of ``stk_holdernumber`` were lost
   that way once already.  So queries are small enough that the cap is nearly
   unreachable, and any response that lands *on* the cap is treated as suspect
   rather than trusted: the per-day fetch flags it, and the windowed fetch splits
   the window into days and refetches.  The test is exact, because a query with
   more rows than the cap returns exactly the cap.
2. **The clock is not applied here.**  ``margin_detail`` must be shifted by one
   day (G0 measured the lag) and ``stk_holdernumber`` must be keyed on
   ``ann_date``.  Both are applied where the signal is joined, so the stored CSV
   stays exactly what the interface said.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from research.experiments.fetch_tushare_daily_basic import call_api, read_token


# Row caps live on the spec, one measured number per endpoint, because a single
# shared constant is exactly how a whole month of ``stk_holdernumber`` once went
# missing: the guard had been set to a value borrowed from a different endpoint,
# and a guard set *above* the true cap never fires, so the truncation arrived
# looking like a clean success.  Rule learned the hard way, applied below:
#
#   * a query with more rows than the cap returns *exactly* the cap, so a response
#     below the cap is provably complete and one at the cap is not;
#   * measure each endpoint's cap, never borrow one.
#
# Measured 2026-09-23 with a one-year range query (2010-01-01 .. 2010-12-31):
# ``moneyflow`` and ``margin_detail`` both returned exactly 6,000 with the span
# collapsed to the tail of the range, and ``stk_holdernumber`` returned exactly
# 5,500 the same way.  ``top_list`` rejected the range outright
# (``api code=50101 必填参数, trade_date``) -- it is date-only, so it has no range
# cap to measure and gets ``None`` rather than a borrowed number.
DEFAULT_ROW_CAP = None

DEFAULT_OUTPUT_DIR = Path("data/raw/tushare/alternative")
DEFAULT_DATASET_DIR = Path("D:/temp/quant_output/datasets")


@dataclass(frozen=True)
class FetchSpec:
    """One candidate signal: which endpoint, which columns, and how to key it.

    ``combine`` says what to do when the interface returns more than one row for
    the same (symbol, date).  Two of these endpoints do exactly that, and neither
    case is a duplicate to be dropped:

    * ``exact`` -- identical rows; safe to collapse.
    * ``unique`` -- several *different* rows (``top_list`` lists a stock once per
      listing reason).  A one-day net flow and a three-day cumulative net flow are
      not the same quantity, so summing them would add apples to oranges while
      picking one would be arbitrary: the day is dropped instead.
    * ``latest_period`` -- several period-ends disclosed in one announcement
      (``stk_holdernumber``).  The count being disclosed is the most recent period
      that has actually ended; a row whose period ends after its own announcement
      is a data error, not a later disclosure.
    """

    name: str
    api: str
    date_column: str
    value_columns: Sequence[str]
    extra_columns: Sequence[str] = field(default_factory=tuple)
    combine: str = "exact"
    #: ``"date"`` sends one ``trade_date`` per call; ``"window"`` sends
    #: ``start_date``/``end_date``.  Which one works is a property of the endpoint,
    #: not a preference: ``stk_holdernumber`` ignores ``trade_date`` outright and
    #: answers every such call with the same full-table first page.
    mode: str = "date"
    #: This endpoint's measured response-row cap, or ``None`` when it has none to
    #: measure.  Not a default and not shared: see the measurement note above.
    #: ``None`` means "no truncation test is possible here", which the manifest
    #: records explicitly rather than reporting a reassuring zero.
    row_cap: Optional[int] = DEFAULT_ROW_CAP

    @property
    def row_cap_measured(self) -> bool:
        return self.row_cap is not None


#: Calendar days per window query.  Small enough that a window reaching the row
#: cap is rare, and windows that do reach it are split into single days rather
#: than trusted.
WINDOW_DAYS = 3


FETCH_SPECS: Sequence[FetchSpec] = (
    FetchSpec(name="moneyflow", api="moneyflow", date_column="trade_date",
              value_columns=("net_mf_amount", "buy_lg_amount", "sell_lg_amount",
                             "buy_elg_amount", "sell_elg_amount"), row_cap=6000),
    FetchSpec(name="margin_detail", api="margin_detail", date_column="trade_date",
              value_columns=("rzye", "rqye", "rzmre", "rzrqye"), row_cap=6000),
    # ``top_list`` is date-only (a range query is rejected), so there is no range
    # cap for it to truncate at -- left unmeasured on purpose instead of borrowing
    # a neighbour's number.  It is also a signal the pre-gate rejected.
    FetchSpec(name="top_list", api="top_list", date_column="trade_date",
              value_columns=("net_amount", "net_rate", "amount"), combine="unique"),
    FetchSpec(name="stk_holdernumber", api="stk_holdernumber", date_column="ann_date",
              value_columns=("holder_num",), extra_columns=("end_date",),
              combine="latest_period", mode="window", row_cap=5500),
)


def resolve_duplicate_keys(spec: FetchSpec, frame: pd.DataFrame) -> tuple:
    """Collapse one spec's repeated (symbol, date) rows, and say what was removed."""

    if frame.empty:
        return frame, {"exact_duplicate_rows": 0, "ambiguous_groups_dropped": 0,
                       "rows_with_impossible_period": 0}

    keys = ["ts_code", spec.date_column]
    before = len(frame)
    frame = frame.drop_duplicates(keep="first")
    exact = before - len(frame)

    report = {"exact_duplicate_rows": int(exact), "ambiguous_groups_dropped": 0,
              "rows_dropped_as_ambiguous": 0, "rows_with_impossible_period": 0}
    if spec.combine == "exact":
        return frame.reset_index(drop=True), report

    if spec.combine == "unique":
        size = frame.groupby(keys, sort=False)["ts_code"].transform("size")
        report["ambiguous_groups_dropped"] = int(frame.loc[size > 1].groupby(keys).ngroups)
        report["rows_dropped_as_ambiguous"] = int((size > 1).sum())
        return frame.loc[size == 1].reset_index(drop=True), report

    if spec.combine == "latest_period":
        announced = pd.to_numeric(frame["ann_date"], errors="coerce")
        period = pd.to_numeric(frame["end_date"], errors="coerce")
        impossible = period > announced
        report["rows_with_impossible_period"] = int(impossible.sum())
        frame = frame.loc[~impossible]
        frame = (frame.assign(_period=pd.to_numeric(frame["end_date"], errors="coerce"))
                      .sort_values(keys + ["_period"])
                      .drop_duplicates(keys, keep="last")
                      .drop(columns="_period"))
        return frame.reset_index(drop=True), report

    raise ValueError("unknown combine rule: {}".format(spec.combine))


def sample_dates(dataset_dir: Path, dataset_ids: Sequence[str],
                 splits: Sequence[str] = ("train", "validation", "test")) -> List[str]:
    """Every date the gate will need, read off the sample files themselves.

    Taken from the samples rather than from a calendar so the fetch cannot drift
    from the rows that will actually be joined to it.

    ``train`` is included because it is not optional: the signals become *model
    features*, and a feature has to exist where the model is fitted.  The first
    fetch covered only validation and test -- enough for a gate that only ever
    scored validation IC, and not enough for the main body that retrains.
    """

    dates = set()
    for dataset_id in dataset_ids:
        for split in splits:
            path = Path(dataset_dir) / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
            if not path.exists():
                raise FileNotFoundError("sample file missing: {}".format(path))
            frame = pd.read_csv(path, usecols=["date"], low_memory=False)
            dates.update(pd.to_datetime(frame["date"]).dt.strftime("%Y%m%d").unique())
    return sorted(dates)


def keep_columns(spec: FetchSpec, payload: pd.DataFrame) -> pd.DataFrame:
    """Narrow a payload to the columns the signal needs, keeping what it has."""

    wanted = ["ts_code", spec.date_column, *spec.value_columns, *spec.extra_columns]
    present = [column for column in wanted if column in payload.columns]
    frame = payload.loc[:, present].copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def fetch_by_date(spec: FetchSpec, token: str, dates: Sequence[str],
                  sleep: float = 0.0, verbose: bool = True) -> tuple:
    """One call per date, always.

    A date *range* is never used, because a truncated range and a complete one are
    the same shape on the wire.  ``stk_holdernumber`` is the proof: it caps range
    queries at 5,500 rows, and a two-week window that came back with exactly that
    many rows looked like a successful fetch while silently dropping whole months.
    A single day cannot be truncated unless one day exceeds the cap, so the guard
    below is a real check rather than a formality.

    When the spec carries no measured cap the guard is skipped and said to be
    skipped -- a guard that cannot fire must not report "0 rows at the cap".
    """

    frames, problems, cap_days = [], [], []
    largest = 0
    cap = spec.row_cap
    iterator = tqdm(dates, desc=spec.name, disable=not verbose)
    for date in iterator:
        try:
            payload = call_api(spec.api, token, {"trade_date": date}, "", timeout=60)
        except Exception as exc:
            problems.append({"date": date, "error": "{}: {}".format(type(exc).__name__, str(exc)[:150])})
            continue
        largest = max(largest, len(payload))
        if cap is not None and len(payload) >= cap:
            cap_days.append({"date": date, "rows": int(len(payload))})
        if len(payload):
            frames.append(keep_columns(spec, payload))
        if sleep:
            time.sleep(sleep)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, {"problems": problems, "at_row_cap": cap_days, "max_rows_in_a_day": largest}


def _window_call(spec: FetchSpec, token: str, start: str, end: str, timeout: int = 60) -> pd.DataFrame:
    return call_api(spec.api, token, {"start_date": start, "end_date": end}, "", timeout=timeout)


def fetch_by_window(spec: FetchSpec, token: str, dates: Sequence[str],
                    window_days: int = WINDOW_DAYS, verbose: bool = True) -> tuple:
    """Small ``start_date``/``end_date`` windows, split further whenever one fills up.

    The truncation test is exact rather than heuristic: a query that has more rows
    than the cap returns *exactly* the cap, so a window that comes back with fewer
    than ``spec.row_cap`` rows is complete, and one that comes back at the cap is
    not.  Only the latter is re-fetched one day at a time -- a single day is the
    smallest unit there is, so it either fits or is unfetchable.

    Windows span calendar days, not trading days, because disclosures do not wait
    for the exchange to open: 2022-10-29 was a Saturday carrying 1,037 of them.

    A windowed endpoint with no measured cap is refused rather than run: the whole
    point of windowing is the truncation test, and there would not be one.
    """

    if spec.row_cap is None:
        raise ValueError(
            "{} has no measured row cap, so a windowed fetch cannot tell a "
            "truncated response from a complete one".format(spec.name))
    cap = spec.row_cap

    days = pd.date_range(pd.to_datetime(min(dates)), pd.to_datetime(max(dates)), freq="D")
    frames, problems, cap_windows, split_days = [], [], [], []
    largest = 0
    windows = [days[index:index + window_days] for index in range(0, len(days), window_days)]
    iterator = tqdm(windows, desc=spec.name, disable=not verbose)
    for window in iterator:
        start, end = window[0].strftime("%Y%m%d"), window[-1].strftime("%Y%m%d")
        try:
            payload = _window_call(spec, token, start, end)
        except Exception as exc:
            problems.append({"window": [start, end],
                             "error": "{}: {}".format(type(exc).__name__, str(exc)[:150])})
            continue
        largest = max(largest, len(payload))
        if len(payload) < cap:
            if len(payload):
                frames.append(keep_columns(spec, payload))
            continue

        cap_windows.append({"window": [start, end], "rows": int(len(payload))})
        for day in window:
            single = day.strftime("%Y%m%d")
            split_days.append(single)
            try:
                one = _window_call(spec, token, single, single)
            except Exception as exc:
                problems.append({"window": [single, single],
                                 "error": "{}: {}".format(type(exc).__name__, str(exc)[:150])})
                continue
            largest = max(largest, len(one))
            if len(one) >= cap:
                # A single day above the cap cannot be fetched at all; recorded
                # rather than silently accepted as that day's full cross-section.
                problems.append({"window": [single, single], "error": "single day exceeds the row cap"})
                continue
            if len(one):
                frames.append(keep_columns(spec, one))

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, {"problems": problems, "at_row_cap": cap_windows,
                   "windows_split_to_days": len(cap_windows), "days_refetched_singly": split_days,
                   "max_rows_in_a_window": largest, "max_rows_in_a_day": None}


def run_fetch(token: str, specs: Sequence[FetchSpec], dates: Sequence[str],
              output_dir: Path, verbose: bool = True, suffix: str = "") -> Dict:
    """Fetch every spec and write one CSV plus a manifest per signal.

    ``suffix`` appends to every filename, and it exists because a smoke test once
    overwrote a completed fetch: ``--limit-days 3`` writes to the same path as a
    full run, so a three-day sample silently replaced 1,419 days of data and the
    file still looked like a valid CSV.  Anything partial must therefore land
    somewhere a full fetch will not mistake for its own output.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dates_requested": len(dates),
              "partial": bool(suffix),
              "date_range": [dates[0], dates[-1]] if len(dates) else [],
              "signals": {}}
    # Refetching one signal must not erase the manifest entries for the others.
    manifest_path = output_dir / "fetch_manifest{}.json".format(suffix)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["signals"].update(previous.get("signals", {}))
        report["partial"] = report["partial"] or bool(previous.get("partial"))

    for spec in specs:
        if spec.mode == "window":
            frame, problems = fetch_by_window(spec, token, dates, verbose=verbose)
        else:
            frame, problems = fetch_by_date(spec, token, dates, verbose=verbose)

        # A repeat of (symbol, date) would silently double a cross-section, and
        # which repeat survives is not a detail -- see ``resolve_duplicate_keys``.
        frame, resolution = resolve_duplicate_keys(spec, frame)

        path = output_dir / "{}{}.csv.gz".format(spec.name, suffix)
        if len(frame):
            frame.to_csv(path, index=False, compression="gzip")
        entry = {
            "api": spec.api,
            "rows": int(len(frame)),
            "symbols": int(frame["ts_code"].nunique()) if len(frame) else 0,
            "date_column": spec.date_column,
            "value_columns": list(spec.value_columns),
            "combine_rule": spec.combine,
            "mode": spec.mode,
            # The cap is an endpoint property and is recorded next to the count that
            # was tested against it, so "0 at the cap" can never be read off a run
            # whose guard does not exist.
            "row_cap": spec.row_cap,
            "row_cap_measured": spec.row_cap_measured,
            **resolution,
            "problems": len(problems["problems"]),
            "windows_at_row_cap": len(problems["at_row_cap"]),
            "max_rows_in_a_window": problems.get("max_rows_in_a_window"),
            "max_rows_in_a_day": problems.get("max_rows_in_a_day"),
            "path": str(path) if len(frame) else None,
        }
        report["signals"][spec.name] = entry
        if verbose:
            # Only one of the two shapes is ever populated, so printing both would
            # mean printing a ``None`` -- and a ``None`` that reaches a format spec is
            # an exception, not a blank column.
            largest = entry["max_rows_in_a_window"] if spec.mode == "window" \
                else entry["max_rows_in_a_day"]
            at_cap = "{}".format(entry["windows_at_row_cap"]) if entry["row_cap_measured"] \
                else "n/a(cap unmeasured)"
            print("  {:<18} rows={:<9} symbols={:<5} max_rows={:<6} ambiguous_dropped={:<6} problems={} at_cap={}".format(
                spec.name, entry["rows"], entry["symbols"], largest,
                entry["ambiguous_groups_dropped"], entry["problems"], at_cap))

    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print("[write] {}".format(manifest_path))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    # The two datasets the gate actually joins.  Verified identical in date coverage
    # to the pre-excess `hs300_pit_v1` (590 + 829 days for each), but naming the same
    # files here removes a discrepancy a later reader would have to go and disprove.
    parser.add_argument("--dataset-ids", nargs="+",
                        default=["excess_hs300_pit_v1", "excess_zz1000_pit_v1"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--signals", nargs="+", default=[spec.name for spec in FETCH_SPECS])
    parser.add_argument("--limit-days", type=int, default=None, help="Smoke-test the first N dates")
    args = parser.parse_args(argv)

    specs = [spec for spec in FETCH_SPECS if spec.name in set(args.signals)]
    dates = sample_dates(Path(args.dataset_dir), args.dataset_ids)
    if args.limit_days:
        dates = dates[: int(args.limit_days)]
    # A partial fetch goes to its own filenames: it is a sample, not a replacement.
    suffix = "_smoke" if args.limit_days else ""
    print("== E8 candidate signal fetch ==")
    print("  sample dates: {} ({} .. {})".format(len(dates), dates[0] if dates else "-",
                                                 dates[-1] if dates else "-"))
    for spec in specs:
        print("  {:<18} api={:<18} mode={:<7} combine={}".format(
            spec.name, spec.api, spec.mode, spec.combine))
    if suffix:
        print("  [partial] writing to *{}.* -- a full fetch will not read these".format(suffix))
    run_fetch(read_token(), specs, dates, Path(args.output_dir), suffix=suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
