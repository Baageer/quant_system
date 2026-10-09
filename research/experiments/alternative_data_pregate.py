"""E8 pre-gate G0: does each candidate signal's clock point at a knowable instant?

Pre-registration: ``docs/research/03_factor_and_valuation/alternative_data_pregate_2026-09.md`` section 4.

G0 looks at **no IC at all**.  It answers one question per candidate -- "could this
value have been known at the signal date?" -- and it accepts only two kinds of
evidence:

1. the interface's own field inventory (is there a *disclosure* timestamp?), and
2. a **measured** publication lag against an anchor interface.

The anchor is ``daily_basic``, and it is not an arbitrary choice: the production
pipeline already joins that interface on the signal date and executes at the next
open, so "same freshness as ``daily_basic``" *is* the definition of end-of-day
data that is usable at ``t``.  A candidate whose newest row is missing for the
newest date the anchor already has is a day behind -- that is measured, not
assumed from how an exchange is believed to operate.

A candidate that cannot be shown to be knowable is not "used carefully"; it is
dropped.  Guessing a lag is how look-ahead gets in.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from research.experiments.fetch_tushare_daily_basic import call_api, read_token


#: Interfaces in this order are preferred as the *disclosure* timestamp.  A column
#: here means "the interface tells us when this became public".
DISCLOSURE_FIELDS = ("ann_date", "ann_time", "pub_date", "disclosure_date", "create_time")

#: Columns that only say *which period* a row describes.  ``report_date`` is here
#: on purpose: an analyst report's own date is not the date it became available.
PERIOD_FIELDS = ("trade_date", "end_date", "report_date")

ANCHOR_API = "daily_basic"
ANCHOR_FIELDS = "ts_code,trade_date"


@dataclass(frozen=True)
class Candidate:
    """One alternative-data signal, described by where its clock lives."""

    name: str
    api: str
    fields: str
    note: str

    @property
    def extra_params(self) -> Dict[str, str]:
        """Params a bare ``trade_date`` probe needs.  Overridden by subclasses."""

        return {}


@dataclass(frozen=True)
class EventCandidate(Candidate):
    """A candidate that cannot be probed with a single ``trade_date``."""

    start_date: str = ""
    end_date: str = ""

    @property
    def extra_params(self) -> Dict[str, str]:
        return {"start_date": self.start_date, "end_date": self.end_date}


CANDIDATES: Sequence[Candidate] = (
    Candidate(name="moneyflow", api="moneyflow",
              fields="ts_code,trade_date,net_mf_amount", note="个股资金流"),
    Candidate(name="margin_detail", api="margin_detail",
              fields="trade_date,ts_code,rzye,rzmre", note="融资融券余额"),
    Candidate(name="top_list", api="top_list",
              fields="trade_date,ts_code,net_amount", note="龙虎榜（稀疏）"),
    EventCandidate(name="stk_holdernumber", api="stk_holdernumber",
                   fields="ts_code,ann_date,end_date,holder_num", note="股东户数",
                   start_date="", end_date=""),
    EventCandidate(name="report_rc", api="report_rc",
                   fields="ts_code,report_date,eps", note="分析师一致预期",
                   start_date="", end_date=""),
)


def disclosure_field(columns: Sequence[str]) -> Optional[str]:
    """The interface's own disclosure timestamp, if it publishes one."""

    for field in DISCLOSURE_FIELDS:
        if field in columns:
            return field
    return None


def period_field(columns: Sequence[str]) -> Optional[str]:
    """The column naming the period a row describes (not when it was published)."""

    for field in PERIOD_FIELDS:
        if field in columns:
            return field
    return None


def newest_anchor_date(token: str, window: int = 12, timeout: int = 60) -> str:
    """The newest ``trade_date`` the anchor interface already has published.

    A closed trading day is required, not a calendar day: the anchor is queried
    over a short recent window and the maximum date with rows wins.  Today may or
    may not be in it -- whichever it is, the answer is the same reference point for
    every candidate.
    """

    frame = call_api(ANCHOR_API, token, {"start_date": _recent_start(window),
                                         "end_date": _today()},
                     ANCHOR_FIELDS, timeout=timeout)
    if "trade_date" not in frame.columns or frame.empty:
        raise ValueError("anchor {} returned no trade_date rows".format(ANCHOR_API))
    return str(frame["trade_date"].dropna().max())


def probe(candidate: Candidate, token: str, trade_date: str, timeout: int = 60) -> pd.DataFrame:
    """One candidate's payload for a single date, or for its configured window.

    ``fields`` is deliberately left empty so the interface returns **its own** full
    column inventory.  Asking for a hand-picked list would mean showing the gate
    only the columns already believed to matter -- and a disclosure timestamp left
    off that list would make an interface look unknowable when it is not.  The
    evidence has to be the interface's answer, not the question.
    """

    params = {"trade_date": trade_date}
    params.update(candidate.extra_params)
    if isinstance(candidate, EventCandidate):
        # Event tables have no trade_date; probe the window ending at the anchor.
        params = {"start_date": candidate.start_date or trade_date,
                  "end_date": candidate.end_date or trade_date}
    return call_api(candidate.api, token, params, "", timeout=timeout)


def classify(candidate: Candidate, payload: pd.DataFrame, trade_date: str) -> Dict[str, object]:
    """The G0 verdict for one candidate: how its clock must be treated.

    The rules are the pre-registered ones, in order:

    1. A disclosure timestamp wins outright -- use it, never the period column.
    2. Otherwise the value must be as fresh as the anchor for ``trade_date``.
    3. One day behind the anchor is not a rejection but it is not usable at ``t``
       either: it is shifted.  The shift is recorded, not silently applied.
    4. No knowable timestamp at all -> dropped.  This is where ``report_rc`` falls:
       it carries ``report_date``, which says which period the estimate is about.
    """

    columns = list(payload.columns)
    verdict = {"signal": candidate.name, "api": candidate.api, "note": candidate.note,
               "columns": columns, "rows": int(len(payload))}

    disclosed = disclosure_field(columns)
    if disclosed:
        verdict.update(mode="ann_date", usable_at_t="yes",
                       timestamp_column=disclosed, shift_days=0,
                       reason="interface publishes a disclosure timestamp; period column not used")
        return verdict

    period = period_field(columns)
    if period != "trade_date":
        verdict.update(mode="dropped", usable_at_t="no", timestamp_column=period or "",
                       shift_days=None,
                       reason="no disclosure timestamp and no trade_date: the clock cannot be placed")
        return verdict

    if len(payload):
        verdict.update(mode="eod", usable_at_t="yes", timestamp_column="trade_date",
                       shift_days=0,
                       reason="as fresh as the {} anchor for {}".format(ANCHOR_API, trade_date))
        return verdict

    verdict.update(mode="eod_lagged", usable_at_t="no", timestamp_column="trade_date",
                   shift_days=1,
                   reason="absent for {} while the anchor has it: one day behind".format(trade_date))
    return verdict


def run_g0(token: str, candidates: Sequence[Candidate] = CANDIDATES, verbose: bool = True) -> Dict:
    """Check every candidate's clock against the anchor and return the verdicts."""

    anchor_date = newest_anchor_date(token)
    rows = []
    for candidate in candidates:
        try:
            payload = probe(candidate, token, anchor_date)
            verdict = classify(candidate, payload, anchor_date)
        except Exception as exc:  # a probe failure is a finding, not a crash
            verdict = {"signal": candidate.name, "api": candidate.api, "note": candidate.note,
                       "rows": 0, "columns": [], "mode": "probe_failed", "usable_at_t": "unknown",
                       "timestamp_column": "", "shift_days": None,
                       "reason": "{}: {}".format(type(exc).__name__, str(exc)[:200])}
        rows.append(verdict)
        if verbose:
            print("  {:<18} {:<12} rows={:<6} {}".format(
                verdict["signal"], verdict["mode"], verdict["rows"], verdict["reason"]))
    return {"anchor_api": ANCHOR_API, "anchor_date": anchor_date, "candidates": rows}


def _today() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


def _recent_start(window_days: int) -> str:
    return (pd.Timestamp.today() - pd.Timedelta(days=int(window_days))).strftime("%Y%m%d")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None, help="Where to write the G0 verdicts as JSON")
    args = parser.parse_args(argv)

    print("== E8 pre-gate G0: is each signal's clock knowable at t? ==")
    report = run_g0(read_token())
    print("  anchor {} newest published trade_date = {}".format(
        report["anchor_api"], report["anchor_date"]))
    for row in report["candidates"]:
        print("  -> {:<18} {}".format(row["signal"], row["usable_at_t"]))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("[write] {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
