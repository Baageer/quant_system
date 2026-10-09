"""Tests for E8 pre-gate G0, the "is this signal's clock knowable at t" check.

Everything runs on synthetic payloads; no network call is made.
"""

import pandas as pd
import pytest

from research.experiments import alternative_data_pregate as gate
from research.experiments.alternative_data_pregate import (
    ANCHOR_API,
    Candidate,
    EventCandidate,
    classify,
    disclosure_field,
    period_field,
)


def payload(**columns):
    """A payload whose only meaningful content is its column list and length."""

    lengths = {len(v) for v in columns.values()}
    size = lengths.pop() if len(lengths) == 1 else 1
    return pd.DataFrame({name: list(values) + [None] * (size - len(list(values)))
                         for name, values in columns.items()})


def test_an_interface_that_publishes_a_disclosure_date_wins_over_the_period_column():
    """``stk_holdernumber`` has both ann_date and end_date; only one is knowable."""

    verdict = classify(Candidate(name="stk_holdernumber", api="stk_holdernumber", fields="", note=""),
                       payload(ts_code=["600519.SH"], ann_date=["20240105"], end_date=["20231231"]),
                       "20260922")

    assert verdict["mode"] == "ann_date"
    assert verdict["timestamp_column"] == "ann_date"
    assert verdict["usable_at_t"] == "yes"
    assert verdict["shift_days"] == 0


def test_a_report_date_is_not_a_disclosure_date():
    """``report_rc``'s only date names the period the estimate is about.

    Treating it as a publication date is exactly the look-ahead the gate exists to
    catch, so the interface is dropped rather than lagged by a guessed amount.
    """

    assert disclosure_field(["ts_code", "report_date", "eps"]) is None
    assert period_field(["ts_code", "report_date", "eps"]) == "report_date"

    verdict = classify(Candidate(name="report_rc", api="report_rc", fields="", note=""),
                       payload(ts_code=["600519.SH"], report_date=["20240105"]), "20260922")

    assert verdict["mode"] == "dropped"
    assert verdict["usable_at_t"] == "no"
    assert verdict["shift_days"] is None


def test_a_trade_date_panel_as_fresh_as_the_anchor_is_usable_at_t():
    verdict = classify(Candidate(name="moneyflow", api="moneyflow", fields="", note=""),
                       payload(ts_code=["600519.SH"], trade_date=["20260922"]), "20260922")

    assert verdict["mode"] == "eod"
    assert verdict["usable_at_t"] == "yes"


def test_a_panel_absent_for_the_anchor_date_is_shifted_not_trusted():
    """Measured on 2026-09-22: margin_detail had 0 rows while daily_basic had 5,554."""

    verdict = classify(Candidate(name="margin_detail", api="margin_detail", fields="", note=""),
                       payload(trade_date=[], ts_code=[]), "20260922")

    assert verdict["mode"] == "eod_lagged"
    assert verdict["usable_at_t"] == "no"
    assert verdict["shift_days"] == 1
    assert "20260922" in verdict["reason"]


def test_an_empty_payload_with_no_trade_date_is_dropped_not_shifted():
    """A shift only rescues an interface whose clock is at least on a trade date."""

    verdict = classify(Candidate(name="mystery", api="mystery", fields="", note=""),
                       payload(ts_code=[], end_date=[]), "20260922")

    assert verdict["mode"] == "dropped"


def test_the_anchor_is_the_interface_the_pipeline_already_trusts():
    assert ANCHOR_API == "daily_basic"


def test_the_probe_asks_for_the_whole_field_inventory(monkeypatch):
    """Hand-picking fields would let the gate miss a disclosure timestamp it hid."""

    seen = {}

    def capture(api_name, token, params, fields="", **kwargs):
        seen["api"] = api_name
        seen["fields"] = fields
        return pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20260922"]})

    monkeypatch.setattr(gate, "call_api", capture)
    gate.probe(Candidate(name="moneyflow", api="moneyflow", fields="ts_code,trade_date", note=""),
               "unused", "20260922")

    assert seen["fields"] == ""


def test_an_event_candidate_probes_a_window_not_a_single_date():
    """``moneyflow``-style probes take trade_date; event tables take start/end."""

    candidate = EventCandidate(name="stk_holdernumber", api="stk_holdernumber", fields="",
                               note="", start_date="20260101", end_date="20260922")

    assert candidate.extra_params == {"start_date": "20260101", "end_date": "20260922"}
    assert Candidate(name="moneyflow", api="moneyflow", fields="", note="").extra_params == {}


def test_a_probe_failure_is_recorded_as_a_verdict_not_raised(monkeypatch):
    """One unreachable interface must not cost the verdicts for the others."""

    monkeypatch.setattr(gate, "newest_anchor_date", lambda token, window=12, timeout=60: "20260922")

    def boom(candidate, token, trade_date, timeout=60):
        if candidate.name == "margin_detail":
            raise ValueError("api code=50101 permission denied")
        return payload(ts_code=["600519.SH"], trade_date=["20260922"])

    monkeypatch.setattr(gate, "probe", boom)
    report = gate.run_g0("unused-token", verbose=False)

    verdicts = {row["signal"]: row for row in report["candidates"]}
    assert verdicts["margin_detail"]["mode"] == "probe_failed"
    assert "permission denied" in verdicts["margin_detail"]["reason"]
    assert verdicts["moneyflow"]["mode"] == "eod"


def test_the_report_carries_the_anchor_date_it_was_measured_against(monkeypatch):
    monkeypatch.setattr(gate, "newest_anchor_date", lambda token, window=12, timeout=60: "20260922")
    monkeypatch.setattr(gate, "probe",
                        lambda c, t, d, timeout=60: payload(ts_code=["600519.SH"], trade_date=["20260922"]))

    report = gate.run_g0("unused-token", verbose=False)

    assert report["anchor_date"] == "20260922"
    assert report["anchor_api"] == "daily_basic"
    assert len(report["candidates"]) == len(gate.CANDIDATES)


def test_the_registered_candidates_are_the_preregistered_ones():
    """The list is part of the pre-registration, so it is pinned here."""

    assert [c.name for c in gate.CANDIDATES] == [
        "moneyflow", "margin_detail", "top_list", "stk_holdernumber", "report_rc",
    ]
    # moneyflow_hsgt is market-level, not cross-sectional; it must not be in the gate.
    assert "moneyflow_hsgt" not in [c.name for c in gate.CANDIDATES]
