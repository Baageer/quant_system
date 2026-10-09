"""Tests for the E8 signal fetcher's key handling and truncation guard.

The vendor returns several rows for the same (symbol, date) on two of the four
endpoints.  Which row survives is a data decision, not a detail, so it is pinned
here rather than left to whatever order the interface happened to use.
"""

import json

import pandas as pd
import pytest

from research.experiments.fetch_alternative_data import (
    FETCH_SPECS,
    FetchSpec,
    fetch_by_date,
    fetch_by_window,
    resolve_duplicate_keys,
    sample_dates,
)


TOP_LIST = FetchSpec(name="top_list", api="top_list", date_column="trade_date",
                     value_columns=("net_amount",), combine="unique")
HOLDER = FetchSpec(name="stk_holdernumber", api="stk_holdernumber", date_column="ann_date",
                   value_columns=("holder_num",), extra_columns=("end_date",),
                   combine="latest_period", mode="window", row_cap=5500)
MONEYFLOW = FetchSpec(name="moneyflow", api="moneyflow", date_column="trade_date",
                      value_columns=("net_mf_amount",), row_cap=6000)


def test_identical_rows_are_collapsed_without_loss():
    frame = pd.DataFrame({"ts_code": ["600519.SH"] * 2, "trade_date": [20240105] * 2,
                          "net_mf_amount": [1.0, 1.0]})

    resolved, report = resolve_duplicate_keys(MONEYFLOW, frame)

    assert len(resolved) == 1
    assert report["exact_duplicate_rows"] == 1


def test_two_different_reasons_on_one_day_drop_the_day_rather_than_guess():
    """A 1-day net flow and a 3-day cumulative net flow are not the same quantity."""

    frame = pd.DataFrame({
        "ts_code": ["002632.SZ", "002632.SZ", "600519.SH"],
        "trade_date": [20240105] * 2 + [20240105],
        "net_amount": [-1.74e7, -1.43e8, 5.0],
    })

    resolved, report = resolve_duplicate_keys(TOP_LIST, frame)

    assert list(resolved["ts_code"]) == ["600519.SH"]
    assert report["ambiguous_groups_dropped"] == 1
    assert report["rows_dropped_as_ambiguous"] == 2


def test_the_latest_period_that_has_actually_ended_is_the_one_disclosed():
    """One announcement carries several period-ends; the newest one is the count."""

    frame = pd.DataFrame({
        "ts_code": ["002637.SZ"] * 2 + ["300950.SZ"],
        "ann_date": [20240420] * 2 + [20240330],
        "end_date": [20231231, 20240331, 20240331],
        "holder_num": [1000, 2000, 14041],
    })

    resolved, _ = resolve_duplicate_keys(HOLDER, frame)

    row = resolved[resolved["ts_code"] == "002637.SZ"].iloc[0]
    assert row["end_date"] == 20240331
    assert row["holder_num"] == 2000


def test_a_period_ending_after_its_own_announcement_is_dropped():
    """A count "as of" a date that had not happened yet is an error, not a disclosure."""

    frame = pd.DataFrame({"ts_code": ["002637.SZ"], "ann_date": [20240318],
                          "end_date": [20241025], "holder_num": [15987]})

    resolved, report = resolve_duplicate_keys(HOLDER, frame)

    assert resolved.empty
    assert report["rows_with_impossible_period"] == 1


def test_an_announcement_left_with_no_usable_period_is_excluded_not_kept():
    """Dropping the impossible row must not fall back to it via ``keep='last'``."""

    frame = pd.DataFrame({"ts_code": ["A", "A"], "ann_date": [20240318] * 2,
                          "end_date": [20241025, 20241231], "holder_num": [1, 2]})

    resolved, _ = resolve_duplicate_keys(HOLDER, frame)

    assert resolved.empty


def test_a_frame_with_no_repeats_is_returned_untouched():
    frame = pd.DataFrame({"ts_code": ["A", "B"], "trade_date": [20240105] * 2,
                          "net_amount": [1.0, 2.0]})

    resolved, report = resolve_duplicate_keys(TOP_LIST, frame)

    assert len(resolved) == 2
    assert report["ambiguous_groups_dropped"] == 0


def test_the_date_universe_comes_from_the_samples_not_a_calendar(tmp_path):
    """A calendar would request days the samples never use, and could miss others."""

    for split, dates in (("train", ["2005-04-29"]),
                         ("validation", ["2019-07-17", "2019-07-18"]),
                         ("test", ["2022-07-20"])):
        pd.DataFrame({"date": dates, "symbol": ["600519"] * len(dates)}).to_csv(
            tmp_path / "trend_probability_{}_tradability_samples_pool_v1.csv".format(split),
            index=False)

    dates = sample_dates(tmp_path, ["pool_v1"])

    assert dates == ["20050429", "20190717", "20190718", "20220720"]


def test_the_date_universe_includes_train_because_features_are_fitted_there(tmp_path):
    """The first fetch covered validation and test only.

    That was enough for a gate that only ever scored validation IC, and not enough
    for the main body: these signals become *features*, and a feature has to exist
    on the rows the model is fitted to.  Reading only the later splits would have
    silently produced a training set with no signal in it.
    """

    for split, dates in (("train", ["2005-04-29"]), ("validation", ["2019-07-17"]),
                         ("test", ["2022-07-20"])):
        pd.DataFrame({"date": dates, "symbol": ["600519"] * len(dates)}).to_csv(
            tmp_path / "trend_probability_{}_tradability_samples_pool_v1.csv".format(split),
            index=False)

    assert sample_dates(tmp_path, ["pool_v1"])[0] == "20050429"

    # A split that is declared but missing must raise rather than be skipped: a
    # quietly narrowed date universe is the same silent-incompleteness failure the
    # row-cap guard exists to prevent, one level up.
    (tmp_path / "trend_probability_train_tradability_samples_pool_v1.csv").unlink()
    with pytest.raises(FileNotFoundError, match="train"):
        sample_dates(tmp_path, ["pool_v1"])


def test_a_response_at_the_row_cap_is_flagged_as_a_possible_truncation(monkeypatch):
    """A truncated response looks exactly like a successful one on the wire."""

    def at_cap(api_name, token, params, fields="", **kwargs):
        size = MONEYFLOW.row_cap
        return pd.DataFrame({"ts_code": ["600519.SH"] * size, "trade_date": [20240105] * size,
                             "net_mf_amount": [1.0] * size})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", at_cap)
    frame, problems = fetch_by_date(MONEYFLOW, "unused", ["20240105"], verbose=False)

    assert len(frame) == MONEYFLOW.row_cap
    assert problems["at_row_cap"] == [{"date": "20240105", "rows": MONEYFLOW.row_cap}]


def test_each_endpoints_row_cap_is_the_measured_one_not_a_borrowed_guess():
    """Caps are per endpoint, and one shared constant is how data went missing.

    ``stk_holdernumber`` was measured to truncate range queries at 5,500 while the
    guard looked for 6,000 -- a guard set *above* the true cap never fires, so the
    truncation arrived looking like a clean success.  Measured 2026-09-23 with a
    one-year range (2010-01-01 .. 2010-12-31): ``moneyflow`` and ``margin_detail``
    each came back with exactly 6,000 rows, ``stk_holdernumber`` with exactly
    5,500, every one of them with the returned span collapsed to a sliver of the
    requested range -- which is what makes the number the cap rather than a count.
    """

    caps = {spec.name: spec.row_cap for spec in FETCH_SPECS}

    assert caps["moneyflow"] == 6000
    assert caps["margin_detail"] == 6000
    assert caps["stk_holdernumber"] == 5500


def test_an_endpoint_with_no_range_mode_is_left_unmeasured_rather_than_borrowed():
    """``top_list`` rejects a range query outright (``必填参数, trade_date``).

    It is date-only, so it has no range cap to measure.  Borrowing a neighbour's
    number would put a guard in the manifest that never fires and cannot be told
    apart from a guard that fired zero times -- the exact confusion this whole
    mechanism exists to prevent.  Left ``None``, and said so.
    """

    spec = next(spec for spec in FETCH_SPECS if spec.name == "top_list")

    assert spec.row_cap is None
    assert spec.row_cap_measured is False


def test_a_window_fetch_without_a_measured_cap_refuses_to_run():
    """Windowing exists to test for truncation; without a cap there is no test."""

    spec = FetchSpec(name="top_list", api="top_list", date_column="trade_date",
                     value_columns=("net_amount",), mode="window")

    with pytest.raises(ValueError, match="no measured row cap"):
        fetch_by_window(spec, "unused", ["20240102"], verbose=False)


def test_an_unmeasured_guard_is_recorded_as_unmeasured(monkeypatch, tmp_path):
    """``at_cap=0`` must not be readable off a run whose guard cannot fire."""

    def payload(api_name, token, params, fields="", **kwargs):
        return pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": [20240102],
                             "net_amount": [1.0]})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", payload)
    from research.experiments.fetch_alternative_data import run_fetch

    report = run_fetch("unused", [TOP_LIST], ["20240102"], tmp_path, verbose=False)

    assert report["signals"]["top_list"]["row_cap"] is None
    assert report["signals"]["top_list"]["row_cap_measured"] is False


def test_no_spec_uses_a_wide_date_range():
    """Windows are days wide, not months: the cap is the thing being outrun."""

    from research.experiments import fetch_alternative_data as module

    assert not hasattr(module, "fetch_by_range")
    assert module.WINDOW_DAYS <= 7


def test_the_disclosure_signal_windows_on_dates_the_others_ask_by_day():
    """``stk_holdernumber`` ignores ``trade_date``; every such call returns the same page.

    The other three endpoints answer a per-day query and would be *starved* by a
    window, so the mode is a property of the endpoint rather than a preference.
    """

    modes = {spec.name: spec.mode for spec in FETCH_SPECS}

    assert modes["stk_holdernumber"] == "window"
    assert [mode for name, mode in modes.items() if name != "stk_holdernumber"] == \
        ["date", "date", "date"]


def test_a_window_that_fills_the_cap_is_split_into_days_not_trusted(monkeypatch):
    """Exactly the cap means "more rows exist": the window is refetched day by day."""

    calls = []

    def sized(api_name, token, params, fields="", **kwargs):
        calls.append(params)
        if "start_date" in params and params["start_date"] != params["end_date"]:
            size = HOLDER.row_cap               # a truncated window
        else:
            size = 4                            # each day is complete
        return pd.DataFrame({
            "ts_code": ["600519.SH"] * size,
            "ann_date": [params.get("start_date", "20240101")] * size,
            "end_date": [params.get("start_date", "20240101")] * size,
            "holder_num": [1.0] * size,
        })

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", sized)
    frame, problems = fetch_by_window(HOLDER, "unused", ["20240102", "20240103", "20240104"],
                                      window_days=3, verbose=False)

    assert problems["windows_split_to_days"] == 1
    assert len(frame) == 12                     # 3 days x 4 rows, none of the truncation
    assert len(problems["problems"]) == 0


def test_a_window_below_the_cap_is_taken_at_face_value(monkeypatch):
    """Below the cap the response is provably complete, so no extra call is spent."""

    calls = []

    def small(api_name, token, params, fields="", **kwargs):
        calls.append(params)
        return pd.DataFrame({"ts_code": ["600519.SH"], "ann_date": ["20240102"],
                             "end_date": ["20240102"], "holder_num": [1.0]})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", small)
    frame, problems = fetch_by_window(HOLDER, "unused", ["20240102", "20240104"],
                                      window_days=3, verbose=False)

    assert len(calls) == 1                      # one window, no day-by-day retry
    assert len(frame) == 1
    assert problems["windows_split_to_days"] == 0


def test_the_largest_day_is_recorded_so_the_guard_has_evidence(monkeypatch):
    """A cap that never fires is only reassuring if something says how close it came."""

    def sized(api_name, token, params, fields="", **kwargs):
        size = {"20240105": 3, "20240106": 7}[params["trade_date"]]
        return pd.DataFrame({"ts_code": ["600519.SH"] * size,
                             "trade_date": [params["trade_date"]] * size,
                             "net_mf_amount": [1.0] * size})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", sized)
    _, problems = fetch_by_date(MONEYFLOW, "unused", ["20240105", "20240106"], verbose=False)

    assert problems["max_rows_in_a_day"] == 7


def test_one_failing_day_does_not_cost_the_others(monkeypatch):
    def flaky(api_name, token, params, fields="", **kwargs):
        if params["trade_date"] == "20240106":
            raise RuntimeError("api code=40203 rate limited")
        return pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": [params["trade_date"]],
                             "net_mf_amount": [1.0]})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", flaky)
    frame, problems = fetch_by_date(MONEYFLOW, "unused", ["20240105", "20240106", "20240107"],
                                    verbose=False)

    assert len(frame) == 2
    assert len(problems["problems"]) == 1
    assert "rate limited" in problems["problems"][0]["error"]


def test_run_fetch_reports_both_modes_without_a_none_in_a_format_string(monkeypatch, tmp_path):
    """The two modes fill different keys; the run summary must not print the other's ``None``.

    This is not hypothetical: the summary line was written for the windowed mode, and
    a plain per-day run then died on ``None.__format__`` -- after its data had already
    been fetched, so the failure cost the whole run.
    """

    def payload(api_name, token, params, fields="", **kwargs):
        rows = 1 if "trade_date" in params else 2
        return pd.DataFrame({"ts_code": ["600519.SH"] * rows,
                             "trade_date": [20240102] * rows, "ann_date": [20240102] * rows,
                             "end_date": [20240102] * rows, "holder_num": [1.0] * rows,
                             "net_mf_amount": [1.0] * rows})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", payload)
    from research.experiments.fetch_alternative_data import run_fetch

    report = run_fetch("unused", [MONEYFLOW, HOLDER], ["20240102", "20240104"],
                       tmp_path, verbose=True)

    assert report["signals"]["moneyflow"]["max_rows_in_a_day"] == 1
    assert report["signals"]["moneyflow"]["max_rows_in_a_window"] is None
    assert report["signals"]["stk_holdernumber"]["max_rows_in_a_window"] == 2
    assert (tmp_path / "fetch_manifest.json").exists()


def test_a_partial_fetch_cannot_overwrite_a_full_one(monkeypatch, tmp_path):
    """A 3-day smoke run once replaced 1,419 days of data and looked like a valid CSV."""

    def payload(api_name, token, params, fields="", **kwargs):
        return pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": [20240102],
                             "net_mf_amount": [1.0]})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", payload)
    from research.experiments.fetch_alternative_data import run_fetch

    full = run_fetch("unused", [MONEYFLOW], ["20240102"], tmp_path, verbose=False)
    partial = run_fetch("unused", [MONEYFLOW], ["20240102"], tmp_path, verbose=False,
                        suffix="_smoke")

    assert full["signals"]["moneyflow"]["path"].endswith("moneyflow.csv.gz")
    assert partial["signals"]["moneyflow"]["path"].endswith("moneyflow_smoke.csv.gz")
    assert partial["partial"] is True
    # The partial run gets its own manifest, so it cannot smuggle a 3-day sample
    # into the record a full run is read back from.
    smoke = json.loads((tmp_path / "fetch_manifest_smoke.json").read_text(encoding="utf-8"))
    assert smoke["partial"] is True
    assert smoke["signals"]["moneyflow"]["path"].endswith("moneyflow_smoke.csv.gz")


def test_refetching_one_signal_keeps_the_others_manifest_entries(monkeypatch, tmp_path):
    """A single-signal refetch is the normal repair path; it must not erase the rest."""

    def payload(api_name, token, params, fields="", **kwargs):
        return pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": [20240102],
                             "ann_date": [20240102], "end_date": [20240102],
                             "holder_num": [1.0], "net_mf_amount": [1.0]})

    monkeypatch.setattr("research.experiments.fetch_alternative_data.call_api", payload)
    from research.experiments.fetch_alternative_data import run_fetch

    run_fetch("unused", [MONEYFLOW], ["20240102"], tmp_path, verbose=False)
    report = run_fetch("unused", [HOLDER], ["20240102"], tmp_path, verbose=False)

    assert set(report["signals"]) == {"moneyflow", "stk_holdernumber"}


def test_a_signal_the_preregistration_dropped_is_not_fetched():
    """``report_rc`` was eliminated at G0; fetching it would reopen a closed door."""

    assert "report_rc" not in [spec.name for spec in FETCH_SPECS]
    assert "moneyflow_hsgt" not in [spec.name for spec in FETCH_SPECS]
