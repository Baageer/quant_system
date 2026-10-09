"""Tests for the E8 pre-gate: G1 (direction knowability) and G2 (non-redundancy).

Everything here is synthetic.  The point is to pin the *clock* semantics and the
cross-pool logic -- the two places where a wrong rule would produce a confident
verdict rather than an error.
"""

import numpy as np
import pandas as pd
import pytest

from research.experiments import alternative_data_gate as gate
from research.experiments.alternative_data_gate import Clock, evaluate_signal, to_symbols


def sample_frame(split, dates, symbols, label_by_day, feature=None):
    """A minimal stand-in for a split: keys, label, and one redundancy target."""

    rows = []
    for symbol in symbols:
        for date, label in zip(dates, label_by_day):
            rows.append({"date": pd.Timestamp(date), "symbol": symbol,
                         "source_split": split, gate.LABEL: label,
                         "frozen_feature": feature.get(symbol, 0.0) if feature else 0.0})
    return pd.DataFrame(rows)


def test_an_exchange_suffix_is_stripped_before_joining():
    """Sample files key on ``600519``; the vendor returns ``600519.SH``."""

    stripped = to_symbols(pd.Series(["600519.SH", "000001.SZ", "600519"]))

    assert list(stripped) == ["600519", "000001", "600519"]


def test_a_numeric_symbol_keeps_its_leading_zeros():
    assert list(to_symbols(pd.Series(["1", "1.SZ"]))) == ["000001", "000001"]


def test_the_panel_is_transformed_then_lagged(tmp_path):
    """A one-day-lagged interface is used a day late, not confidently on the same day.

    ``rzye`` doubles on 01-03 (+100%).  With a one-day publication lag that move
    must be attributed to 01-04, not to 01-03 -- and the date must stay put while
    the value moves, or the shift turns into look-ahead.
    """

    frame = pd.DataFrame({
        "ts_code": ["600519.SH"] * 4,
        "trade_date": [20240102, 20240103, 20240104, 20240105],
        "rzye": [100.0, 100.0, 200.0, 200.0],
    })
    frame.to_csv(tmp_path / "margin_detail.csv.gz", index=False, compression="gzip")

    panel = gate.load_signal_panel(tmp_path, Clock(name="margin_detail", date_column="trade_date",
                                                   value_column="rzye", form="pct_change",
                                                   lag_days=1))
    by_date = dict(zip(panel["known_date"].dt.strftime("%Y%m%d"), panel["value"]))

    assert by_date["20240104"] == pytest.approx(0.0)   # knowable on 01-04: the flat move
    assert by_date["20240105"] == pytest.approx(1.0)   # the +100% of 01-04 arrives a day late
    assert "20240103" not in by_date                   # nothing precedes it to be knowable


def test_a_zero_base_yields_no_growth_rather_than_an_infinity(tmp_path):
    """A newly margin-traded name must not own the day's ranking by dividing by zero."""

    frame = pd.DataFrame({"ts_code": ["600519.SH", "600519.SH"],
                          "trade_date": [20240102, 20240103],
                          "rzye": [0.0, 50.0]})
    (tmp_path / "margin_detail.csv.gz").parent.mkdir(exist_ok=True)
    frame.to_csv(tmp_path / "margin_detail.csv.gz", index=False, compression="gzip")

    panel = gate.load_signal_panel(tmp_path, Clock(name="margin_detail", date_column="trade_date",
                                                   value_column="rzye", form="pct_change"))
    assert panel["value"].isna().all() or np.isfinite(panel["value"]).all()


def test_the_asof_join_uses_the_latest_announcement_at_or_before_t():
    samples = pd.DataFrame({"date": pd.to_datetime(["2024-02-15", "2024-05-15"]),
                            "symbol": ["600519", "600519"], "source_split": "validation",
                            gate.LABEL: [0.0, 0.0]})
    panel = pd.DataFrame({"symbol": ["600519"] * 3,
                          "known_date": pd.to_datetime(["2024-01-10", "2024-02-01", "2024-06-01"]),
                          "value": [1.0, 2.0, 3.0]})
    clock = Clock(name="stk_holdernumber", date_column="ann_date", value_column="holder_num",
                  form="pct_change", asof=True)

    merged = gate.attach_signal(samples, panel, clock).sort_values("date")

    assert list(merged["value"]) == [2.0, 2.0]        # 2024-06-01 is in the future for both


def test_the_asof_join_reports_how_stale_the_disclosure_is():
    samples = pd.DataFrame({"date": pd.to_datetime(["2024-02-15"]), "symbol": ["600519"],
                            "source_split": "validation", gate.LABEL: [0.0]})
    panel = pd.DataFrame({"symbol": ["600519"], "known_date": pd.to_datetime(["2024-01-10"]),
                          "value": [1.0]})
    clock = Clock(name="stk_holdernumber", date_column="ann_date", value_column="holder_num",
                  form="pct_change", asof=True)

    merged = gate.attach_signal(samples, panel, clock)

    assert merged["signal_age_days"].iloc[0] == 36


def test_an_exact_clock_does_not_reach_backwards():
    """``moneyflow`` is usable at t, so a missing day stays missing, not filled."""

    samples = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                            "symbol": ["600519", "600519"], "source_split": "validation",
                            gate.LABEL: [0.0, 0.0]})
    panel = pd.DataFrame({"symbol": ["600519"], "known_date": pd.to_datetime(["2024-01-02"]),
                          "value": [7.0]})
    clock = Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
                  form="level")

    merged = gate.attach_signal(samples, panel, clock).sort_values("date")

    assert list(merged["value"].isna()) == [False, True]


def _two_half_panel(first_ic_sign, second_ic_sign, test_label_sign=1.0):
    """Validation in two halves with a controllable sign, plus a test block."""

    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    symbols = ["s{:02d}".format(i) for i in range(30)]
    labels, values = [], []
    for index, date in enumerate(dates):
        direction = first_ic_sign if index < 4 else second_ic_sign
        for rank, symbol in enumerate(symbols):
            values.append({"date": date, "symbol": symbol,
                           "value": direction * rank,
                           gate.LABEL: rank / 1000.0, "source_split": "validation",
                           "frozen_feature": 0.0})
    test_dates = pd.date_range("2024-06-01", periods=4, freq="D")
    test_rows = []
    for date in test_dates:
        for rank, symbol in enumerate(symbols):
            test_rows.append({"date": date, "symbol": symbol,
                              "value": test_label_sign * rank,
                              gate.LABEL: rank / 1000.0, "source_split": "test",
                              "frozen_feature": 0.0})
    return pd.DataFrame(values + test_rows)


def test_halves_with_opposite_signs_fail_criterion_1a():
    panel = _two_half_panel(first_ic_sign=1.0, second_ic_sign=-1.0)
    clock = Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
                  form="level")

    row = evaluate_signal(panel, ["frozen_feature"], clock)

    assert row["halves"]["first"]["mean_ic"] > 0
    assert row["halves"]["second"]["mean_ic"] < 0
    assert row["criteria"]["1a"] is False


def test_halves_with_the_same_sign_pass_1a_and_a_strong_one_passes_1b():
    panel = _two_half_panel(first_ic_sign=1.0, second_ic_sign=1.0)
    clock = Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
                  form="level")

    row = evaluate_signal(panel, ["frozen_feature"], clock)

    assert row["criteria"]["1a"] is True
    assert row["criteria"]["1b"] is (abs(row["halves"]["first"]["mean_ic"]) >= gate.MIN_ABS_IC
                                     and abs(row["halves"]["second"]["mean_ic"]) >= gate.MIN_ABS_IC)


def test_no_criterion_reads_the_test_split():
    """Test is reported; if it could change a verdict the test numbers would be spent."""

    clock = Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
                  form="level")
    first = evaluate_signal(_two_half_panel(1.0, 1.0, test_label_sign=1.0), ["frozen_feature"], clock)
    flipped = evaluate_signal(_two_half_panel(1.0, 1.0, test_label_sign=-1.0), ["frozen_feature"], clock)

    assert first["criteria"] == flipped["criteria"]
    assert first["test"]["mean_ic"] == pytest.approx(-flipped["test"]["mean_ic"])


def test_a_signal_passing_one_pool_alone_fails_1c():
    rows = [{"signal": "moneyflow", "pool": "hs300", "judged": True,
             "validation": {"mean_ic": 0.05}, "criteria": {"1a": True, "1b": True}},
            {"signal": "moneyflow", "pool": "zz1000", "judged": True,
             "validation": {"mean_ic": 0.01}, "criteria": {"1a": False, "1b": False}}]

    gate.apply_cross_pool(rows)

    assert all(row["criteria"]["1c"] is False for row in rows)
    assert all(row["criteria"]["passes"] is False for row in rows)


def test_pools_passing_with_opposite_signs_fail_1c():
    """A sign that flips between pools is not a knowable direction, it is two guesses."""

    rows = [{"signal": "moneyflow", "pool": "hs300", "judged": True,
             "validation": {"mean_ic": 0.05}, "criteria": {"1a": True, "1b": True}},
            {"signal": "moneyflow", "pool": "zz1000", "judged": True,
             "validation": {"mean_ic": -0.05}, "criteria": {"1a": True, "1b": True}}]

    gate.apply_cross_pool(rows)

    assert all(row["criteria"]["1c"] is False for row in rows)


def test_pools_passing_with_the_same_sign_pass_1c():
    rows = [{"signal": "moneyflow", "pool": "hs300", "judged": True,
             "validation": {"mean_ic": 0.05}, "criteria": {"1a": True, "1b": True, "2": True}},
            {"signal": "moneyflow", "pool": "zz1000", "judged": True,
             "validation": {"mean_ic": 0.03}, "criteria": {"1a": True, "1b": True, "2": True}}]

    gate.apply_cross_pool(rows)

    assert all(row["criteria"]["1c"] is True for row in rows)
    assert all(row["criteria"]["passes"] is True for row in rows)


def test_a_report_only_form_never_passes_the_gate():
    """The size-proxy forms are computed to be shown, not to be judged."""

    rows = [{"signal": "margin_detail", "pool": "hs300", "judged": True,
             "validation": {"mean_ic": 0.05}, "criteria": {"1a": True, "1b": True, "2": True}},
            {"signal": "margin_detail", "pool": "zz1000", "judged": True,
             "validation": {"mean_ic": 0.05}, "criteria": {"1a": True, "1b": True, "2": True}}]
    gate.apply_cross_pool(rows)
    shown = dict(rows[0])
    shown["judged"] = False
    shown["criteria"] = {"1a": True, "1b": True, "1c": True, "2": True}

    gate.apply_cross_pool([*rows, shown])

    assert shown["criteria"]["passes"] is False


def test_redundancy_fails_at_the_threshold_and_passes_below_it():
    """0.85 is the house's own correlation_threshold, so the boundary is closed."""

    rows = [{"criteria": {}, "redundancy": {"a": 0.50, "b": 0.85}},
            {"criteria": {}, "redundancy": {"a": 0.50, "b": 0.8499}}]

    gate.apply_g2(rows)

    assert rows[0]["criteria"]["2"] is False
    assert rows[0]["max_abs_rank_corr_feature"] == "b"
    assert rows[1]["criteria"]["2"] is True


def test_redundancy_uses_the_absolute_value():
    """A perfectly *negated* copy of a frozen feature is still a duplicate."""

    rows = [{"criteria": {}, "redundancy": {"a": -0.95}}]

    gate.apply_g2(rows)

    assert rows[0]["max_abs_rank_corr"] == pytest.approx(0.95)
    assert rows[0]["criteria"]["2"] is False


def test_a_signal_with_no_paired_rows_is_reported_not_crashed():
    clock = Clock(name="top_list", date_column="trade_date", value_column="net_amount", form="level")
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "symbol": ["600519"],
                          "source_split": "validation", gate.LABEL: [0.0], "value": [np.nan],
                          "frozen_feature": [0.0]})

    row = evaluate_signal(panel, ["frozen_feature"], clock)

    assert row["paired_rows"] == 0
    assert row["criteria"] == {"1a": False, "1b": False}
    assert np.isnan(row["validation"]["mean_ic"])
    # G2 must then also refuse rather than treat "no correlation measured" as passing.
    verdict = {"criteria": {}, "redundancy": row["redundancy"]}
    gate.apply_g2([verdict])
    assert verdict["criteria"]["2"] is False


def test_coverage_is_reported_so_a_sparse_signal_cannot_hide_behind_its_ic():
    """``top_list`` covers a handful of names a day; its IC must be read with that."""

    clock = Clock(name="top_list", date_column="trade_date", value_column="net_amount", form="level")
    panel = _two_half_panel(1.0, 1.0)
    panel.loc[panel["source_split"] == "validation", "value"] = np.nan
    panel.loc[panel.index[:10], "value"] = 1.0

    row = evaluate_signal(panel, ["frozen_feature"], clock)

    assert 0.0 < row["validation_coverage"] < 1.0
    assert row["validation_label_rows"] > row["validation_rows"]


def test_the_registered_clocks_are_the_preregistered_ones():
    judged = [(clock.name, clock.form, clock.lag_days) for clock in gate.CLOCKS if clock.judged]

    assert judged == [("moneyflow", "level", 0), ("margin_detail", "pct_change", 1),
                      ("top_list", "level", 0), ("stk_holdernumber", "pct_change", 0)]
    assert "report_rc" not in [clock.name for clock in gate.CLOCKS]


def test_the_two_frozen_feature_sets_are_not_the_same_set():
    """HS300 has 28 frozen features and ZZ1000 has 30; G2 must not share one list."""

    assert gate.POOL_MODELS["hs300"]["model_id"] != gate.POOL_MODELS["zz1000"]["model_id"]
    assert gate.POOL_MODELS["hs300"]["dataset_id"] == "excess_hs300_pit_v1"
    assert gate.POOL_MODELS["zz1000"]["dataset_id"] == "excess_zz1000_pit_v1"


# --- the chunked read -------------------------------------------------------
#
# The panel is now read in chunks for row-local clocks, because the moneyflow file
# is 21.9 M rows once the training window is included and reading it whole costs
# several gigabytes of RSS.  Chunking a row-local transform is the same computation,
# but "the same computation" is an argument -- these pin it to a comparison.

LEVEL = Clock(name="moneyflow", date_column="trade_date", value_column="net_mf_amount",
              form="level")


def write_moneyflow(tmp_path, rows):
    pd.DataFrame({"ts_code": [row[0] for row in rows],
                  "trade_date": [row[1] for row in rows],
                  "net_mf_amount": [row[2] for row in rows],
                  "buy_lg_amount": [0.0] * len(rows)}).to_csv(
        tmp_path / "moneyflow.csv.gz", index=False, compression="gzip")


def test_a_chunked_panel_equals_the_whole_file_panel(tmp_path):
    """The chunk boundary is where a row-local transform could still go wrong."""

    write_moneyflow(tmp_path, [("600519.SH", 20240100 + day, float(day))
                               for day in range(1, 21)])

    whole = gate.load_signal_panel(tmp_path, LEVEL, chunk_rows=0)
    chunked = gate.load_signal_panel(tmp_path, LEVEL, chunk_rows=3)

    assert len(chunked) == len(whole) == 20
    assert chunked.equals(whole)


def test_a_clock_with_a_shift_is_never_chunked(tmp_path):
    """A growth rate computed across a chunk boundary is a fact about the boundary.

    ``margin_detail`` shifts by symbol and by day, so it needs the whole series; the
    loader must refuse to chunk it rather than produce a panel whose first row of
    every chunk has no predecessor.
    """

    pd.DataFrame({"ts_code": ["600519.SH"] * 6,
                  "trade_date": [20240102, 20240103, 20240104,
                                 20240105, 20240108, 20240109],
                  "rzye": [100.0, 100.0, 200.0, 200.0, 400.0, 400.0]}).to_csv(
        tmp_path / "margin_detail.csv.gz", index=False, compression="gzip")
    clock = Clock(name="margin_detail", date_column="trade_date", value_column="rzye",
                  form="pct_change")

    whole = gate.load_signal_panel(tmp_path, clock, chunk_rows=0)
    chunked = gate.load_signal_panel(tmp_path, clock, chunk_rows=2)

    # The doubled values are the even-indexed rows, so a chunked read would report a
    # leap at each boundary that never happened.  Pinned exactly, and pinned equal to
    # the whole-file answer rather than merely close to it.
    assert chunked["value"].tolist() == [pytest.approx(0.0), pytest.approx(1.0),
                                         pytest.approx(0.0), pytest.approx(1.0),
                                         pytest.approx(0.0)]
    assert chunked.equals(whole)


def test_a_repeated_key_refuses_rather_than_picking_one_of_the_two(tmp_path):
    """``keep='last'`` is order-dependent, and a chunked read has no order to keep.

    Returning the first row instead would look like a working panel with a slightly
    different value on the duplicated day.
    """

    write_moneyflow(tmp_path, [("600519.SH", 20240102, 1.0),
                               ("600519.SH", 20240102, 2.0),
                               ("600519.SH", 20240103, 3.0)])

    with pytest.raises(ValueError, match="repeated"):
        gate.load_signal_panel(tmp_path, LEVEL, chunk_rows=1)


def test_an_unneeded_column_is_never_read(tmp_path):
    """``moneyflow`` carries four amount columns the panel never looks at.

    Dropping them is what keeps a 21.9 M row read inside memory, so it is asserted
    rather than left as a property of the code that happens to hold today.
    """

    write_moneyflow(tmp_path, [("600519.SH", 20240102, 7.5)])

    panel = gate.load_signal_panel(tmp_path, LEVEL)

    assert list(panel.columns) == ["symbol", "known_date", "value"]
    assert float(panel["value"].iloc[0]) == 7.5
