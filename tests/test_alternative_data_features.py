"""Tests for the E8 append-only feature build.

The module's whole job is to *not* disturb the frozen samples while adding two
columns to them, so most of what is pinned here is failure detection: a check that
cannot fail is not a check.  Synthetic frames only -- no real dataset is touched.
"""

import pandas as pd
import pytest

from research.experiments.alternative_data_features import (
    SIGNALS,
    attach_signals,
    load_panels,
    build_split,
    read_samples_as_text,
    render_values,
    sample_path,
    signal_column,
    verify_append_only,
)


LEVEL = SIGNALS[0]
ASOF = SIGNALS[1]
LEVEL_COLUMN = signal_column(LEVEL)
ASOF_COLUMN = signal_column(ASOF)


def write_moneyflow(signal_dir, rows):
    """rows: (ts_code, trade_date, net_mf_amount)"""

    pd.DataFrame({"ts_code": [r[0] for r in rows],
                  "trade_date": [r[1] for r in rows],
                  "net_mf_amount": [r[2] for r in rows]}).to_csv(
        signal_dir / "moneyflow.csv.gz", index=False, compression="gzip")


def write_holders(signal_dir, rows):
    """rows: (ts_code, ann_date, holder_num)"""

    pd.DataFrame({"ts_code": [r[0] for r in rows],
                  "ann_date": [r[1] for r in rows],
                  "end_date": [r[1] for r in rows],
                  "holder_num": [r[2] for r in rows]}).to_csv(
        signal_dir / "stk_holdernumber.csv.gz", index=False, compression="gzip")


def test_leading_zeros_survive_because_symbol_is_a_string_not_a_number(tmp_path):
    """``000001`` read the ordinary way becomes ``1``, and the join key is gone.

    This is not hypothetical: the round-trip was measured before the module was
    written, and it is why every read here goes through ``dtype=str``.
    """

    path = tmp_path / "s.csv"
    path.write_text("date,symbol,ret\n2024-01-02,000001,0.5\n2024-01-02,000100,0.25\n",
                    encoding="utf-8")

    assert list(read_samples_as_text(path)["symbol"]) == ["000001", "000100"]


def test_a_rendered_float_reads_back_as_the_same_float():
    values = pd.Series([0.1, -1.0 / 3.0, 1e-17, float("nan")])

    rendered = render_values(values)

    assert rendered[0] == repr(0.1)
    assert float(rendered[1]) == -1.0 / 3.0
    assert float(rendered[2]) == 1e-17
    # Missing matches the sample files' own convention, which is an empty field.
    assert rendered[3] == ""


def test_rendered_values_are_positional_not_index_aligned():
    """Assigning onto a reordered frame must follow position, not a stale index."""

    rendered = render_values(pd.Series([1.0, 2.0], index=[5, 7]))

    assert list(rendered) == [repr(1.0), repr(2.0)]


def test_the_appended_column_names_are_the_two_registered_ones():
    """The names are part of the registration, and must stay clear of the features."""

    assert [signal_column(clock) for clock in SIGNALS] == [
        "alt_moneyflow_level", "alt_stk_holdernumber_pct_change"]
    assert len({signal_column(clock) for clock in SIGNALS}) == len(SIGNALS)


def test_values_land_on_their_own_rows_after_the_join_reorders_them(tmp_path):
    """``attach_signal`` sorts, so positional assignment would misalign the file.

    The sample rows are written in descending date order on purpose: that is the
    exact order the join will rearrange.
    """

    write_moneyflow(tmp_path, [
        ("600519.SH", 20240102, 20.0),
        ("600519.SH", 20240103, 30.0),
        ("000001.SZ", 20240102, 10.0),
    ])
    raw = pd.DataFrame({"date": ["2024-01-03", "2024-01-02", "2024-01-02"],
                        "symbol": ["600519", "000001", "600519"],
                        "frozen_feature": ["a", "b", "c"]})

    out = attach_signals(raw, load_panels(tmp_path, (LEVEL,)), signals=(LEVEL,))

    assert list(out[LEVEL_COLUMN]) == [repr(30.0), repr(10.0), repr(20.0)]
    assert list(out["frozen_feature"]) == ["a", "b", "c"]


def test_the_disclosure_signal_carries_forward_without_moving_rows(tmp_path):
    """The asof branch is the one that genuinely reorders; same guarantee applies."""

    write_holders(tmp_path, [
        ("600519.SH", 20231201, 100),
        ("600519.SH", 20240101, 110),   # +10%
        ("000001.SZ", 20240101, 200),
        ("000001.SZ", 20240102, 150),   # -25%
    ])
    raw = pd.DataFrame({"date": ["2024-01-15", "2024-01-15"],
                        "symbol": ["000001", "600519"]})

    out = attach_signals(raw, load_panels(tmp_path, (ASOF,)), signals=(ASOF,))

    by_symbol = dict(zip(out["symbol"], out[ASOF_COLUMN]))
    assert float(by_symbol["600519"]) == pytest.approx(0.10)
    assert float(by_symbol["000001"]) == pytest.approx(-0.25)


def test_a_row_with_no_signal_is_left_empty_not_zero(tmp_path):
    """Zero is a value; absent is not.  A flow of zero is not the same as no flow."""

    write_moneyflow(tmp_path, [("600519.SH", 20240102, 20.0)])
    raw = pd.DataFrame({"date": ["2024-01-02", "2024-01-09"],
                        "symbol": ["600519", "600519"]})

    out = attach_signals(raw, load_panels(tmp_path, (LEVEL,)), signals=(LEVEL,))

    assert list(out[LEVEL_COLUMN]) == [repr(20.0), ""]


def make_source(tmp_path, name="src"):
    path = tmp_path / "{}.csv".format(name)
    path.write_text("date,symbol,ret\n2024-01-02,000001,0.5\n2024-01-02,000100,0.25\n",
                    encoding="utf-8")
    return path


def write_target(tmp_path, text, name="tgt"):
    path = tmp_path / "{}.csv".format(name)
    path.write_text(text, encoding="utf-8")
    return path


def test_append_only_verification_passes_on_a_faithful_append(tmp_path):
    source = make_source(tmp_path)
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                   "2024-01-02,000001,0.5,1.0\n"
                                   "2024-01-02,000100,0.25,\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is True
    assert report["new_columns"] == ["alt_moneyflow_level"]
    assert report["column_order_preserved"] is True
    assert report["differing_shared_columns"] == {}


def test_two_blank_cells_are_not_a_difference(tmp_path):
    """The regression: the check used to report every blank cell as a moved value.

    Measured, not imagined.  ``dtype=str`` turns an empty field into float ``NaN``,
    and ``NaN != NaN``, so the first real run of this check reported 1,475,014
    differing cells on one split -- every one of them a blank paired with a blank,
    and zero genuinely different values.  The file was correct; the check was not.
    """

    source = write_target(tmp_path, "date,symbol,ret\n2024-01-02,000001,\n",
                          name="blank_src")
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                    "2024-01-02,000001,,\n", name="blank_tgt")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is True
    assert report["differing_shared_columns"] == {}


def test_a_blank_read_back_is_an_empty_string_not_a_float(tmp_path):
    """The primitive under the comparison above, and the reason it works."""

    source = write_target(tmp_path, "date,symbol,ret\n2024-01-02,000001,\n",
                          name="empty_cell")

    assert read_samples_as_text(source)["ret"].iloc[0] == ""


def test_a_blank_against_a_value_is_still_a_difference(tmp_path):
    """Disabling NA inference must not have made the comparison vacuous."""

    source = write_target(tmp_path, "date,symbol,ret\n2024-01-02,000001,\n",
                          name="halfblank_src")
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                    "2024-01-02,000001,0.5,\n", name="halfblank_tgt")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["differing_shared_columns"] == {"ret": 1}


def test_an_na_token_against_a_blank_is_a_difference(tmp_path):
    """The pair a NaN-aware comparison would have accepted, and why it was not used.

    With NA inference on, the literal text ``NA`` and an empty field both arrive as
    the same float and a "treat two NaN as equal" rule calls them identical -- which
    is how a comparison stops being about the file.  As text they are two different
    files, and this check is about the bytes.
    """

    source = write_target(tmp_path, "date,symbol,ret\n2024-01-02,000001,NA\n",
                          name="na_src")
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                    "2024-01-02,000001,,\n", name="na_tgt")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["differing_shared_columns"] == {"ret": 1}


def test_append_only_verification_fails_when_a_shared_value_moved(tmp_path):
    """The control column is the entire basis of the comparison; guard it."""

    source = make_source(tmp_path)
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                   "2024-01-02,000001,0.7,1.0\n"
                                   "2024-01-02,000100,0.25,\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["differing_shared_columns"] == {"ret": 1}


def test_append_only_verification_fails_on_a_stripped_symbol(tmp_path):
    """The corruption this module was measured against, as a regression test."""

    source = make_source(tmp_path)
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                   "2024-01-02,1,0.5,1.0\n"
                                   "2024-01-02,100,0.25,\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert "symbol" in report["differing_shared_columns"]


def test_append_only_verification_fails_on_an_extra_column(tmp_path):
    """A bonus column is a second variable in what is meant to be an A/B."""

    source = make_source(tmp_path)
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level,my_idea\n"
                                   "2024-01-02,000001,0.5,1.0,x\n"
                                   "2024-01-02,000100,0.25,,y\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["unexpected_new_columns"] == ["my_idea"]


def test_append_only_verification_fails_when_a_source_column_is_dropped(tmp_path):
    source = make_source(tmp_path)
    target = write_target(tmp_path, "date,ret,alt_moneyflow_level\n"
                                   "2024-01-02,0.5,1.0\n"
                                   "2024-01-02,0.25,\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["dropped_columns"] == ["symbol"]


def test_append_only_verification_fails_when_the_column_order_is_rearranged(tmp_path):
    """Order is not cosmetic: the preprocessor's first k entries must be the old ones."""

    source = make_source(tmp_path)
    target = write_target(tmp_path, "alt_moneyflow_level,date,symbol,ret\n"
                                   "1.0,2024-01-02,000001,0.5\n"
                                   ",2024-01-02,000100,0.25\n")

    report = verify_append_only(source, target, ["alt_moneyflow_level"])

    assert report["ok"] is False
    assert report["column_order_preserved"] is False


def test_build_split_writes_a_verified_target(tmp_path):
    dataset_dir, signal_dir = tmp_path / "ds", tmp_path / "sig"
    dataset_dir.mkdir()
    signal_dir.mkdir()
    sample_path(dataset_dir, "src_id", "train").write_text(
        "date,symbol,ret\n2024-01-02,000001,0.5\n", encoding="utf-8")
    write_moneyflow(signal_dir, [("000001.SZ", 20240102, 7.5)])
    write_holders(signal_dir, [("000001.SZ", 20240101, 10),
                               ("000001.SZ", 20240102, 11)])

    report = build_split(dataset_dir, signal_dir, "src_id", "tgt_id", "train")

    assert report["ok"] is True
    written = pd.read_csv(sample_path(dataset_dir, "tgt_id", "train"), dtype=str)
    assert list(written["symbol"]) == ["000001"]          # the join key survived
    assert float(written[LEVEL_COLUMN].iloc[0]) == 7.5
    assert float(written[ASOF_COLUMN].iloc[0]) == pytest.approx(0.10)


def test_a_chunked_build_produces_the_same_file_as_a_single_pass_one(tmp_path):
    """The build streams; a chunk boundary must not move a value.

    A misalignment here would land each chunk's values one row off, which looks like
    a working file with a slightly different signal -- and the append-only check
    would still pass, because it only compares the columns that were not touched.
    """

    dataset_dir, signal_dir = tmp_path / "ds", tmp_path / "sig"
    dataset_dir.mkdir()
    signal_dir.mkdir()
    rows = 37
    dates = ["2024-01-{:02d}".format(day) for day in range(1, rows + 1)]
    pd.DataFrame({"date": dates, "symbol": ["000001"] * rows,
                  "ret": [str(index) for index in range(rows)]}).to_csv(
        sample_path(dataset_dir, "src_id", "train"), index=False)
    write_moneyflow(signal_dir, [("000001.SZ", int(date.replace("-", "")), float(index))
                                 for index, date in enumerate(dates, start=1)])

    whole = build_split(dataset_dir, signal_dir, "src_id", "one_chunk", "train",
                        signals=(LEVEL,), chunk_rows=10_000)
    split = build_split(dataset_dir, signal_dir, "src_id", "many_chunks", "train",
                        signals=(LEVEL,), chunk_rows=5)

    assert whole["ok"] is True and split["ok"] is True
    assert split["source_rows"] == rows
    assert split["source_rows"] == whole["source_rows"]
    assert pd.read_csv(sample_path(dataset_dir, "many_chunks", "train"), dtype=str).equals(
        pd.read_csv(sample_path(dataset_dir, "one_chunk", "train"), dtype=str))


def test_a_leftover_target_is_replaced_not_extended(tmp_path):
    """The build appends, so a stale target from an earlier run would be kept."""

    dataset_dir, signal_dir = tmp_path / "ds", tmp_path / "sig"
    dataset_dir.mkdir()
    signal_dir.mkdir()
    sample_path(dataset_dir, "src_id", "train").write_text(
        "date,symbol,ret\n2024-01-02,000001,0.5\n", encoding="utf-8")
    write_moneyflow(signal_dir, [("000001.SZ", 20240102, 7.5)])
    sample_path(dataset_dir, "tgt_id", "train").write_text(
        "date,symbol,ret\n2020-01-02,999999,9.9\n", encoding="utf-8")

    report = build_split(dataset_dir, signal_dir, "src_id", "tgt_id", "train",
                         signals=(LEVEL,))

    assert report["ok"] is True
    written = pd.read_csv(sample_path(dataset_dir, "tgt_id", "train"), dtype=str)
    assert list(written["symbol"]) == ["000001"]


def test_streaming_verification_still_notices_a_value_that_moved(tmp_path):
    """The chunked comparison must keep the failure detection the whole-file one had."""

    source = make_source(tmp_path, name="stream_src")
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                   "2024-01-02,000001,0.5,1.0\n"
                                   "2024-01-02,000100,0.25,\n", name="stream_tgt")

    report = verify_append_only(source, target, ["alt_moneyflow_level"], chunk_rows=1)

    assert report["ok"] is True

    write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                           "2024-01-02,000001,0.5,1.0\n"
                           "2024-01-02,000100,0.99,\n", name="stream_tgt")
    report = verify_append_only(source, target, ["alt_moneyflow_level"], chunk_rows=1)

    assert report["ok"] is False
    assert report["differing_shared_columns"] == {"ret": 1}


def test_a_target_with_extra_rows_is_not_reported_as_agreeing(tmp_path):
    """``zip`` would stop at the shorter file; the streaming compare must not."""

    source = make_source(tmp_path, name="len_src")
    target = write_target(tmp_path, "date,symbol,ret,alt_moneyflow_level\n"
                                   "2024-01-02,000001,0.5,1.0\n"
                                   "2024-01-02,000100,0.25,\n"
                                   "2024-01-02,000300,0.75,\n", name="len_tgt")

    report = verify_append_only(source, target, ["alt_moneyflow_level"], chunk_rows=10_000)

    assert report["ok"] is False
