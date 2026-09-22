import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.fetch_index_history import (
    BENCHMARK_COLUMNS,
    merge_index_snapshots,
    normalize_index_frame,
    read_snapshot,
    write_benchmark,
)


@contextlib.contextmanager
def raises(exception, match=None):
    """Minimal stand-in for ``pytest.raises`` -- pytest is not installed here."""

    try:
        yield
    except exception as exc:
        if match is not None and match not in str(exc):
            raise AssertionError("{} not found in {!r}".format(match, str(exc)))
    else:
        raise AssertionError("{} was not raised".format(exception.__name__))


def em_frame():
    """The shape ``ak.stock_zh_index_daily_em`` returns."""

    return pd.DataFrame({
        "date": pd.to_datetime(["2014-10-17", "2014-10-20"]),
        "open": [6118.98, 6068.78],
        "close": [6058.67, 6154.52],
        "high": [6140.12, 6154.69],
        "low": [5943.59, 6068.52],
        "volume": [97125862, 79024983],
        "amount": [1.169908e11, 9.836696e10],
    })


def test_normalize_emits_the_benchmark_schema_in_order():
    frame = normalize_index_frame(em_frame(), "000852", "中证1000")

    assert list(frame.columns) == BENCHMARK_COLUMNS
    assert frame["index_code"].tolist() == ["000852", "000852"]
    assert frame["index_name"].tolist() == ["中证1000", "中证1000"]
    assert frame["date"].tolist() == ["2014-10-17", "2014-10-20"]


def test_normalize_sorts_and_drops_rows_without_a_usable_close():
    frame = em_frame()
    frame.loc[2] = [pd.Timestamp("2014-10-15"), 1.0, None, 1.0, 1.0, 1.0, 1.0]

    out = normalize_index_frame(frame, "000852", "中证1000")

    assert out["date"].tolist() == ["2014-10-17", "2014-10-20"]


def test_normalize_fills_absent_optional_columns_rather_than_inventing_them():
    frame = em_frame()[["date", "close"]]

    out = normalize_index_frame(frame, "000852", "中证1000")

    assert list(out.columns) == BENCHMARK_COLUMNS
    assert out["high"].isna().all()
    assert out["amount"].isna().all()


def test_normalize_rejects_a_frame_without_close():
    with raises(ValueError, match="date and close"):
        normalize_index_frame(em_frame()[["date", "open"]], "000852", "中证1000")


def test_normalize_rejects_non_positive_close():
    frame = em_frame()
    frame.loc[0, "close"] = -1.0

    with raises(ValueError, match="non-positive"):
        normalize_index_frame(frame, "000852", "中证1000")


def test_normalize_rejects_an_empty_frame():
    with raises(ValueError, match="empty"):
        normalize_index_frame(pd.DataFrame(), "000852", "中证1000")


def test_merge_keeps_the_last_snapshot_on_an_overlapping_date():
    older = normalize_index_frame(
        pd.DataFrame({"date": pd.to_datetime(["2018-01-02", "2018-01-03"]), "close": [100.0, 101.0]}),
        "000852", "中证1000")
    newer = normalize_index_frame(
        pd.DataFrame({"date": pd.to_datetime(["2018-01-03", "2018-01-04"]), "close": [999.0, 102.0]}),
        "000852", "中证1000")

    merged = merge_index_snapshots([older, newer])

    assert merged["date"].tolist() == ["2018-01-02", "2018-01-03", "2018-01-04"]
    assert merged.loc[merged["date"] == "2018-01-03", "close"].iloc[0] == 999.0


def test_merge_spans_a_disjoint_join_without_gaps_at_the_seam():
    early = normalize_index_frame(
        pd.DataFrame({"date": pd.to_datetime(["2017-12-29"]), "close": [100.0]}), "000852", "中证1000")
    late = normalize_index_frame(
        pd.DataFrame({"date": pd.to_datetime(["2018-01-02"]), "close": [101.0]}), "000852", "中证1000")

    merged = merge_index_snapshots([early, late])

    assert merged["date"].tolist() == ["2017-12-29", "2018-01-02"]
    assert len(merged.drop_duplicates("date")) == 2


def test_merge_rejects_snapshots_missing_benchmark_columns():
    with raises(ValueError, match="missing benchmark columns"):
        merge_index_snapshots([pd.DataFrame({"date": ["2018-01-02"], "close": [100.0]})])


def test_merge_rejects_an_all_empty_input():
    with raises(ValueError, match="no index snapshots"):
        merge_index_snapshots([pd.DataFrame(), None])


def test_write_then_read_round_trips_with_the_utf8_bom():
    """The pipeline reads with ``utf-8-sig``; a BOM-less file would still load,
    but the reader's contract is exercised here on the exact bytes we write."""

    frame = normalize_index_frame(em_frame(), "000852", "中证1000")
    path = Path(tempfile.gettempdir()) / "_idx_fetch_test.csv"
    try:
        write_benchmark(frame, path)
        assert path.read_bytes()[:3] == b"\xef\xbb\xbf"
        back = read_snapshot(path)
        assert back["date"].tolist() == frame["date"].tolist()
        assert np.allclose(back["close"], frame["close"])
    finally:
        path.unlink(missing_ok=True)


def test_read_snapshot_rejects_a_file_without_close():
    path = Path(tempfile.gettempdir()) / "_idx_fetch_bad.csv"
    try:
        pd.DataFrame({"date": ["2018-01-02"]}).to_csv(path, index=False)
        with raises(ValueError, match="date and close"):
            read_snapshot(path)
    finally:
        path.unlink(missing_ok=True)
