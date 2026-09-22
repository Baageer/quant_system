"""Point-in-time membership must react to index events without ever leaking forward."""

import pandas as pd
import pytest

from research.market_regime.index_membership import (
    INDEX_CODE_BY_POOL,
    load_index_weight,
    membership_intervals,
    membership_mask,
)

# Three month-end snapshots.  000001 is a member throughout; 000002 is added at
# the second snapshot and dropped at the third; 000003 only appears at the last.
SNAPSHOTS = {
    "20200123": ["000001.SZ"],
    "20200228": ["000001.SZ", "000002.SZ"],
    "20200331": ["000001.SZ", "000003.SZ"],
}


@pytest.fixture
def membership_dir(tmp_path):
    rows = [
        {"index_code": "000300.SH", "con_code": code, "trade_date": date, "weight": 1.0}
        for date, codes in SNAPSHOTS.items()
        for code in codes
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "000300.SH.csv", index=False, encoding="utf-8-sig")
    return tmp_path


def probe_membership(membership_dir, dates, symbols, **kwargs):
    frame = pd.DataFrame(
        [(d, s) for d in pd.to_datetime(dates) for s in symbols], columns=["date", "symbol"]
    )
    mask = membership_mask(frame, "hs300", membership_dir=membership_dir, **kwargs)
    return frame.assign(is_member=mask.to_numpy())


def test_load_normalizes_symbol_to_six_digits(membership_dir):
    weights = load_index_weight("000300.SH", membership_dir)
    assert set(weights["symbol"]) == {"000001", "000002", "000003"}
    assert weights["trade_date"].max() == pd.Timestamp("2020-03-31")


def test_member_only_from_its_first_snapshot_onward(membership_dir):
    frame = probe_membership(membership_dir, ["2020-01-22", "2020-01-23"], ["000001"])
    assert frame["is_member"].tolist() == [False, True]


def test_added_and_dropped_names_follow_the_snapshots(membership_dir):
    frame = probe_membership(
        membership_dir, ["2020-02-27", "2020-02-28", "2020-03-30", "2020-03-31"],
        ["000002", "000003"],
    )
    by_symbol = frame.groupby("symbol")["is_member"].apply(list).to_dict()
    # 000002 joins at the 02-28 snapshot and leaves at the 03-31 snapshot
    assert by_symbol["000002"] == [False, True, True, False]
    # 000003 is absent until the final snapshot
    assert by_symbol["000003"] == [False, False, False, True]


def test_never_a_member_symbol_is_always_false(membership_dir):
    frame = probe_membership(membership_dir, ["2020-01-23", "2020-03-31"], ["999999"])
    assert not frame["is_member"].any()


def test_dates_before_the_first_snapshot_are_not_members(membership_dir):
    frame = probe_membership(membership_dir, ["2019-12-31"], ["000001"])
    assert not frame["is_member"].any()


def test_lag_days_pushes_the_snapshot_back(membership_dir):
    # the 02-28 snapshot only becomes usable once a 7-day publication lag has passed
    frame = probe_membership(
        membership_dir, ["2020-02-28", "2020-03-05", "2020-03-06"], ["000002"], lag_days=7
    )
    assert frame["is_member"].tolist() == [False, False, True]


def test_intervals_split_a_re_entry_into_two_spells(membership_dir):
    weights = load_index_weight("000300.SH", membership_dir)
    # 000002 is present only at the middle snapshot -> one spell
    intervals = membership_intervals("000300.SH", membership_dir)
    spells = intervals[intervals["symbol"] == "000002"]
    assert len(spells) == 1
    assert spells.iloc[0]["start_date"] == pd.Timestamp("2020-02-28")
    assert spells.iloc[0]["end_date"] == pd.Timestamp("2020-02-28")
    assert len(weights) == 5


def test_unknown_pool_raises(membership_dir):
    frame = pd.DataFrame({"date": [pd.Timestamp("2020-02-28")], "symbol": ["000001"]})
    with pytest.raises(KeyError):
        membership_mask(frame, "not_a_pool", membership_dir=membership_dir)


def test_missing_cache_raises_with_a_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_tushare_index_weight"):
        load_index_weight("000300.SH", tmp_path)


def test_pool_constant_covers_the_research_chain():
    assert INDEX_CODE_BY_POOL == {"hs300": "000300.SH", "zz500": "000905.SH", "zz1000": "000852.SH"}
