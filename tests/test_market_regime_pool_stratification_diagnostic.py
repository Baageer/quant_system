import numpy as np
import pandas as pd

from research.market_regime.pool_stratification_diagnostic import (
    assign_buckets,
    bucket_membership,
    cross_sectional_by_bucket,
    selection_mix,
)
from research.market_regime.tradability_ranking_evaluation import select_top_k
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    outcome_quality,
)


DATES = ("2022-07-20", "2022-07-21")


def make_scores(symbols, dates=DATES, scores=None):
    rows = []
    for date in dates:
        for index, symbol in enumerate(symbols):
            rows.append({"date": pd.Timestamp(date), "symbol": symbol,
                         SCORE_COLUMN: float(index) if scores is None else float(scores[index]),
                         "future_excess_return_10d": float(index) / 100.0,
                         "future_mae_10d": -0.01})
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def test_assign_buckets_splits_each_date_into_equal_terciles():
    liquidity = pd.DataFrame({
        "date": pd.Timestamp("2022-07-20"),
        "symbol": list("ABCDEFGHI"),
        "liquidity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    })

    buckets = assign_buckets(liquidity).set_index("symbol")["bucket"]

    assert list(buckets.loc[list("ABC")]) == ["small"] * 3
    assert list(buckets.loc[list("DEF")]) == ["mid"] * 3
    assert list(buckets.loc[list("GHI")]) == ["large"] * 3


def test_assign_buckets_keeps_each_date_balanced_independently():
    liquidity = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-20"] * 6 + ["2022-07-21"] * 6),
        "symbol": list("ABCDEF") * 2,
        "liquidity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
    })

    buckets = assign_buckets(liquidity)

    for _date, block in buckets.groupby("date"):
        assert sorted(block["bucket"]) == ["large", "large", "mid", "mid", "small", "small"]


def test_bucket_membership_aligns_to_the_scores_rows_not_a_merge_index():
    """The regression guard: a merge would hand back a fresh RangeIndex here."""
    scores = make_scores(["A", "B", "C", "D"])
    # Buckets listed in a different order, covering only two of the four symbols.
    buckets = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-21"] * 2 + ["2022-07-20"] * 2),
        "symbol": ["D", "A", "D", "A"],
        "bucket": ["large", "small", "large", "small"],
    })

    mask = bucket_membership(scores, buckets, "small")

    expected = ((scores["date"] == pd.Timestamp("2022-07-20")) & (scores["symbol"] == "A")) | \
               ((scores["date"] == pd.Timestamp("2022-07-21")) & (scores["symbol"] == "A"))
    assert mask.tolist() == expected.tolist()
    assert int(mask.sum()) == 2


def test_bucket_membership_is_empty_for_a_bucket_with_no_rows():
    scores = make_scores(["A", "B"])
    buckets = pd.DataFrame({"date": pd.Timestamp("2022-07-20"), "symbol": ["A", "B"],
                            "bucket": ["small", "small"]})

    assert not bucket_membership(scores, buckets, "large").any()


def make_balanced_buckets():
    """Nine symbols per date, three per bucket, for every date the scores cover."""

    liquidity = pd.DataFrame({
        "date": pd.to_datetime([date for date in DATES for _ in range(9)]),
        "symbol": list("ABCDEFGHI") * len(DATES),
        "liquidity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * len(DATES),
    })
    return assign_buckets(liquidity)


def test_selection_mix_shares_sum_to_one_and_tilt_is_flat_when_uniform():
    scores = make_scores(list("ABCDEFGHI"))
    buckets = make_balanced_buckets()

    mix = selection_mix(scores, buckets, 1.0)  # select everything

    assert mix["universe_share"].sum() == 1.0
    assert mix["selected_share"].sum() == 1.0
    assert np.allclose(mix["tilt_vs_universe"], 1.0)
    assert mix["universe_rows"].tolist() == [6, 6, 6]  # 3 names x 2 dates


def test_selection_mix_detects_a_tilt_toward_one_bucket():
    # Only the three lowest-liquidity names can ever be selected.
    scores = make_scores(list("ABCDEFGHI"), scores=[9, 8, 7, 0, 0, 0, 0, 0, 0])
    buckets = make_balanced_buckets()

    mix = selection_mix(scores, buckets, 1 / 3).set_index("bucket")

    assert np.isclose(mix.loc["small", "selected_share"], 1.0)
    assert np.isclose(mix.loc["large", "selected_share"], 0.0)
    assert mix.loc["small", "tilt_vs_universe"] > 2.0


def test_cross_sectional_table_has_three_buckets_plus_the_full_pool():
    scores = make_scores(list("ABCDEFGHI"))
    buckets = make_balanced_buckets()

    table = cross_sectional_by_bucket(scores, buckets, horizon=10, top_k=0.34)

    assert table["bucket"].tolist() == ["small", "mid", "large", "ALL"]
    full = scores.sort_values(["date", "symbol"]).reset_index(drop=True)
    expected = outcome_quality(full, select_top_k(full, 0.34, SCORE_COLUMN), 10)
    assert np.isclose(table.iloc[-1]["selected_excess_over_universe"],
                      expected["selected_excess_over_universe"])
