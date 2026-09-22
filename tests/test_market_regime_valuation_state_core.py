import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.market_regime.index_valuation_diagnostic import (
    derive_erp,
    evaluate_anchors,
    load_anchors,
)
from research.market_regime.valuation_state_core import (
    BAND_CUTS,
    KIND_PRICE_MULTIPLE,
    KIND_RATE,
    KIND_YIELD,
    assert_no_forward_columns,
    apply_plausible_bounds,
    BAND_ORDER,
    band_from_percentile,
    band_from_score,
    cheapness_score,
    composite_state,
    cross_sectional_percentile,
    dedupe_by_date,
    own_vs_peer_cell,
    rel_own_vs_peers,
    trailing_median_gap,
    trailing_percentile,
    trailing_zscore,
    window_label,
)


class raises:
    """Minimal stand-in for ``pytest.raises`` -- pytest is not installed here."""

    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            raise AssertionError("expected {} but nothing was raised".format(self.expected.__name__))
        return issubclass(exc_type, self.expected)


def test_percentile_is_self_inclusive_matching_the_repo_definition():
    """Ties and the current value itself both count, as in feature_event_study."""

    rising = pd.Series([1.0, 2.0, 3.0, 4.0])
    falling = pd.Series([4.0, 3.0, 2.0, 1.0])

    assert trailing_percentile(rising, 4, 1).iloc[-1] == 1.0
    assert trailing_percentile(falling, 4, 1).iloc[-1] == 0.25
    # A constant series is at the top of its own range, not the middle.
    assert trailing_percentile(pd.Series([5.0, 5.0, 5.0]), 3, 1).iloc[-1] == 1.0


def test_percentile_window_is_honest_when_history_is_short():
    """An unfilled window yields NaN rather than a mislabelled long-window number."""

    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    percentile = trailing_percentile(series, 3)

    assert pd.isna(percentile.iloc[0]) and pd.isna(percentile.iloc[1])
    assert percentile.iloc[2] == 1.0 and percentile.iloc[3] == 1.0


def test_percentile_never_uses_future_values():
    """A value change at index k cannot move any percentile at index < k."""

    base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mutated = base.copy()
    mutated.iloc[4] = 999.0

    before = trailing_percentile(base, 3, 1).iloc[:4]
    after = trailing_percentile(mutated, 3, 1).iloc[:4]

    assert before.equals(after)


def test_expanding_percentile_has_its_own_floor():
    """'since inception' still refuses to report from a handful of observations."""

    short = pd.Series(np.arange(1.0, 21.0))

    # 20 observations is below the expanding window's own floor, so nothing is
    # reported -- 'since inception' must not mean 'since last Tuesday'.
    assert trailing_percentile(short, None).isna().all()

    long_enough = pd.Series(np.arange(1.0, 41.0))
    expanding = trailing_percentile(long_enough, None)
    assert expanding.iloc[:35].isna().all()
    assert expanding.iloc[-1] == 1.0

    # With a loose explicit floor it does report, and monotonically.
    loose = trailing_percentile(short, None, min_periods=2)
    assert loose.iloc[1] == 1.0


def test_band_edges_are_inclusive_and_nan_stays_nan():
    percentile = pd.Series([0.10, 0.30, 0.70, 0.90, 0.05, np.nan])

    bands = band_from_percentile(percentile)

    assert list(bands.iloc[:5]) == ["very_cheap", "cheap", "fair", "expensive", "very_cheap"]
    assert pd.isna(bands.iloc[5])


def test_custom_band_cuts_change_the_boundaries():
    percentile = pd.Series([0.25, 0.75])

    bands = band_from_percentile(percentile, cuts=(0.20, 0.40, 0.60, 0.80))

    assert list(bands) == ["cheap", "expensive"]


def test_cheapness_score_flips_price_multiples_but_not_yields():
    """The direction convention is the single easiest thing to invert silently."""

    percentile = pd.Series([0.05, 0.95])
    flipped = [0.95, 0.05]
    passed = [0.05, 0.95]

    # Compare with a tolerance: ``1 - 0.8`` is 0.19999999999999996 in binary
    # float, and the score is a rank-preserving transform so the noise is inert.
    for metric in ("pb", KIND_PRICE_MULTIPLE):
        assert np.allclose(list(cheapness_score(metric, percentile)), flipped)
    for metric in ("dv_ttm", KIND_YIELD):
        assert np.allclose(list(cheapness_score(metric, percentile)), passed)


def test_cheapness_score_rejects_the_risk_free_rate():
    """A bond yield is an input to the ERP, not a valuation state of its own."""

    with raises(ValueError):
        cheapness_score(KIND_RATE, pd.Series([0.5]))


def test_band_from_score_agrees_with_the_score_it_was_given():
    """A cheap score must never come back labelled expensive, for any kind.

    This is the composition rule that the two diagnostic layers call; getting it
    wrong inverts every label in the output while leaving all the numbers
    internally consistent, which is why it needs its own test.
    """

    percentile = pd.Series([0.05, 0.95])
    for metric in ("pb", "pe_ttm", KIND_PRICE_MULTIPLE, "dv_ttm", KIND_YIELD):
        score = cheapness_score(metric, percentile)
        bands = band_from_score(score)

        # Stated kind-agnostically on purpose: which of the two observations is
        # the cheap one depends on the kind (a 5th-percentile PE is cheap, a
        # 5th-percentile dividend yield is not), but "the higher score gets the
        # cheaper band" must hold for every kind.
        pairs = list(zip(score, bands))
        cheapest = max(pairs, key=lambda pair: pair[0])[1]
        priciest = min(pairs, key=lambda pair: pair[0])[1]
        assert cheapest == BAND_ORDER[0], metric
        assert priciest == BAND_ORDER[-1], metric
        assert set(bands) == {BAND_ORDER[0], BAND_ORDER[-1]}, metric


def test_band_from_score_treats_a_low_dividend_yield_as_expensive():
    """A yield in its lowest decile is an expensive market, not a cheap one."""

    low_yield = pd.Series([0.08])  # the yield is near its own historic low
    high_yield = pd.Series([0.92])

    assert list(band_from_score(cheapness_score("dv_ttm", low_yield))) == ["very_expensive"]
    assert list(band_from_score(cheapness_score("dv_ttm", high_yield))) == ["very_cheap"]
    # The mirror case: a price multiple at its historic low *is* very cheap.
    assert list(band_from_score(cheapness_score("pe_ttm", low_yield))) == ["very_cheap"]


def test_band_from_score_propagates_nan_and_honours_custom_cuts():
    score = pd.Series([0.5, np.nan, 0.95])

    bands = band_from_score(score, cuts=(0.20, 0.40, 0.60, 0.80))

    assert list(bands.iloc[[0, 2]]) == ["fair", "very_cheap"]
    assert pd.isna(bands.iloc[1])


def test_composite_state_uses_median_and_counts_voters():
    scores = {
        "pb": pd.Series([0.1, np.nan, 0.9, 0.5]),
        "ps_ttm": pd.Series([0.3, 0.4, 0.1, 0.5]),
        "pe_ttm": pd.Series([0.2, 0.4, np.nan, 0.5]),
    }

    state = composite_state(scores)

    assert state["n_metrics"].tolist() == [3, 2, 2, 3]
    assert state["state_score"].iloc[0] == 0.2  # median of 0.1/0.2/0.3
    assert state["state_score"].iloc[2] == 0.5  # median of 0.9/0.1


def test_composite_state_min_metrics_can_veto_a_lonely_metric():
    scores = {"pb": pd.Series([0.1]), "ps_ttm": pd.Series([np.nan])}

    state = composite_state(scores, min_metrics=2)

    assert pd.isna(state["state_score"].iloc[0])
    assert state["n_metrics"].iloc[0] == 1


def test_rel_own_vs_peers_is_zero_mean_by_construction():
    """The zero mean is algebra, not evidence of market efficiency."""

    rng = np.random.RandomState(0)
    own = pd.Series(rng.rand(500))
    peer = pd.Series(rng.rand(500))

    relative = rel_own_vs_peers(own, peer)

    assert abs(float(relative.mean())) < 0.05


def test_own_vs_peer_cell_separates_the_off_diagonal():
    own = pd.Series([0.10, 0.90, 0.50, np.nan])
    peer = pd.Series([0.90, 0.10, 0.50, 0.10])

    cells = own_vs_peer_cell(own, peer)

    assert cells.iloc[0] == "cheap_self_expensive_peers"
    assert cells.iloc[1] == "expensive_self_cheap_peers"
    assert cells.iloc[2] == "fair_self_fair_peers"
    assert pd.isna(cells.iloc[3])


def test_cross_sectional_percentile_guards_thin_groups():
    frame = pd.DataFrame({
        "date": pd.Timestamp("2026-09-30"),
        "sector": ["big"] * 10 + ["tiny"] * 3,
        "pb": list(np.arange(1.0, 11.0)) + [1.0, 2.0, 3.0],
    })

    ranks = cross_sectional_percentile(frame, "pb", ["date", "sector"], min_peers=8)

    assert ranks.iloc[:10].notna().all()
    assert ranks.iloc[10:].isna().all()


def test_forward_looking_columns_are_refused():
    """The structural guarantee that this stays a descriptive study."""

    assert_no_forward_columns(["date", "symbol", "pb_pct_60m"])

    for column in ("future_excess_return_10d", "label", "target_grade", "ret_fwd_5d"):
        with raises(ValueError):
            assert_no_forward_columns(["date", column])


def test_plausible_bounds_blank_rather_than_clip():
    """Clipping would rewrite the values a percentile ranks against."""

    frame = pd.DataFrame({"pe_ttm": [15.0, -3.0, 0.0, 99999.0, np.nan], "pb": [1.2, 1.3, 1.4, 1.5, 1.6]})

    bounded = apply_plausible_bounds(frame, {"pe_ttm": (0.1, 5000.0)})

    assert bounded["pe_ttm"].iloc[0] == 15.0
    assert bounded["pe_ttm"].iloc[1:].isna().all()
    assert bounded["pb"].equals(frame["pb"])


def test_dedupe_by_date_keeps_the_last_row_per_key():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-31", "2026-01-31", "2026-02-28"]),
        "symbol": ["000001", "000001", "000001"],
        "value": [1.0, 2.0, 3.0],
    })

    deduped = dedupe_by_date(frame, ["date", "symbol"], sort_by="date")

    assert len(deduped) == 2
    assert deduped["value"].tolist() == [2.0, 3.0]


def test_trailing_zscore_and_median_gap_are_causal_and_signed():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])

    zscore = trailing_zscore(series, 3, 2)
    gap = trailing_median_gap(series, 3, 2)

    assert pd.isna(zscore.iloc[0])
    assert zscore.iloc[-1] > 0 and gap.iloc[-1] > 0
    assert gap.iloc[-1] == 10.0 / 4.0 - 1.0


def test_window_label_uses_months_not_trading_days():
    """``_60m`` must never be mistakable for the frozen features' ``_250``."""

    assert window_label(60) == "60m"
    assert window_label(None) == "all"
    assert window_label(120) == "120m"


def test_band_cuts_are_ordered_and_cover_the_unit_interval():
    assert list(BAND_CUTS) == sorted(BAND_CUTS)
    assert all(0.0 < cut < 1.0 for cut in BAND_CUTS)


# --------------------------------------------------------------------------- #
# the index layer's as-of honesty
# --------------------------------------------------------------------------- #

def _write_anchor(dir_path, name, rows):
    frame = pd.DataFrame(rows, columns=["date", "anchor", "value"])
    frame.to_csv(Path(dir_path) / "{}_anchors.csv".format(name), index=False, encoding="utf-8-sig")


def _incomplete_month_anchor_dir():
    """An anchor whose newest month is only half published.

    Months 1-8 are complete.  September ends on the 30th but the vendor's newest
    rows are 2026-09-14 (PE) and 2026-09-05 (bond) -- both inside the September
    bucket, with the bond leg the staler of the two.
    """

    dir_path = tempfile.mkdtemp(prefix="valuation_anchor_")
    rows = []
    for month in range(1, 9):
        rows.append(("2026-{:02d}-27".format(month), "cn_yield_10y", 2.0))
        rows.append(("2026-{:02d}-28".format(month), "hs300_pe_ttm", 10.0 + month))
    rows.append(("2026-09-05", "cn_yield_10y", 2.0))
    rows.append(("2026-09-14", "hs300_pe_ttm", 20.0))
    _write_anchor(dir_path, "index", rows)
    return dir_path


def test_as_of_reports_the_real_observation_not_the_month_end_label():
    """The headline row must not claim a date the vendor has not published."""

    dir_path = _incomplete_month_anchor_dir()
    try:
        monthly, observed = load_anchors(Path(dir_path))
        panel, observed = derive_erp(monthly, observed)
        evaluation = evaluate_anchors(panel, windows=[3], observed=observed)
        row = evaluation.loc[
            evaluation["anchor"].eq("hs300_pe_ttm") & evaluation["window"].eq("3m")
        ].iloc[0]

        assert row["period_end"] == "2026-09-30"  # the bucket label
        assert row["as_of"] == "2026-09-14"       # the truth
        assert row["as_of"] != row["period_end"]
        assert row["level"] == 20.0
    finally:
        shutil.rmtree(dir_path, ignore_errors=True)


def test_erp_as_of_is_the_staler_of_its_two_legs():
    """An ERP is not knowable before both its inputs are published."""

    dir_path = _incomplete_month_anchor_dir()
    try:
        monthly, observed = load_anchors(Path(dir_path))
        panel, observed = derive_erp(monthly, observed)
        evaluation = evaluate_anchors(panel, windows=[3], observed=observed)
        row = evaluation.loc[
            evaluation["anchor"].eq("erp_hs300") & evaluation["window"].eq("3m")
        ].iloc[0]

        assert row["period_end"] == "2026-09-30"
        assert row["as_of"] == "2026-09-05"  # the bond leg, not the PE leg
        assert row["level"] == 100.0 / 20.0 - 2.0
    finally:
        shutil.rmtree(dir_path, ignore_errors=True)
