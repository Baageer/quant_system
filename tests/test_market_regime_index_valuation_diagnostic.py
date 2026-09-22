"""Regression guards for the index / all-market (L1) valuation layer.

Two defects shipped out of this module, and neither had a test standing where
it would have been caught:

* ``scope_summary`` computed ``min_cheapness`` / ``max_cheapness`` to derive
  ``spread`` and then dropped both on the way out, while the Streamlit figure
  builder read them -- a ``KeyError`` on every render.
* every band came from the raw percentile, which is right for a price multiple
  and inverted for a yield: an all-A dividend yield sitting at the 8th
  percentile of five years was labelled "very cheap".

Both are one assertion deep, so they are asserted here rather than trusted.

These tests are zero-argument by necessity -- ``tests/run_tests.py`` calls each
test function with no arguments.
"""

import numpy as np
import pandas as pd

from research.market_regime.index_valuation_diagnostic import (
    ANCHOR_SPECS,
    KIND_RATE,
    evaluate_anchors,
    scope_summary,
)


# Columns ``build_scope_summary_figure`` in app_streamlit.py reads.  The list is
# duplicated deliberately: importing the app would pull in streamlit, and a test
# that cannot run without a UI dependency is a test that stops running.
APP_SUMMARY_COLUMNS = (
    "scope", "window", "n_anchors", "median_cheapness",
    "min_cheapness", "max_cheapness", "spread", "bands",
)


def _panel(n_months=180, seed=0):
    """A month-end panel carrying one anchor of each kind the layer bands.

    The shape matters more than the values: a price multiple a fifth of the way
    down its own range and a yield a fifth of the way down its range must come
    out with opposite labels, so both live in every panel built here.
    """

    rng = np.random.RandomState(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    # Both series rise, so the latest observation is the highest on record --
    # and "highest on record" means the opposite thing for the two kinds: the
    # multiple has never been more expensive, the yield has never been cheaper.
    trend = np.linspace(1.0, 2.0, n_months)
    return pd.DataFrame({
        "hs300_pe_ttm": 10.0 + 4.0 * trend + rng.randn(n_months) * 0.01,
        "hs300_pb": 1.0 + 0.5 * trend,
        "market_dividend_yield": 2.0 + 1.0 * trend,
        "cn_yield_10y": 2.5 + 0.1 * rng.randn(n_months),
    }, index=dates)


def test_scope_summary_exports_the_columns_its_consumers_read():
    """The spread is derived from two endpoints; both must leave the function."""

    table = evaluate_anchors(_panel(), windows=(60, None))
    summary = scope_summary(table, window="60m")

    assert not summary.empty
    assert set(APP_SUMMARY_COLUMNS) <= set(summary.columns)
    # The endpoints are not decoration -- they are what ``spread`` is made of.
    assert np.allclose(
        summary["spread"],
        summary["max_cheapness"] - summary["min_cheapness"],
    )
    assert (summary["min_cheapness"] <= summary["median_cheapness"]).all()
    assert (summary["median_cheapness"] <= summary["max_cheapness"]).all()


def test_scope_summary_empty_frame_carries_the_same_columns():
    """A window with nothing in it must not produce a differently-shaped frame.

    The bug above hid in the populated branch, but the empty branch is the one
    a caller hits first when a vendor throttles an endpoint -- and it is the
    easier of the two to leave stale.
    """

    table = evaluate_anchors(_panel(), windows=(60,))
    empty = scope_summary(table, window="120m")

    assert empty.empty
    assert set(APP_SUMMARY_COLUMNS) <= set(empty.columns)


def test_a_yield_and_a_multiple_at_the_same_percentile_get_opposite_bands():
    """The band follows the score, not the percentile.

    Both anchors below sit at percentile ~1.0 of their own history.  For the
    price multiple that is the most expensive observation on record; for the
    yield it is the cheapest.  A band read off the percentile puts the same
    label on both.
    """

    table = evaluate_anchors(_panel(), windows=(None,))
    latest = table.loc[table["window"].eq("all")].set_index("anchor")

    multiple = latest.loc["hs300_pe_ttm"]
    dividend = latest.loc["market_dividend_yield"]

    assert multiple["percentile"] > 0.99 and dividend["percentile"] > 0.99
    assert multiple["band"] == "very_expensive"
    assert dividend["band"] == "very_cheap"
    # ...and the scores are the mirror image of each other, not the same number.
    assert np.isclose(multiple["cheapness"], 1.0 - multiple["percentile"], atol=1e-9)
    assert np.isclose(dividend["cheapness"], dividend["percentile"], atol=1e-9)


def test_cheapness_never_equals_the_raw_level_for_a_multiple():
    """A guard against the cheapest possible mix-up.

    ``cheapness_score`` maps a *percentile*; handing it a level (12.75 rather
    than 0.77) produces a number in no valid range and a band that is always
    the same one.  The scores must stay inside [0, 1] and must not track the
    level.
    """

    table = evaluate_anchors(_panel(), windows=(60, 120, None))
    banded = table.loc[table["cheapness"].notna()]

    assert len(banded) > 0
    assert banded["cheapness"].between(0.0, 1.0).all()
    levels = banded["level"].to_numpy()
    if np.ptp(levels) > 0:
        assert abs(np.corrcoef(levels, banded["cheapness"].to_numpy())[0, 1]) < 1.0


def test_rate_anchors_are_never_banded():
    """The risk-free leg is carried for the ERP, not judged on its own."""

    table = evaluate_anchors(_panel(), windows=(None,))
    rates = table.loc[table["scope"].eq("rate")]

    assert not rates.empty
    assert rates["cheapness"].isna().all()
    assert rates["band"].isna().all()
    # Only the rate anchors are exempt: everything else that has a percentile
    # must also have a label, or a reader sees a bar with no band.
    others = table.loc[table["scope"].ne("rate") & table["percentile"].notna()]
    assert others["cheapness"].notna().all()
    assert others["band"].notna().all()


def test_a_window_longer_than_the_history_reports_nothing():
    """Window honesty at the layer where it is easiest to lie."""

    table = evaluate_anchors(_panel(n_months=40), windows=(36, 120))
    by_window = table.groupby("window")["percentile"].apply(lambda s: s.notna().any())

    assert by_window["36m"]
    assert not by_window["120m"]
    # The row still exists, carrying the sample count that explains the gap.
    long_window = table.loc[table["window"].eq("120m")]
    assert (long_window["n_obs"] <= 40).all()
    assert long_window["level"].isna().all()


def test_every_banded_anchor_declares_a_scope_and_a_default_label():
    """The scope is what keeps two populations from being averaged together."""

    for anchor, spec in ANCHOR_SPECS.items():
        assert spec["scope"], anchor
        assert spec["label"], anchor
        assert spec["kind"] in {"price_multiple", "yield", KIND_RATE}, anchor
