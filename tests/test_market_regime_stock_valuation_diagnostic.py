import numpy as np
import pandas as pd

from research.market_regime.stock_valuation_diagnostic import (
    PAIRING_WINDOW,
    WINDOWS_MONTHS_STOCK,
    composite_own_state,
    compute_cross_section,
    compute_own_state,
    flag_undefined,
    pair_own_and_pool,
    to_month_end,
    undefined_share,
)
from research.market_regime.valuation_state_core import band_from_score, cheapness_score


def _panel(n_symbols=12, n_months=48, seed=0):
    """A synthetic pool with one deliberately loss-making symbol."""

    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    frames = []
    for index in range(n_symbols):
        symbol = "{:06d}".format(index + 1)
        pe = 10.0 + index + 2.0 * np.sin(np.arange(n_months) / 6.0) + rng.randn(n_months) * 0.3
        frames.append(pd.DataFrame({
            "date": dates,
            "symbol": symbol,
            "pe_ttm": pe,
            "pb": 1.0 + pe / 15.0,
            "ps_ttm": 2.0 + rng.randn(n_months) * 0.01,
            "pcf": pe,
        }))
    return pd.concat(frames, ignore_index=True)


def _with_scores(panel):
    own = compute_own_state(panel, windows=WINDOWS_MONTHS_STOCK)
    scores = pd.Series(np.nan, index=own.index, dtype=float)
    for metric in panel.columns.intersection(["pe_ttm", "pb", "ps_ttm", "pcf"]):
        mask = own["metric"].eq(metric)
        scores.loc[mask] = cheapness_score(metric, own.loc[mask, "percentile_own"]).to_numpy()
    own["cheapness"] = scores
    own["band_own"] = band_from_score(own["cheapness"])
    return own


def test_non_positive_multiple_is_flagged_not_ranked():
    """The single most misleading output a valuation screen can produce."""

    panel = pd.DataFrame({
        "date": pd.date_range("2018-01-31", periods=3, freq="ME"),
        "symbol": "000001",
        "pe_ttm": [12.0, -8.0, 0.0],
        "pb": [1.2, 1.1, 1.0],
    })

    flagged = flag_undefined(panel, metrics=("pe_ttm", "pb"))

    # Removed from the rank...
    assert pd.isna(flagged["pe_ttm"].iloc[1]) and pd.isna(flagged["pe_ttm"].iloc[2])
    # ...preserved for audit...
    assert flagged["pe_ttm_raw"].iloc[1] == -8.0
    # ...and counted.
    assert flagged["pe_ttm_undefined"].tolist() == [False, True, True]
    # A negative PE must not be mistaken for the cheapest observation on record.
    assert flagged["pb_undefined"].sum() == 0


def test_loss_maker_gets_no_own_percentile_band():
    """A loss-maker must not be labelled 'very cheap' by a naive percentile."""

    panel = _panel(n_symbols=6, n_months=48)
    panel.loc[panel["symbol"].eq("000001") & panel["date"].ge("2020-01-31"), "pe_ttm"] = -5.0
    flagged = flag_undefined(panel)
    own = _with_scores(flagged)

    loss_maker = own.loc[
        own["symbol"].eq("000001") & own["metric"].eq("pe_ttm") & own["window"].eq("36m")
    ].sort_values("date")
    healthy = own.loc[
        own["symbol"].eq("000002") & own["metric"].eq("pe_ttm") & own["window"].eq("36m")
    ].sort_values("date")

    assert loss_maker["undefined"].sum() > 0
    assert loss_maker.loc[loss_maker["undefined"], "percentile_own"].isna().all()
    assert loss_maker.loc[loss_maker["undefined"], "band_own"].isna().all()
    # The healthy symbol is unaffected -- the exclusion is per observation, not
    # per symbol.  48 months of history fills a 36-month window from month 36.
    assert healthy["percentile_own"].notna().sum() == len(healthy) - 35


def test_undefined_share_is_reported_not_dropped():
    panel = _panel(n_symbols=10, n_months=36)
    panel.loc[panel["symbol"].isin(["000001", "000002"]) & panel["date"].ge("2020-01-31"), "pe_ttm"] = -1.0
    flagged = flag_undefined(panel)
    own = _with_scores(flagged)

    summary = undefined_share(own)
    latest = summary.loc[summary["date"].eq(summary["date"].max()) & summary["metric"].eq("pe_ttm")]

    assert latest["n_symbols"].iloc[0] == 10
    assert latest["n_undefined"].iloc[0] == 2
    assert latest["share_undefined"].iloc[0] == 0.2


def test_window_honesty_survives_into_the_per_symbol_layer():
    """48 months of history cannot produce a 60-month percentile."""

    panel = _panel(n_symbols=4, n_months=48)
    own = _with_scores(flag_undefined(panel))

    short = own.loc[own["window"].eq("36m"), "percentile_own"]
    long = own.loc[own["window"].eq("60m"), "percentile_own"]

    assert short.notna().any()
    assert long.isna().all()  # 48 < 60, so nothing is reported
    # ...and the sample count explains why.
    assert own.loc[own["window"].eq("60m"), "n_obs"].max() == 48


def test_composite_state_vetoes_a_lonely_metric():
    """A stock measurable on one multiple is not a multi-metric verdict."""

    # Leave exactly one metric standing: two are absent, and PE is undefined
    # because the company lost money.  The edits come *before* flagging --
    # mutating the frame afterwards would leave the ``_undefined`` flags stale.
    panel = _panel(n_symbols=4, n_months=48)
    panel["ps_ttm"] = np.nan
    panel["pcf"] = np.nan
    panel["pe_ttm"] = -1.0
    own = _with_scores(flag_undefined(panel))

    strict = composite_own_state(own, min_metrics=2)
    loose = composite_own_state(own, min_metrics=1)

    # Rows before the window fills have zero voters, not one -- so the claim
    # is about the rows that have any metric at all.
    covered = loose["state_score"].notna()
    assert covered.any()
    assert (strict.loc[covered, "n_metrics"] == 1).all()
    assert (loose.loc[covered, "n_metrics"] == 1).all()
    assert strict["state_score"].isna().all()
    assert loose["state_score"].notna().any()


def test_composite_state_blends_across_metrics_with_a_voter_count():
    panel = _panel(n_symbols=4, n_months=48)
    own = _with_scores(flag_undefined(panel))

    composite = composite_own_state(own, min_metrics=2)

    assert composite["n_metrics"].max() == 4
    assert set(composite.columns) >= {"date", "symbol", "window", "state_score", "band_composite"}
    scores = composite.loc[composite["state_score"].notna(), "state_score"]
    assert scores.between(0.0, 1.0).all()


def test_month_end_grid_is_the_same_for_every_symbol():
    """Sampling density must not leak into a cross-sectional percentile."""

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-03", "2020-01-20", "2020-02-05", "2020-02-27"]),
        "pe_ttm": [1.0, 2.0, 3.0, 4.0],
    })

    monthly = to_month_end(frame)

    assert list(monthly.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    # Last observation in the month wins -- no averaging of non-month-end days.
    assert monthly["pe_ttm"].tolist() == [2.0, 4.0]
    # The real observation date travels beside the bucket label.
    assert monthly["observed_date"].tolist() == [
        pd.Timestamp("2020-01-20"), pd.Timestamp("2020-02-27")
    ]


def test_latest_snapshot_reports_the_real_observation_not_the_future_month_end():
    """An in-progress month must not be stamped with a date that has not happened."""

    # 2022-08 is incomplete: the newest row is the 12th, but the bucket ends on
    # the 31st.
    dates = list(pd.date_range("2022-01-28", periods=7, freq="ME")) + [pd.Timestamp("2022-08-12")]
    panel = pd.DataFrame({
        "date": dates,
        "symbol": "000001",
        "pe_ttm": np.arange(1.0, len(dates) + 1),
        "pb": 1.0,
    })

    monthly = to_month_end(panel)
    last = monthly.iloc[-1]

    assert monthly.index[-1] == pd.Timestamp("2022-08-31")
    assert last["observed_date"] == pd.Timestamp("2022-08-12")
    assert last["observed_date"] != monthly.index[-1]


def test_cross_section_refuses_a_thin_month():
    panel = _panel(n_symbols=5, n_months=12)

    cross = compute_cross_section(panel, min_peers=8)

    assert cross["percentile_pool"].isna().all()
    assert (cross["n_peers"] == 5).all()


def test_pairing_is_an_inner_join_on_both_measurements():
    """Half a measurement is not a cell."""

    panel = _panel(n_symbols=12, n_months=72)
    own = _with_scores(flag_undefined(panel))
    cross = compute_cross_section(flag_undefined(panel), min_peers=8)

    paired = pair_own_and_pool(own, cross, window=PAIRING_WINDOW)

    assert len(paired) > 0
    assert paired["percentile_own"].notna().all()
    assert paired["percentile_pool"].notna().all()
    assert paired["window"].eq("60m").all()
    # Every label is one of the nine cells the core defines, and nothing else.
    expected = {
        "{}_self_{}_peers".format(own_bucket, peer_bucket)
        for own_bucket in ("cheap", "fair", "expensive")
        for peer_bucket in ("cheap", "fair", "expensive")
    }
    assert set(paired["cell"].dropna().unique()) <= expected


def _open_month_panel(n_symbols=8, n_months=48, seed=0):
    """A month-end panel whose final bucket is an incomplete month.

    Built through the same ``to_month_end`` the pipeline uses, so the fixture
    cannot drift from the real grid: the last bucket is labelled by its future
    month end while its newest real observation is the 12th.
    """

    panel = _panel(n_symbols=n_symbols, n_months=n_months, seed=seed)
    last = panel["date"].max()
    open_month = (last + pd.offsets.MonthEnd(1)).replace(day=12)
    frames = []
    for symbol, group in panel.groupby("symbol"):
        group = pd.concat([
            group,
            group.tail(1).assign(date=open_month, pe_ttm=group["pe_ttm"].iloc[-1] + 0.1),
        ], ignore_index=True)
        monthly = to_month_end(group).reset_index().assign(symbol=str(symbol))
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True)


def test_every_layer_carries_the_real_observation_date():
    """The month-end label is a grid, not a date.  No layer may ship it alone.

    An in-progress month is bucketed by its *future* end, so a table offering
    only ``date`` tells the reader a state exists on a day that has not happened
    -- and the latest row is the one a reader acts on.
    """

    panel = _open_month_panel()
    flagged = flag_undefined(panel)
    own = _with_scores(flagged)
    cross = compute_cross_section(flagged, min_peers=4)
    composite = composite_own_state(own, min_metrics=2)

    assert "observed_date" in panel.columns
    for frame in (own, cross, composite):
        assert "observed_date" in frame.columns
    # ...and it is a real date, never the bucket label repeated.
    stamps = own.dropna(subset=["observed_date"])
    assert stamps["observed_date"].max() < stamps["date"].max()
    assert stamps["observed_date"].max().day == 12


def test_undefined_share_reports_the_observation_date():
    panel = _open_month_panel(n_symbols=8)
    panel.loc[
        panel["symbol"].isin(["000001", "000002"]) & panel["date"].ge("2020-01-31"), "pe_ttm"
    ] = -1.0
    own = _with_scores(flag_undefined(panel))

    summary = undefined_share(own)
    latest = summary.loc[summary["date"].eq(summary["date"].max())]

    assert "observed_date" in summary.columns
    assert latest["observed_date"].notna().all()
    assert (latest["observed_date"] < latest["date"]).all()


def test_own_band_points_the_same_way_as_its_percentile():
    """A stock at the bottom of its own history must not be labelled expensive.

    The numbers can all be right while every label is inverted -- the cheapness
    score is high (=cheap) where the percentile is low, so a band derived from
    the score without flipping reads as expensive.  Asserting on the column the
    output actually ships is the only way to catch that.
    """

    own = _with_scores(flag_undefined(_panel(n_symbols=6, n_months=60)))
    scored = own.dropna(subset=["percentile_own", "band_own"])
    assert len(scored) > 0

    # The cuts are inclusive on the *percentile* side, so 0.10 is very_cheap and
    # 0.90 is already expensive -- the strict comparison on the upper end is the
    # boundary, not slop.
    cheapest = scored.loc[scored["percentile_own"].le(0.10), "band_own"].unique()
    priciest = scored.loc[scored["percentile_own"].gt(0.90), "band_own"].unique()

    assert set(cheapest) == {"very_cheap"}
    assert set(priciest) == {"very_expensive"}


def test_composite_band_points_the_same_way_as_its_score():
    """Stated as an ordering, because a median across metrics rarely hits the
    extreme bands: the inverted version kept the same fake range but put the
    cheap half of the distribution in the expensive bands."""

    panel = _panel(n_symbols=6, n_months=60)
    composite = composite_own_state(_with_scores(flag_undefined(panel)), min_metrics=2)
    scored = composite.dropna(subset=["state_score", "band_composite"])

    cheap_side = scored.loc[scored["band_composite"].isin(["very_cheap", "cheap"]), "state_score"]
    expensive_side = scored.loc[
        scored["band_composite"].isin(["expensive", "very_expensive"]), "state_score"
    ]

    assert not cheap_side.empty and not expensive_side.empty
    assert cheap_side.median() > expensive_side.median()


def test_stock_windows_exclude_the_ten_year_window():
    """A window the data can never fill is noise, not a measurement."""

    assert 120 not in [window for window in WINDOWS_MONTHS_STOCK if window is not None]
    assert None in WINDOWS_MONTHS_STOCK
