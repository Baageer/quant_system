"""Shared measurement primitives for the descriptive valuation-state diagnostics.

This module answers "where does today's valuation sit in its own history / in
its peer group", and **nothing else**.  It deliberately contains no forward
return, no label, and no IC machinery: the predictive use of valuation was
closed by three independent studies (see
``docs/research/value_factor_regime_study_summary.md``), so a fourth attempt is
out of scope by construction rather than by discipline.  :func:`assert_no_forward_columns`
enforces that boundary on any frame leaving these diagnostics.

Two conventions are load-bearing and easy to get wrong:

* **Percentiles are self-inclusive** -- the value at ``t`` counts in its own
  window, matching ``feature_event_study._rolling_percentile``
  (``np.sum(array <= array[-1]) / len(array)``).  Excluding self would leave the
  expanding window undefined at the first observation and bias the statistic by
  1/n, concentrated on exactly the young stocks where the window already binds.
* **Direction is encoded in one place.**  A raw price multiple (PE/PB/PS/PCF)
  rising means *more* expensive, but a yield (dividend yield, earnings yield)
  rising means *cheaper* -- the mirror.  :func:`cheapness_score` is the single
  place that sign lives, so no caller has to remember it.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# Percentile cuts for the five-band state label.  Recorded in every manifest:
# an unrecorded selection rule is not a reproducible result.
BAND_CUTS = (0.10, 0.30, 0.70, 0.90)
BAND_ORDER = ("very_cheap", "cheap", "fair", "expensive", "very_expensive")

# Windows in months.  ``None`` is the expanding "since inception" window.
WINDOWS_MONTHS = (36, 60, 120, None)
# Below this, do not ship a percentile at all -- a 24-month percentile is one
# valuation swing, not a measurement.
MIN_WINDOW_MONTHS = 36
# Expanding windows need a floor of their own or a 3-month-old stock gets a
# "since inception" percentile that is pure noise.
MIN_EXPANDING_OBS = 36

# A month-end panel needs a much wider as-of tolerance than a daily one:
# month-end to month-end is <=31 days, a CNY gap pushes Jan->Feb to ~36, and one
# suspended month goes further.  ``fundamental_feature_study``'s 7D is correct
# for its daily legulegu panel and would silently blank ~85% of a monthly one.
ASOF_TOLERANCE = pd.Timedelta("45D")
FRESH_LAG_DAYS = 31

# A cross-sectional percentile over fewer names than this is not a measurement.
MIN_PEERS = 8
# Below this share of the universe carrying valuation, refuse the cross-section.
MIN_UNIVERSE_COVERAGE = 0.80

# Plausibility bounds -- a data-error filter, not an economic one.  Bounds are
# recorded in the manifest.  Deliberately NOT winsorising: winsorising changes
# the ranks a percentile is built from.
PLAUSIBLE_BOUNDS = {
    "pe_ttm": (0.1, 5000.0),
    "pe_static": (0.1, 5000.0),
    "pb": (0.01, 200.0),
    "ps_ttm": (0.01, 500.0),
    "pcf": (0.1, 5000.0),
}

# Metrics whose raw value is a price multiple.  Higher = more expensive.
PRICE_MULTIPLES = ("pe_ttm", "pe_static", "pb", "ps_ttm", "pcf")
# Metrics whose raw value is a yield.  Higher = cheaper.  The mirror.
YIELD_METRICS = ("dv_ttm", "ep_ttm")
# Metrics valid for every going concern; PE/PCF are conditional on the sign of
# their denominator and DV is valid only for payers.
UNCONDITIONAL_METRICS = ("pb", "ps_ttm")
CONDITIONAL_METRICS = ("pe_ttm", "pe_static", "pcf")

FORBIDDEN_COLUMN_PATTERN = re.compile(r"^(future_|label|target|ret_fwd|fwd_)", re.IGNORECASE)


def assert_no_forward_columns(columns: Iterable[str]) -> None:
    """Refuse to emit a frame carrying a forward-looking column.

    This is the structural guarantee that the valuation-state diagnostics stay
    descriptive.  Any join against ``trend_probability_*_tradability_samples_*``
    (which carry ``future_excess_return_{h}d``) trips this immediately, instead
    of quietly turning the module into a fourth attempt at the closed
    predictive line.
    """

    bad = [str(column) for column in columns if FORBIDDEN_COLUMN_PATTERN.match(str(column))]
    if bad:
        raise ValueError(
            "forward-looking columns are not allowed in a descriptive valuation study: {}".format(bad)
        )


# --------------------------------------------------------------------------- #
# rolling statistics -- causal by construction (trailing windows only)
# --------------------------------------------------------------------------- #

def trailing_percentile(
    values: pd.Series, window: Optional[int], min_periods: Optional[int] = None
) -> pd.Series:
    """Percentile of each observation within its own trailing window, self included.

    ``window=None`` uses an expanding window.  ``min_periods`` defaults to
    ``window`` (so an unfilled window yields NaN) rather than a loose fraction:
    with a permissive floor, "10y percentile" silently becomes "whatever history
    exists", and a 2007 reading carries a label it has not earned.  Callers that
    want a looser floor must pass it explicitly and ship the sample count
    alongside -- see :func:`attach_percentile`.

    ``method="max"`` makes ties match the repo's ``count(<= x) / n`` definition
    exactly; pandas' default average-rank would differ slightly on ties.
    """

    series = pd.to_numeric(values, errors="coerce")
    if window is None:
        floor = MIN_EXPANDING_OBS if min_periods is None else int(min_periods)
        return series.expanding(min_periods=floor).rank(pct=True, method="max")
    floor = int(window) if min_periods is None else int(min_periods)
    if floor > int(window):
        raise ValueError("min_periods {} exceeds window {}".format(floor, window))
    return series.rolling(int(window), min_periods=floor).rank(pct=True, method="max")


def trailing_zscore(values: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """Causal z-score against the trailing window.

    Shipped beside the percentile as a cross-check: if a symbol's percentile and
    z-score disagree in *rank* order, it has a fat tail or a structural break,
    which is a fact worth surfacing rather than averaging away.
    """

    series = pd.to_numeric(values, errors="coerce")
    floor = int(window) if min_periods is None else int(min_periods)
    mean = series.rolling(int(window), min_periods=floor).mean()
    std = series.rolling(int(window), min_periods=floor).std()
    return (series - mean) / std.where(std > 0)


def trailing_median_gap(values: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """``x / median(trailing window) - 1`` -- an outlier-robust level analogue."""

    series = pd.to_numeric(values, errors="coerce")
    floor = int(window) if min_periods is None else int(min_periods)
    median = series.rolling(int(window), min_periods=floor).median()
    return series / median.where(median > 0) - 1.0


def window_label(window: Optional[int]) -> str:
    """Column-safe window suffix. ``None`` -> ``all``, else ``{n}m`` (months).

    Month-based windows are suffixed ``_60m``, never ``_250``: the frozen
    predictive features spell ``_250`` for 250 *trading days*, and reusing that
    name for a 60-month window invites a false equivalence.
    """

    return "all" if window is None else "{}m".format(int(window))


def attach_percentile(
    frame: pd.DataFrame,
    column: str,
    windows: Sequence[Optional[int]] = WINDOWS_MONTHS,
    min_periods: Optional[int] = None,
    with_cross_checks: bool = True,
) -> pd.DataFrame:
    """Add trailing percentiles (and sample counts) for one raw column.

    Every percentile ships beside its own ``n_obs``: a percentile without its
    sample count is not interpretable, and this repo has already been burned by
    a metric that looked strong because it was measuring the wrong thing.
    """

    out = frame.copy()
    for window in windows:
        label = window_label(window)
        out["{}_pct_{}".format(column, label)] = trailing_percentile(out[column], window, min_periods)
        out["{}_n_obs_{}".format(column, label)] = (
            pd.to_numeric(out[column], errors="coerce").notna()
            .rolling(int(window) if window else len(out), min_periods=1).sum()
        )
        if with_cross_checks and window is not None:
            out["{}_z_{}".format(column, label)] = trailing_zscore(out[column], window, min_periods)
            out["{}_med_gap_{}".format(column, label)] = trailing_median_gap(out[column], window, min_periods)
    return out


# --------------------------------------------------------------------------- #
# state labelling
# --------------------------------------------------------------------------- #

def band_from_percentile(
    percentile: pd.Series, cuts: Sequence[float] = BAND_CUTS
) -> pd.Series:
    """Map a *low-is-cheap* ordinal onto the five valuation bands.

    Despite the name, the argument is not a raw percentile: it must already
    point the right way, i.e. **0 = cheapest, 1 = most expensive**.  For a price
    multiple a raw percentile happens to satisfy that; for a yield it is the
    mirror.  Callers should not reason about this at all -- go through
    :func:`band_from_score`, which takes the cheapness score where the direction
    convention already lives.  This function stays exposed only as the ordinal
    mapper that :func:`band_from_score` is built from.

    NaN stays NaN -- an unmeasurable state is not "fair".
    """

    values = pd.to_numeric(percentile, errors="coerce")
    c0, c1, c2, c3 = cuts
    # Four ordered cuts open four lower bands; the top band is the default.
    labels = np.select(
        [values <= c0, values <= c1, values <= c2, values <= c3],
        list(BAND_ORDER[:4]),
        default=BAND_ORDER[4],
    )
    result = pd.Series(labels, index=values.index, dtype=object)
    return result.where(values.notna(), np.nan)


def band_from_score(score: pd.Series, cuts: Sequence[float] = BAND_CUTS) -> pd.Series:
    """Map a cheapness score (1 = cheapest) onto the five bands.

    This is the entry point callers should use.  Deriving a band from a raw
    percentile instead looks harmless and is not: it reports the right answer
    for a price multiple and the *inverted* one for a yield, because a dividend
    yield in its lowest decile is an expensive market, not a cheap one.  Going
    through the score means the direction convention has exactly one carrier
    (:func:`cheapness_score`) instead of one per call site.

    The mirrored cuts are written out rather than delegated to
    :func:`band_from_percentile` on ``1 - score``.  That composition is
    mathematically identical but not numerically so: ``1 - (1 - 0.9)`` is
    0.9000000000000001, so an observation sitting exactly on a cut would land
    one band higher than the percentile route puts it -- and exact cuts are the
    common case here, since a 60-month window produces ``k / 60`` percentiles.
    """

    values = pd.to_numeric(score, errors="coerce")
    c0, c1, c2, c3 = cuts
    # First condition is the cheapest band, mirroring the ``<=`` route.
    labels = np.select(
        [values >= 1.0 - c0, values >= 1.0 - c1, values >= 1.0 - c2, values >= 1.0 - c3],
        list(BAND_ORDER[:4]),
        default=BAND_ORDER[4],
    )
    result = pd.Series(labels, index=values.index, dtype=object)
    return result.where(values.notna(), np.nan)


# Coarse anchor kinds, so the index layer (which names its columns
# ``hs300_pe_ttm`` / ``market_mv_gdp``) can use the same direction rule without
# re-deriving the sign at the call site.
KIND_PRICE_MULTIPLE = "price_multiple"
KIND_YIELD = "yield"
KIND_RATE = "rate"  # not a valuation state at all: the risk-free leg


def cheapness_score(metric: str, percentile: pd.Series) -> pd.Series:
    """Convert a self-inclusive percentile into a cheapness score in [0, 1].

    ``1`` = as cheap as this metric ever gets.  This is the single place the
    direction convention lives: a percentile of a *price multiple* has to be
    flipped, a percentile of a *yield* already points the right way.  Encoding
    it once is deliberate -- a sign error here would invert every headline
    number while leaving the tables internally consistent.

    Accepts either a metric name (``"pb"``, ``"dv_ttm"``) or a coarse kind
    (``KIND_PRICE_MULTIPLE`` / ``KIND_YIELD``).  ``KIND_RATE`` is rejected on
    purpose: a bond yield is an input to an equity risk premium, not a
    statement about whether equities are cheap.
    """

    values = pd.to_numeric(percentile, errors="coerce")
    if metric in YIELD_METRICS or metric == KIND_YIELD:
        return values
    if metric in PRICE_MULTIPLES or metric == KIND_PRICE_MULTIPLE:
        return 1.0 - values
    raise ValueError("metric {!r} has no known direction convention".format(metric))


def composite_state(
    scores: Mapping[str, pd.Series], min_metrics: int = 1
) -> pd.DataFrame:
    """Blend per-metric cheapness scores into one state, keeping the metric count.

    Uses the **median**, not the mean, so one absurd or outlier metric cannot
    drag the state -- and ``n_metrics`` travels with it so a reader can always
    see how many metrics actually voted.  The failure mode of every valuation
    dashboard is a blended score that quietly drops the loss-makers while still
    rendering a confident number.
    """

    if not scores:
        raise ValueError("no metric scores supplied")
    frame = pd.DataFrame(scores)
    count = frame.notna().sum(axis=1)
    state = frame.median(axis=1, skipna=True)
    return pd.DataFrame({
        "state_score": state.where(count >= int(min_metrics)),
        "n_metrics": count,
    })


def rel_own_vs_peers(own_pct: pd.Series, peer_pct: pd.Series) -> pd.Series:
    """Own-history percentile minus peer percentile.

    Read the result as cells, never as an average: both terms are near-uniform
    on [0, 1], so the difference is ~triangular with expectation *exactly* zero
    by construction.  A zero mean here would be algebra, not market efficiency
    -- the same trap as the "全期均值为零" finding in the closed value study.
    """

    return pd.to_numeric(own_pct, errors="coerce") - pd.to_numeric(peer_pct, errors="coerce")


def own_vs_peer_cell(own_pct: pd.Series, peer_pct: pd.Series, cut: float = 0.30) -> pd.Series:
    """Nine-cell 2D label: own cheap/fair/expensive x peers cheap/fair/expensive.

    The off-diagonal cells carry the information a single-layer percentile
    cannot: ``cheap_self_expensive_peers`` is cheap because *it* fell while its
    group did not (idiosyncratic distress), which a universe percentile alone
    would report as simply "cheap".
    """

    own = pd.to_numeric(own_pct, errors="coerce")
    peer = pd.to_numeric(peer_pct, errors="coerce")

    def bucket(series: pd.Series) -> pd.Series:
        labels = np.select(
            [series <= cut, series >= 1.0 - cut], ["cheap", "expensive"], default="fair"
        )
        return pd.Series(labels, index=series.index, dtype=object).where(series.notna(), np.nan)

    return bucket(own) + "_self_" + bucket(peer) + "_peers"


# --------------------------------------------------------------------------- #
# cross-sectional guards
# --------------------------------------------------------------------------- #

def cross_sectional_percentile(
    frame: pd.DataFrame, value_column: str, group_columns: Sequence[str],
    min_peers: int = MIN_PEERS,
) -> pd.Series:
    """Within-group self-inclusive rank of ``value_column``, NaN under ``min_peers``.

    A three-name "industry percentile" still returns a number between 0 and 1
    and means nothing; the guard makes that an explicit NaN instead.
    """

    values = pd.to_numeric(frame[value_column], errors="coerce")
    grouped = values.groupby([frame[column] for column in group_columns], dropna=True)
    count = grouped.transform("count")
    ranked = grouped.rank(pct=True, method="max")
    return ranked.where(count >= int(min_peers))


def dedupe_by_date(frame: pd.DataFrame, keys: Sequence[str], sort_by: Optional[str] = None) -> pd.DataFrame:
    """Drop duplicate ``keys`` keeping the last row, oldest-first.

    Mirrors ``load_valuation_history``'s ``keep="last"`` rule so an incremental
    refresh that appends newer rows wins the overlap.
    """

    out = frame.copy()
    order = list(keys) + ([sort_by] if sort_by and sort_by not in keys else [])
    out = out.sort_values(order)
    return out.drop_duplicates(list(keys), keep="last").reset_index(drop=True)


def apply_plausible_bounds(
    frame: pd.DataFrame, bounds: Mapping[str, Sequence[float]] = PLAUSIBLE_BOUNDS
) -> pd.DataFrame:
    """Blank implausible values so they cannot enter a rank.

    NaN rather than clipping: clipping would rewrite the very values the
    percentile ranks against.
    """

    out = frame.copy()
    for column, (low, high) in bounds.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        out[column] = values.where(values.between(low, high))
    return out
