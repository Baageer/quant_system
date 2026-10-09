"""E5 step 0: how concentrated is the production portfolio *already*, by industry?

E5 asks whether the池-relative alpha is really just industry rotation, and its
first action is written into the pre-registration as a **cheap filter**:

    先量现状的行业集中度。若最大行业权重 < 20%，直接判「无可中性化的集中度」，
    关闭 E5，省下全部成本。

This module is that measurement and nothing else.  It runs **no candidate**, fits
**no model**, and changes **no parameter**: it takes the production chain as it
stands and reports what it holds.

Pre-registration (written before any number below was produced)
---------------------------------------------------------------

* **Object measured: the held portfolio**, read from the engine's own
  ``positions`` at each close and valued at the same forward-filled ``close`` the
  engine values with.  *Not* the selected basket -- for 中证1000 the basket is
  ``ceil(1000 * 0.1) = 100`` names under a 20-name cap, so the basket is five
  times wider than anything ever held and measuring it would answer a question
  nobody asked.
* **Windows: validation and test, both reported**, and the trigger is read on
  **both**.  A concentration statistic is descriptive, not a performance claim,
  so reading test here is not a selection -- but the trigger is stated as
  "close only if *neither* window shows it", which is the conservative direction
  for closing.
* **Trigger, operational form** (the plans doc gives the number, this fixes the
  comparator): **close E5 iff the maximum daily industry weight is below 20% in
  both pools and in both windows.**  Any value at or above 20% keeps E5 alive to
  E5b.  Read against the bounds from the coverage bullet, the rule becomes
  three-valued and each arm is unambiguous: upper bound below 20% -> **close**;
  lower bound at or above 20% -> **keep alive**; lower below and upper at or
  above -> **indeterminate**, and that is reported as such rather than resolved
  by picking an end.
* **Coverage gate, checked first**: ``stock_basic.industry`` is a *current*
  snapshot with blanks, so a pool whose blank share is above 2% is flagged as
  unmeasurable this way.  Measured before writing this module on the *pool*:
  HS300 2/253 = 0.8%, ZZ1000 0/488 = 0.0% -- **passed**, which is why the
  measurement was authorised at all.  Names with no industry are reported as a
  separate ``UNKNOWN`` bucket rather than dropped, because dropping them would
  silently shrink the denominator and *raise* every weight.

  **Amendment, made after the gate fired on the held book (disclosed, not
  hidden).**  On the *held* names the same gate trips: HS300 reaches 11.1%
  (validation) / 14.8% (test) of held names unlabelled, ZZ1000 0.0%.  Its entire
  cause is two delisted names -- 600837 海通证券 (delisted 2025-03-04) and
  601989 中国重工 (delisted 2025-09-05) -- both ``list_status='D'`` with a blank
  industry in the snapshot; only 6 of 339 delisted rows carry one, the local
  fetch already requests ``industry`` for ``list_status="L,D,P"``, and
  ``bak_basic`` (the point-in-time route) is permission-denied (code 40203).  The
  gate was written to refuse "a number computed over whatever happened to be
  labelable", and for this statistic that refusal is unnecessary, for a reason
  strong enough to state as a bound rather than a caveat: **an unlabelled name
  can only ever push the maximum industry weight *up*.**
  :func:`industry_of` puts it in its own bucket, so adding it to any real
  industry can raise that industry or leave it alone, and it can never lower the
  largest one.  Therefore the largest *labelled* industry weight is a **lower
  bound** on the true maximum, and labelled + unknown weight is an **upper
  bound**; a verdict read on the lower bound holds under *every* assignment of
  the missing labels.  So the run proceeds, keeps ``UNKNOWN`` in the frame for
  the primary metric as originally pre-registered, and reads the trigger on the
  bounds.  The 2% cap is *not* raised -- ``coverage_gate_passed`` is recorded
  false wherever it trips, and the verdict stays three-valued so that "the bound
  does not settle it" remains a possible outcome.
* **Two references, because "20%" is unreadable on its own.**  An absolute
  maximum industry weight means nothing without knowing what an unselected
  portfolio of the same pool would score: a pool that is itself 25% banks makes a
  20% bank holding a *tilt away*.  So the module also reports (a) the
  **same-date universe**, equal-weighted, and (b) a **like-for-like random
  basket** -- 200 within-date permutations of the score, same width, same pool.
  The permutation is *within date*, not global, because the statistic is a
  within-date cross-sectional property; this is deliberately not the global
  permutation ``random_portfolio_baseline`` uses for its *return* null.
* **Self-check, and it is a hard gate**: weights are rebuilt from
  ``shares x close + cash`` and compared against the engine's own
  ``portfolio_value``.  If they do not reconcile the holdings being measured are
  not the engine's, and every number below is void.

What this module cannot say: it does not test whether industry-neutralising
would *help* -- that is E5b, and it only runs if this filter fails to close the
line.  The static-snapshot forward look is accepted here under option (a) of the
pre-registration and reported as an approximation, not as a PIT measurement.

Example::

    python -m research.experiments.industry_concentration --pools hs300 zz1000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.ranking_ablation_harness import backtest_arm
from research.experiments.random_portfolio_baseline import _score_slots
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)

HORIZON = 10
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
#: The pre-registered filter, verbatim from the plans doc: below this, there is
#: nothing for a neutralisation to remove and E5 closes without any further cost.
CONCENTRATION_TRIGGER = 0.20
#: A pool whose held names lack an industry label above this share cannot be
#: measured with a snapshot map; the run refuses rather than reporting a number
#: computed over whatever happened to be labelable.
MAX_UNKNOWN_SHARE = 0.02
PERMUTATIONS = 200
SEED = 20260920

INDUSTRY_SOURCE = "data/raw/tushare/stock_basic/stock_basic.csv"
UNKNOWN = "UNKNOWN"

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome"},
    "zz1000": {"model_id": "zz1000_pit_outcome"},
}


def pad_symbol(value: object) -> str:
    """``'000001.SZ'`` / ``'1'`` / ``1`` -> ``'000001'``.

    The published prediction files store the symbol as a bare integer, so the
    leading zeros are gone by the time it is read back; ``stock_basic`` stores
    the same code zero-padded.  Joining on the raw values drops every Shenzhen
    name whose code starts with a zero -- which is most of them -- and produces a
    coverage number that looks like a data problem rather than a join bug.
    """

    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6)


def load_industry_map(path: Path, column: str = "industry") -> Dict[str, str]:
    """Zero-padded symbol -> industry, with blanks left out on purpose.

    A blank is *missing information*, not an industry, so it must not be folded
    into a bucket at load time -- :func:`industry_of` turns it into ``UNKNOWN`` so
    the share of unlabelled names stays visible and countable.
    """

    frame = pd.read_csv(path, dtype=str)
    missing = {"symbol", column}.difference(frame.columns)
    if missing:
        raise ValueError("{} is missing {}".format(path, sorted(missing)))
    labels = frame[column].fillna("").str.strip()
    labels = labels.where(~labels.isin(["", "nan", "None"]), "")
    return {pad_symbol(symbol): label
            for symbol, label in zip(frame["symbol"], labels) if label}


def industry_of(symbol: object, mapping: Mapping[str, str]) -> str:
    return mapping.get(pad_symbol(symbol), UNKNOWN)


def concentration_bounds(weights: pd.DataFrame) -> pd.DataFrame:
    """The largest industry weight bracketed by what the missing labels could do.

    ``lower`` is the largest weight over *labelled* industries only, and ``upper``
    adds the ``UNKNOWN`` bucket to it.  The true maximum over any assignment of
    the unlabelled names lies in ``[lower, upper]``, and both ends are attainable
    -- all of them in one fresh industry gives ``lower``, all of them in the
    largest labelled industry gives ``upper`` -- so this is a tight bracket and
    not merely a safe one.

    It exists because the coverage gate can fire on the held book while the
    verdict is nevertheless settled: when ``lower`` already clears the trigger,
    the missing labels cannot rescue the portfolio, and no data repair is needed
    to say so.  Reading the trigger on ``lower`` alone would silently assume the
    unknown names are harmlessly dispersed; reading it on ``upper`` alone would
    assume the worst.
    """

    if weights.empty:
        raise ValueError("no industry weights to bracket")
    labelled = weights.drop(columns=[name for name in (UNKNOWN,) if name in weights.columns])
    if labelled.empty or labelled.shape[1] == 0:
        raise ValueError("no labelled industry columns to bracket")
    lower = labelled.max(axis=1)
    unknown = (weights[UNKNOWN] if UNKNOWN in weights.columns
               else pd.Series(0.0, index=weights.index))
    return pd.DataFrame({
        "lower": lower,
        "upper": lower + unknown,
        #: True when the bracket is tight enough that the trigger reads the same
        #: at both ends -- i.e. this day's verdict does not depend on the blanks.
        "decisive": (lower >= CONCENTRATION_TRIGGER) | ((lower + unknown) < CONCENTRATION_TRIGGER),
    })


def concentration_metrics(weights: pd.DataFrame, n_hold: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per date: the largest industry weight, its name, HHI, and effective count.

    ``effective_industries = 1 / HHI`` is reported because HHI alone is hard to
    read: it says "this many equal-sized industries would be this concentrated".

    ``n_hold`` is optional and carried through only so the caller can see that a
    concentrated day was a *narrow* day; it is never used in the statistic.
    """

    if weights.empty:
        raise ValueError("no industry weights to summarise")
    totals = weights.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(totals, 1.0, atol=1e-9):
        raise ValueError(
            "industry weights must partition to 1; worst row sums to {:.6f}".format(
                float(totals[np.argmax(np.abs(totals - 1.0))])))
    hhi = (weights ** 2).sum(axis=1)
    frame = pd.DataFrame({
        "max_industry_weight": weights.max(axis=1),
        "max_industry": weights.idxmax(axis=1),
        "hhi": hhi,
        "effective_industries": 1.0 / hhi.replace(0.0, np.nan),
        "n_industries": (weights > 0).sum(axis=1),
    })
    if n_hold is not None:
        frame["n_hold"] = n_hold.reindex(frame.index)
    return frame


def portfolio_industry_weights(
    results: pd.DataFrame,
    close_panel: pd.DataFrame,
    mapping: Mapping[str, str],
) -> tuple:
    """Daily industry weights of the held portfolio, valued the way the engine values.

    Returns ``(weights, exposure, n_hold)``.  ``weights`` is a frame indexed by
    date whose columns are industries and whose rows sum to **1** -- it is the
    composition of the *invested* book.  ``exposure`` is ``gross /
    portfolio_value`` per day, and it is returned separately rather than folded
    into the rows on purpose: this engine runs a variable-exposure path, so a
    partly-cash portfolio would otherwise have its industry weights silently
    renormalised up by the cash share, which is exactly the kind of quantity that
    gets compared across two arms without anyone noticing they were scaled
    differently.

    ``n_hold`` is the number of *names* held that day, and it is the third thing
    a maximum industry weight has to be read against.  The two are positively
    related by construction -- one name out of three is 33% whatever it is, and
    nothing with a 20-name cap can beat 5% per name -- so a concentrated day in a
    two-name book is arithmetic, not a tilt.  Reporting the weight without the
    count is how a near-empty book gets mistaken for a model that likes banks.
    """

    rows, exposure, counts = {}, {}, {}
    for date, row in results.iterrows():
        positions = row["positions"] or {}
        if not positions or date not in close_panel.index:
            continue
        prices = close_panel.loc[date]
        values = {}
        for symbol, shares in positions.items():
            price = prices.get(symbol, np.nan)
            if not np.isfinite(price):
                raise ValueError("no valuation price for held {} on {}".format(symbol, date))
            values[symbol] = float(shares) * float(price)
        gross = sum(values.values())
        if gross <= 0:
            continue
        buckets: Dict[str, float] = {}
        for symbol, value in values.items():
            name = industry_of(symbol, mapping)
            buckets[name] = buckets.get(name, 0.0) + value / gross
        rows[date] = buckets
        exposure[date] = gross / max(float(row["portfolio_value"]), 1.0)
        counts[date] = len(values)
    if not rows:
        raise ValueError("no held positions with prices in this window")
    return (pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index(),
            pd.Series(exposure).sort_index(),
            pd.Series(counts).sort_index())


def check_weights_reconcile(results: pd.DataFrame, close_panel: pd.DataFrame) -> Dict[str, float]:
    """Rebuild ``shares x close + cash`` and compare with the engine's own value.

    This is the only thing standing between "I measured the production portfolio"
    and "I measured a dict of numbers that happened to be lying around", so it is
    a gate rather than a printed diagnostic.
    """

    worst, worst_date = 0.0, None
    for date, row in results.iterrows():
        positions = row["positions"] or {}
        if not positions or date not in close_panel.index:
            continue
        prices = close_panel.loc[date]
        gross = sum(float(shares) * float(prices.get(symbol, np.nan))
                    for symbol, shares in positions.items())
        rebuilt = gross + float(row["cash"])
        relative = abs(rebuilt - float(row["portfolio_value"])) / max(abs(float(row["portfolio_value"])), 1.0)
        if relative > worst:
            worst, worst_date = relative, date
    if worst > 1e-9:
        raise ValueError(
            "rebuilt portfolio value disagrees with the engine by {:.3e} on {}; the "
            "holdings this module measured are not the engine's".format(worst, worst_date))
    return {"reconcile_max_relative_error": float(worst), "reconcile_worst_date": str(worst_date)}


def close_panel_from(market_data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The engine's valuation matrix: ``close`` per symbol, forward-filled.

    Forward-filled for the same reason the engine does it -- a suspended name
    still has a value -- and that means a held name with a gap is valued at its
    last trade, not dropped out of the denominator.
    """

    panel = pd.DataFrame({symbol: frame["close"] for symbol, frame in market_data.items()})
    return panel.sort_index().ffill()


def empty_bucket_weights(counts: Mapping[str, float], total: float) -> Dict[str, float]:
    if total <= 0:
        raise ValueError("cannot weight an empty bucket set")
    return {name: float(value) / float(total) for name, value in counts.items()}


def universe_industry_weights(
    scores: pd.DataFrame, mapping: Mapping[str, str]
) -> pd.DataFrame:
    """Equal-weighted industry weights of **every name in the pool that day**.

    This is the reference a maximum industry weight has to be read against: a
    portfolio cannot be less concentrated than the pool it draws from, and if the
    pool itself is 25% one industry then a 20% holding is a tilt *away*.
    """

    rows = {}
    for date, block in scores.groupby("date", sort=True):
        counts: Dict[str, float] = {}
        for symbol in block["symbol"]:
            name = industry_of(symbol, mapping)
            counts[name] = counts.get(name, 0.0) + 1.0
        rows[date] = empty_bucket_weights(counts, float(len(block)))
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def permuted_basket_weights(
    scores: pd.DataFrame,
    mapping: Mapping[str, str],
    top_k: float,
    held_counts: Optional[pd.Series] = None,
    draws: int = PERMUTATIONS,
    seed: int = SEED,
    max_positions: int = MAX_POSITIONS,
) -> pd.DataFrame:
    """Like-for-like random baskets: same pool, **same number of names**, shuffled within date.

    Within date rather than globally because the measured statistic is a
    within-date cross-sectional property -- a global permutation would also move
    names between dates, which is not a portfolio anyone could have held in this
    strategy.  (``random_portfolio_baseline`` permutes globally; that is the right
    convention for a *return* null and the wrong one for a *concentration* null.)

    Two corrections are baked in here, both of them repairs to earlier versions of
    this function, because both mistakes manufacture a finding:

    * **Rows are ``(draw, date)``, not one row per draw.**  The first version
      pooled every date into a single composition per draw, answering "over four
      years, what was a random basket's average industry mix".  Pooling averages
      away precisely the single-day pile-up the trigger is written on, so the
      held book looked dramatically worse than it was.
    * **The basket is ``held_counts`` wide, not ``max_positions`` wide.**
      ``max_positions`` is the *cap*; the book actually runs 3 to 20 names, and
      the narrow days are the tail of every exit path.  A fixed 20-name basket
      against a 3-name book is a width difference, not a tilt -- three names in
      three different industries is already 33%, and the earlier version scored
      that as a 100th-percentile concentration.  Passing the realised count is
      what makes the comparison say "this book, versus any other book of the same
      size from the same pool".

    ``held_counts`` is a Series of ``date -> names held``; days absent from it are
    not drawn, since the portfolio is only a portfolio on the days it holds.
    Falling back to the cap is allowed but is the weaker comparison, so callers
    that have the counts should always pass them.
    """

    rng = np.random.default_rng(seed)
    blocks = list(scores.groupby("date", sort=True))
    widths = {}
    for date, block in blocks:
        if held_counts is not None:
            if date not in held_counts.index:
                continue
            width = int(held_counts.loc[date])
        else:
            width = min(max(int(np.ceil(len(block) * float(top_k))), 1), int(max_positions))
        if width < 1:
            continue
        widths[date] = min(width, len(block))
    if not widths:
        raise ValueError("no dates to draw baskets for; held_counts did not overlap the scores")

    frames = []
    for draw in range(int(draws)):
        rows = {}
        for date, block in blocks:
            if date not in widths:
                continue
            order = rng.permutation(len(block))[:widths[date]]
            counts: Dict[str, float] = {}
            for symbol in block["symbol"].to_numpy()[order]:
                name = industry_of(symbol, mapping)
                counts[name] = counts.get(name, 0.0) + 1.0
            rows[date] = empty_bucket_weights(counts, float(widths[date]))
        frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
        frame["draw"] = int(draw)
        frames.append(frame)
    combined = pd.concat(frames)
    combined.index.name = "date"
    return combined.reset_index().set_index(["draw", "date"])


def held_percentile_vs_baskets(
    held_metrics: pd.DataFrame, basket_metrics: pd.DataFrame
) -> pd.Series:
    """Per day, where the held book's max industry weight falls in that day's null.

    This is the like-for-like comparison, and it is deliberately not "held max vs
    basket max".  The basket side is a maximum over 200 draws, so it is an extreme
    over two hundred times the samples and drifts *up* with the draw count; read
    against it, a genuinely concentrated book looks ordinary.  Ranking the held
    value inside each day's own 200 draws removes both the draw count and the day
    itself: under "the portfolio is a random basket of the same width", the
    percentile is uniform on (0, 1] with median ~0.5.
    """

    by_day = basket_metrics["max_industry_weight"].unstack("draw")
    # The basket covers every scored day; the held book covers only the days it
    # actually had positions, so held is a subset and the reindex is one-way.
    # It is checked in that direction on purpose: a two-way join would silently
    # drop the unmatched held days -- the pre-entry and fully-exited days, which
    # are exactly the ones a concentrated book would show up on -- and still
    # print a median over whatever survived.
    missing = held_metrics.index.difference(by_day.index)
    if len(missing):
        raise ValueError(
            "{} held days have no basket on the same date (first {})".format(
                len(missing), missing[0]))
    # Rank on held days only.  Reindexing the held book *up* to the basket's
    # calendar instead would put NaN on every unheld day, score each of them as
    # a zero percentile, and pull the median toward the flat days that say
    # nothing -- the portfolio is only a portfolio on the days it holds.
    by_day = by_day.loc[held_metrics.index]
    held = held_metrics["max_industry_weight"]
    ranks = np.array([
        float((row <= value).mean())
        for row, value in zip(by_day.to_numpy(), held.to_numpy())
    ])
    return pd.Series(ranks, index=by_day.index, name="basket_percentile")


#: Width bands for :func:`stratify_by_width`.  The point of the split is that the
#: held count *is* a driver of the statistic -- three names in three industries
#: is already 33% -- so a tilt that only appears on narrow days is a statement
#: about the exit path, not about industry preferences.
WIDTH_BANDS = ((1, 9), (10, 14), (15, 20))


def stratify_by_width(
    held_metrics: pd.DataFrame,
    percentile: pd.Series,
    basket_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Concentration and its null, by how many names the book held that day.

    The width-matched basket makes the percentile fair *within* a day, but the
    aggregate can still be carried by one width band: if the tilt only shows when
    the book is down to four names, the honest conclusion is about the exit path,
    not about industry selection, and the aggregate median hides that.  So the
    second dimension is reported rather than argued about.
    """

    by_day = basket_metrics["max_industry_weight"].unstack("draw").loc[held_metrics.index]
    rows = []
    for low, high in WIDTH_BANDS:
        mask = (held_metrics["n_hold"] >= low) & (held_metrics["n_hold"] <= high)
        if not mask.any():
            continue
        rows.append({
            "width_low": low, "width_high": high, "days": int(mask.sum()),
            "share_of_days": float(mask.mean()),
            "held_max_median": float(held_metrics.loc[mask, "max_industry_weight"].median()),
            "null_max_median": float(by_day.loc[mask].median().median()),
            "percentile_median": float(percentile[mask].median()),
            "share_above_null_p95": float((percentile[mask] > 0.95).mean()),
        })
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """Distribution summary of one concentration series.

    The **max** is the pre-registered comparator, but the median is reported
    beside it so a single bad day cannot be mistaken for a structural tilt.
    """

    column = metrics["max_industry_weight"].astype(float)
    summary = {
        "{}max_industry_weight_mean".format(prefix): float(column.mean()),
        "{}max_industry_weight_median".format(prefix): float(column.median()),
        "{}max_industry_weight_p95".format(prefix): float(column.quantile(0.95)),
        "{}max_industry_weight_max".format(prefix): float(column.max()),
        "{}hhi_median".format(prefix): float(metrics["hhi"].median()),
        "{}effective_industries_median".format(prefix): float(metrics["effective_industries"].median()),
        "{}days_over_trigger".format(prefix): int((column >= CONCENTRATION_TRIGGER).sum()),
        "{}days".format(prefix): int(len(column)),
    }
    if "n_hold" in metrics.columns:
        summary["{}n_hold_median".format(prefix)] = float(metrics["n_hold"].median())
    return summary


def run_pool(
    pool: str,
    split: str,
    dataset_dir: Path,
    industry_path: Path,
    draws: int = PERMUTATIONS,
    config_path: str = "config/settings.yaml",
) -> Dict[str, object]:
    spec = POOLS[pool]
    model_id = spec["model_id"]
    mapping = load_industry_map(industry_path)
    top_k = published_top_k(dataset_dir, model_id)

    # Only the *published* scores are needed: this module reproduces the incumbent
    # arm, so it reads the same score column the gates baseline against rather
    # than re-deriving it.  No model is fit and no feature panel is loaded.
    stored = load_outcome_scores(dataset_dir, model_id, split, HORIZON)
    raw_market_data, skipped = build_rank_market_data(stored, None, SCORE_COLUMN)
    if not raw_market_data:
        raise ValueError("no price data for {} {}".format(pool, split))
    slots = _score_slots(stored, raw_market_data)

    started = time.time()
    run = backtest_arm(
        stored, raw_market_data, slots, SCORE_COLUMN,
        stored["date"].min(), stored["date"].max(),
        top_k, HORIZON, INITIAL_CAPITAL, MAX_POSITIONS, config_path, label=pool)
    results = run["results"]
    print("  {} {}: {} days backtested in {:.0f}s ({} symbols skipped for prices)".format(
        pool, split, len(results), time.time() - started, len(skipped)))

    panel = close_panel_from(raw_market_data)
    reconcile = check_weights_reconcile(results, panel)
    print("  self-check: rebuilt value vs engine, max relative error {:.2e} (worst {})".format(
        reconcile["reconcile_max_relative_error"], reconcile["reconcile_worst_date"]))

    held, exposure, n_hold = portfolio_industry_weights(results, panel, mapping)
    held_metrics = concentration_metrics(held, n_hold)
    bounds = concentration_bounds(held)

    unknown_share = {}
    for date, row in results.iterrows():
        positions = row["positions"] or {}
        if not positions:
            continue
        unknown = sum(1 for symbol in positions if industry_of(symbol, mapping) == UNKNOWN)
        unknown_share[date] = unknown / float(len(positions))
    unknown = pd.Series(unknown_share).sort_index()
    worst_unknown = float(unknown.max()) if len(unknown) else 0.0
    # Reported, not fatal: see the amendment in the module docstring.  The
    # missing labels can only raise the maximum, so the labelled-only figure is a
    # lower bound and the verdict below is read on the bracket.
    coverage_ok = worst_unknown <= MAX_UNKNOWN_SHARE
    if not coverage_ok:
        print("  coverage gate FIRED: up to {:.1%} of held names carry no industry "
              "label (cap {:.0%}); reading the trigger on the bracket instead".format(
                  worst_unknown, MAX_UNKNOWN_SHARE))

    universe = concentration_metrics(universe_industry_weights(stored, mapping))
    baskets = permuted_basket_weights(stored, mapping, top_k, held_counts=n_hold, draws=draws)
    basket_metrics = concentration_metrics(baskets)
    percentile = held_percentile_vs_baskets(held_metrics, basket_metrics)
    strata = stratify_by_width(held_metrics, percentile, basket_metrics)

    held_summary = summarise(held_metrics, "held_")
    universe_summary = summarise(universe, "universe_")
    basket_summary = summarise(basket_metrics, "random_basket_")
    held_summary["held_basket_percentile_median"] = float(percentile.median())
    held_summary["held_basket_percentile_mean"] = float(percentile.mean())
    held_summary["held_share_days_above_p95"] = float((percentile > 0.95).mean())
    lower = float(bounds["lower"].max())
    upper = float(bounds["upper"].max())
    # Three-valued on purpose: "the bound does not settle it" has to stay sayable,
    # otherwise the bracket is decoration and the answer is whichever end closes
    # the line.
    if upper < CONCENTRATION_TRIGGER:
        verdict = "close"
    elif lower >= CONCENTRATION_TRIGGER:
        verdict = "keep_alive"
    else:
        verdict = "indeterminate"
    print("  held portfolio: max industry weight median {:.1%}, p95 {:.1%}, worst {:.1%} ({})".format(
        held_summary["held_max_industry_weight_median"],
        held_summary["held_max_industry_weight_p95"],
        held_summary["held_max_industry_weight_max"],
        held_metrics["max_industry"].iloc[int(np.argmax(held_metrics["max_industry_weight"].to_numpy()))]))
    print("  references: universe max {:.1%} / random basket max {:.1%} / held max {:.1%}".format(
        universe_summary["universe_max_industry_weight_max"],
        basket_summary["random_basket_max_industry_weight_max"],
        held_summary["held_max_industry_weight_max"]))
    print("  like-for-like: held vs same-day random basket -> median percentile {:.1%}, "
          "{:.1%} of days above the basket's 95th".format(
              held_summary["held_basket_percentile_median"],
              held_summary["held_share_days_above_p95"]))
    for _, band in strata.iterrows():
        print("    width {:>2}-{:<2} n={:<4} ({:>4.0%} of days)  held {:.1%} vs null {:.1%}  "
              "percentile {:.1%}  >p95 {:.1%}".format(
                  int(band["width_low"]), int(band["width_high"]), int(band["days"]),
                  band["share_of_days"], band["held_max_median"], band["null_max_median"],
                  band["percentile_median"], band["share_above_null_p95"]))
    print("  trigger bracket on held max: lower {:.1%} .. upper {:.1%} -> {} "
          "({:.1%} of days decisive)".format(
              lower, upper, verdict, float(bounds["decisive"].mean())))
    print("  held effective industries (median): {:.1f}   names held (median): {:.0f}   "
          "unknown-label share (worst day): {:.1%}".format(
              held_summary["held_effective_industries_median"],
              held_summary.get("held_n_hold_median", float("nan")), worst_unknown))
    print("  held gross exposure: mean {:.1%} median {:.1%} (industry weights are of the invested book)".format(
        float(exposure.mean()), float(exposure.median())))

    return {
        "pool": pool, "split": split, "model_id": model_id, "top_k": top_k,
        "max_positions": MAX_POSITIONS, "horizon": HORIZON,
        "concentration_trigger": CONCENTRATION_TRIGGER,
        "max_unknown_share_allowed": MAX_UNKNOWN_SHARE,
        "draws": int(draws), "seed": SEED,
        "symbols_skipped_for_prices": len(skipped),
        "coverage_gate_passed": bool(coverage_ok),
        "verdict": verdict,
        "held_max_industry_weight_lower": lower,
        "held_max_industry_weight_upper": upper,
        "held_bracket_decisive_share": float(bounds["decisive"].mean()),
        "worst_unknown_share": worst_unknown,
        **reconcile, **held_summary, **universe_summary, **basket_summary,
        "held_exposure_mean": float(exposure.mean()),
        "held_exposure_median": float(exposure.median()),
        "held_daily": held_metrics, "held_bounds_daily": bounds,
        "held_percentile_daily": percentile.to_frame(),
        "width_strata": strata,
        "universe_daily": universe, "random_basket_daily": basket_metrics,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--industry-path", default=INDUSTRY_SOURCE)
    parser.add_argument("--pools", nargs="+", default=sorted(POOLS), choices=sorted(POOLS))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--draws", type=int, default=PERMUTATIONS)
    parser.add_argument("--config-path", default="config/settings.yaml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verdicts = []
    for pool in args.pools:
        for split in args.splits:
            result = run_pool(pool, split, Path(args.dataset_dir), Path(args.industry_path),
                              draws=args.draws, config_path=args.config_path)
            prefix = "industry_concentration_{}_{}".format(pool, split)
            result["held_daily"].to_csv(output_dir / (prefix + "_held_daily.csv"))
            result["held_bounds_daily"].to_csv(output_dir / (prefix + "_held_bounds_daily.csv"))
            result["held_percentile_daily"].to_csv(output_dir / (prefix + "_percentile_daily.csv"))
            result["width_strata"].to_csv(output_dir / (prefix + "_width_strata.csv"), index=False)
            result["universe_daily"].to_csv(output_dir / (prefix + "_universe_daily.csv"))
            result["random_basket_daily"].to_csv(output_dir / (prefix + "_random_basket_daily.csv"))
            payload = {key: value for key, value in result.items()
                       if not key.endswith("_daily") and not isinstance(value, pd.DataFrame)}
            (output_dir / (prefix + "_summary.json")).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8")
            verdicts.append(result)

    print("\n=== E5 step 0 verdict (pre-registered) ===")
    for item in verdicts:
        print("  {:<8} {:<11} held max {:>6.1%} (bracket {:>6.1%}..{:>6.1%})  "
              "random basket max {:>6.1%}  universe max {:>6.1%}  -> {}".format(
                  item["pool"], item["split"],
                  item["held_max_industry_weight_max"],
                  item["held_max_industry_weight_lower"],
                  item["held_max_industry_weight_upper"],
                  item["random_basket_max_industry_weight_max"],
                  item["universe_max_industry_weight_max"],
                  item["verdict"]))
    # E5 closes only if every cell closes.  One cell keeping it alive is enough
    # to fund E5b, and any indeterminate cell blocks the close -- "we could not
    # tell" is not a reason to spend nothing, it is a reason to stay open.
    if all(item["verdict"] == "close" for item in verdicts):
        print("  every cell below {:.0%} even at the upper bound -> "
              "无可中性化的集中度，E5 关闭".format(CONCENTRATION_TRIGGER))
    elif any(item["verdict"] == "indeterminate" for item in verdicts):
        print("  at least one cell's bracket straddles {:.0%} -> E5 不可判定，"
              "不据此关闭".format(CONCENTRATION_TRIGGER))
    else:
        print("  at least one cell is at or above {:.0%} even at the lower bound -> "
              "E5 存活，做 E5b".format(CONCENTRATION_TRIGGER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
