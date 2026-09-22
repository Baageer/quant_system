"""Present what a *frozen* outcome model picks, without re-running any research.

Two calibers are exposed, and they answer different questions:

**signal** -- the cross-sectional ``top_k`` slice by ``entry_score`` on each
rebalance date, equal-weighted.  This is the model's own ranking, untainted by
position limits, holding continuity or costs, so it isolates "does the ranking
put the right names on top".  Its return is the model's target label
(``future_excess_return_{h}d``, excess vs the pool's benchmark).

**portfolio** -- what the engine actually ends up holding (<= ``max_positions``
names), after the weekly cadence, ``min_holding_days`` and costs.  Produced by
``build_stock_selection_2026`` through the *same* chain the published backtest
uses; this module only reads its artifacts.

Nothing here selects anything: ``top_k`` and the execution config are read from
the published ranking manifest, and the cadence mirrors ``make_rank_strategy``.
Both are re-derived rather than re-chosen.

**The window ends in an open book.** A pick needs ``h`` sessions after entry
before its label exists, so realized returns stop roughly ``h`` sessions before
the price data does; later picks are still open.  ``label_available`` marks the
split, and every consumer is expected to show it rather than drop those rows.
"""

import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.data_loader import load_local_price_history_multi
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.tradability_ranking_evaluation import select_top_k
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN

DEFAULT_DATASET_DIR = Path(DATASET_DIR)

HORIZON = 10

#: ``make_rank_strategy`` strikes on the first session of each ISO week; that is
#: ``weekday=0`` in the generalised cadence of ``weekday_strike_dates``.
REBALANCE_WEEKDAY = 0

FORWARD_ROOT = Path("data/raw/tushare/forward_2026")
STOCK_BASIC_PATH = "data/raw/tushare/stock_basic/stock_basic.csv"

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome", "benchmark": "000300", "index_name": "沪深300"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "benchmark": "000852", "index_name": "中证1000"},
}


def excess_column(horizon: int = HORIZON) -> str:
    return "future_excess_return_{}d".format(horizon)


def mae_column(horizon: int = HORIZON) -> str:
    return "future_mae_{}d".format(horizon)


def rebalance_dates(sessions: Iterable) -> List[pd.Timestamp]:
    """The sessions a production weekly rebalance strikes, in order.

    The first session of each ISO week -- deliberately *not* "every Monday".
    After a holiday the week's first session is a Tuesday, and a Monday filter
    would silently drop that week.  Semantics mirror
    ``make_rank_strategy`` in ``tradability_ranking_evaluation``; the two are
    pinned together by a test against ``weekday_strike_dates(dates, 0)``.
    """

    seen: Dict[tuple, pd.Timestamp] = {}
    for value in sessions:
        stamp = pd.Timestamp(value)
        iso = stamp.isocalendar()
        key = (int(iso[0]), int(iso[1]))
        if key not in seen or stamp < seen[key]:
            seen[key] = stamp
    return sorted(seen.values())


def load_selection_scores(
    pool: str, arm: str = "frozen", dataset_dir: Optional[Path] = None,
    forward_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Read a forward score table (``forward_2026_scores_{pool}_{arm}.csv``).

    These are the tables ``score_forward_2026`` wrote and the parity check
    licensed, so they are the frozen model's own output rather than a re-score.
    """

    dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
    forward_root = Path(forward_root) if forward_root else FORWARD_ROOT
    path = dataset_dir / "forward_2026_scores_{}_{}.csv".format(pool, arm)
    if not path.exists():
        raise FileNotFoundError(
            "No score table at {}; run score_forward_2026 first".format(path))
    frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.zfill(6)
    return frame.dropna(subset=["date", "symbol", SCORE_COLUMN]).reset_index(drop=True)


def build_picks(
    scores: pd.DataFrame, top_k: float, dates: Optional[Sequence] = None,
    score_column: str = SCORE_COLUMN, horizon: int = HORIZON,
) -> pd.DataFrame:
    """The top-``top_k`` cross-sectional slice, one block per selected date.

    Selection goes through ``select_top_k`` so the boundary rule
    (``rank <= ceil(n * top_k)``) is the production one rather than a
    re-implementation that would drift at the rounding edge.
    """

    frame = scores
    if dates is not None:
        wanted = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        frame = scores.loc[scores["date"].isin(wanted)]
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "rank", "entry_score"])

    mask = select_top_k(frame, top_k, score_column)
    picks = frame.loc[mask.to_numpy()].copy()
    picks["rank"] = (picks.groupby("date")[score_column]
                     .rank(method="first", ascending=False).astype(int))
    excess, mae = excess_column(horizon), mae_column(horizon)
    keep = ["date", "symbol", "rank", score_column]
    for column in (excess, mae, "predicted_excess_return_10d", "predicted_mae_10d"):
        if column in picks.columns:
            keep.append(column)
    picks = picks.loc[:, keep]
    if excess in picks.columns:
        picks = picks.rename(columns={excess: "realized_excess", mae: "realized_mae"})
        picks["label_available"] = picks["realized_excess"].notna()
    else:
        picks["label_available"] = False
    picks = picks.rename(columns={"predicted_excess_return_10d": "predicted_excess",
                                  "predicted_mae_10d": "predicted_mae"})
    return picks.sort_values(["date", "rank"]).reset_index(drop=True)


def summarize_picks(
    picks: pd.DataFrame, scores: pd.DataFrame, horizon: int = HORIZON,
) -> pd.DataFrame:
    """Per-date: what the top-K basket realized, against the whole pool.

    ``pool_excess`` is the equal-weight realized excess of *every* scored name
    that date, so ``edge = basket - pool`` is the part attributable to the
    ranking rather than to the market.  Rows whose labels are not yet published
    carry NaN rather than being dropped -- an open book is a state, not a gap.
    """

    excess = excess_column(horizon)
    pool = (scores.dropna(subset=[excess])
            .groupby("date")[excess].agg(pool_excess="mean", pool_symbols="size"))
    graded = picks.groupby("date").agg(
        basket_excess=("realized_excess", "mean"),
        graded_symbols=("label_available", "sum"),
        basket_symbols=("symbol", "size"),
        mean_predicted=("predicted_excess", "mean"),
    )
    summary = graded.join(pool, how="left")
    summary["edge"] = summary["basket_excess"] - summary["pool_excess"]
    summary["label_available"] = summary["graded_symbols"] > 0
    return summary.reset_index()


def load_symbol_names(path: str = STOCK_BASIC_PATH) -> Dict[str, str]:
    """``symbol -> name``.  Empty when never fetched; never fatal.

    Names are presentation only, so a missing table degrades the display to
    bare codes and must not stop a selection.
    """

    table = Path(path)
    if not table.exists():
        return {}
    frame = pd.read_csv(table, dtype={"symbol": str}, low_memory=False)
    if "name" not in frame.columns or "symbol" not in frame.columns:
        return {}
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.zfill(6)
    frame["name"] = frame["name"].astype(str).str.strip()
    return dict(zip(frame["symbol"], frame["name"]))


def attach_names(frame: pd.DataFrame, names: Optional[Dict[str, str]] = None,
                 symbol_col: str = "symbol") -> pd.DataFrame:
    names = load_symbol_names() if names is None else names
    out = frame.copy()
    out["name"] = out[symbol_col].astype(str).map(names).fillna(out[symbol_col].astype(str))
    return out


def draft_summary(picks: pd.DataFrame) -> pd.DataFrame:
    """Picks whose label does not exist yet -- the still-open part of the book.

    Deliberately a separate frame: mixing these into a realized-return average
    silently dilutes it, because NaN-means would quietly shrink the denominator.
    """

    graded = picks.loc[picks["label_available"]].copy()
    open_ = picks.loc[~picks["label_available"]].copy()
    return open_, graded


# --- reading the artifacts --------------------------------------------------
#
# ``build_stock_selection_2026`` owns writing these; everything below only
# reads.  Readers degrade to empty frames rather than raising, so a half-built
# output directory shows an empty panel instead of a traceback -- except
# ``load_selection_scores``, whose absence means there is nothing to show at all.


def selection_artifact_paths(output_dir, pool: str) -> Dict[str, Path]:
    root = Path(output_dir)
    return {
        "signals": root / "stock_selection_signals_{}.csv".format(pool),
        "summary": root / "stock_selection_signal_summary_{}.csv".format(pool),
        "equity": root / "stock_selection_portfolio_equity_{}.csv".format(pool),
        "trades": root / "stock_selection_portfolio_trades_{}.csv".format(pool),
        "holdings": root / "stock_selection_portfolio_holdings_{}.csv".format(pool),
    }


def load_selection_manifest(forward_root: Optional[Path] = None) -> Dict[str, object]:
    """The manifest ``build_stock_selection_2026`` wrote, or ``{}``.

    Carries the window metrics and, crucially, ``extension_check`` -- a caller
    that quotes portfolio numbers is expected to have read the status first.
    """

    root = Path(forward_root) if forward_root else FORWARD_ROOT
    path = root / "stock_selection_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def manifest_pool(manifest: Dict[str, object], pool: str) -> Dict[str, object]:
    for entry in manifest.get("pools", []) or []:
        if entry.get("pool") == pool:
            return entry
    return {}


def extension_trustworthy(entry: Dict[str, object]) -> bool:
    """Whether the portfolio numbers for this pool may be quoted at all.

    The extended book is only the published book if the overlap matched; a
    missing manifest is *not* a pass, because that is exactly the state in which
    an unaudited number would get quoted.
    """

    return (entry.get("extension_check") or {}).get("status") == "PASS"


def load_selection_artifacts(
    pool: str, output_dir=None, names: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    paths = selection_artifact_paths(output_dir or DEFAULT_DATASET_DIR, pool)
    frames: Dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if not path.exists():
            frames[key] = pd.DataFrame()
            continue
        frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].astype(str).str.strip().str.zfill(6)
            if "name" not in frame.columns:
                frame = attach_names(frame, names)
        frames[key] = frame
    if not frames["equity"].empty:
        frames["equity"] = frames["equity"].set_index("date").sort_index()
    return frames


def parse_positions(cell) -> Dict[str, int]:
    """Holdings live in the equity table as a dict literal inside a string.

    Returns ``{}`` for the empty book and for any cell that does not parse --
    a malformed position cell should blank one row, not kill the page.
    """

    if isinstance(cell, dict):
        raw = cell
    elif isinstance(cell, str) and cell.strip():
        try:
            raw = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return {}
    else:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for symbol, shares in raw.items():
        try:
            out[str(symbol).strip().zfill(6)] = int(shares)
        except (TypeError, ValueError):
            continue
    return out


def book_as_of(equity: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """The book on the last session at or before ``as_of``.

    The engine is not guaranteed to hold anything on the last row, and the
    window ends mid-week, so the caller gets the most recent non-empty book
    rather than an empty frame just because that particular day held nothing.
    """

    if equity.empty:
        return pd.DataFrame(columns=["date", "symbol", "shares"])
    frame = equity
    if as_of is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(as_of)]
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "shares"])
    for stamp, cell in frame["positions"].iloc[::-1].items():
        held = parse_positions(cell)
        if held:
            return pd.DataFrame(
                {"date": stamp, "symbol": list(held), "shares": list(held.values())})
    return pd.DataFrame(columns=["date", "symbol", "shares"])


def book_with_values(book: pd.DataFrame, closes: Dict[str, float],
                     portfolio_value: Optional[float] = None,
                     names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Attach name, market value and weight to a book.

    ``closes`` is ``symbol -> last close``; a symbol with no quote keeps its row
    with a NaN value rather than disappearing, because a holding that fell out
    of the price cache is a fact worth showing.
    """

    if book.empty:
        return book.assign(name=[], market_value=[], weight=[])
    out = attach_names(book, names)
    out["close"] = out["symbol"].map(closes)
    out["market_value"] = out["shares"] * out["close"]
    total = float(out["market_value"].sum())
    base = float(portfolio_value) if portfolio_value else total
    out["weight"] = out["market_value"] / base if base else np.nan
    return out.sort_values("market_value", ascending=False).reset_index(drop=True)


def daily_selection_paths(output_dir=None, forward_root: Optional[Path] = None) -> Dict[str, Path]:
    """Where ``daily_selection_2026`` writes today's list and its provenance."""

    return {
        "picks": Path(output_dir or DEFAULT_DATASET_DIR) / "daily_selection.csv",
        "manifest": (Path(forward_root) if forward_root else FORWARD_ROOT)
        / "daily_selection_manifest.json",
    }


def load_daily_selection(output_dir=None, forward_root: Optional[Path] = None,
                         names: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Today's picks plus the manifest that says how fresh they are.

    Returns ``{"picks": DataFrame, "manifest": dict}``; both degrade to an empty
    frame / dict when the generator has never run, so the page can say "not
    generated yet" instead of raising.
    """

    paths = daily_selection_paths(output_dir, forward_root)
    picks = pd.DataFrame()
    if paths["picks"].exists():
        picks = pd.read_csv(paths["picks"], dtype={"symbol": str}, low_memory=False)
        if "date" in picks.columns:
            picks["date"] = pd.to_datetime(picks["date"], errors="coerce")
        if "symbol" in picks.columns:
            picks["symbol"] = picks["symbol"].astype(str).str.strip().str.zfill(6)
            if "name" not in picks.columns:
                picks = attach_names(picks, names)
    manifest: Dict[str, object] = {}
    if paths["manifest"].exists():
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except (ValueError, OSError):
            manifest = {}
    return {"picks": picks, "manifest": manifest}


def daily_pool_entry(manifest: Dict[str, object], pool: str) -> Dict[str, object]:
    for entry in manifest.get("pools", []) or []:
        if entry.get("pool") == pool:
            return entry
    return {}


def price_window(symbol: str, price_dirs: Sequence, end=None, sessions: int = 120,
                 adjustment: str = "raw_hfq_pct") -> pd.DataFrame:
    """OHLC for charting: the last ``sessions`` bars up to and including ``end``.

    Merges the historical cache and the forward tree in the order given, so the
    chart shows a continuous series across the 2025-12-31 splice.
    """

    frame = load_local_price_history_multi(symbol, list(price_dirs), adjustment)
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if sessions:
        frame = frame.iloc[-int(sessions):]
    return frame.reset_index()


