"""Emit the picks for the latest published session: "what does the model say today".

This is the forward-looking counterpart to ``build_stock_selection_2026``, which
picks over a *finished* window.  Nothing about the model changes: ``top_k`` and
the model id still come from the published ranking manifest, and the score table
is still the one ``score_forward_2026`` wrote.  The only new thing is that it
answers for the newest session rather than a fixed one.

**A pick has no realized return.**  The label is the *next* 10 sessions of
excess return, so for the latest session it cannot exist yet -- the freshest
scorable session is by construction ``horizon`` sessions short of a label.  This
script therefore reports predictions only, and the manifest says so; any realised
number here would be a bug.  (For the same reason this is a *signal-day* list:
the production chain buys at the next session's open, which is what the
``entry_date`` field records.)

**Staleness is reported, not hidden.**  The score table can only be as fresh as
the last time the heavy chain ran.  When the score table trails the price tree,
this script still emits picks for the session it has and stamps both dates into
the manifest -- a silent fallback to an old session is the one failure that
would not look like one.  ``--refresh`` runs the chain first.

Example::

    python -m research.experiments.daily_selection_2026
    python -m research.experiments.daily_selection_2026 --refresh
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from research.experiments.forward_2026_backtest import published_execution
from research.experiments.fetch_tushare_daily_basic import call_api, read_token
from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.stock_selection import (
    HORIZON,
    POOLS,
    attach_names,
    build_picks,
    load_selection_scores,
    load_symbol_names,
    price_window,
)


FORWARD_ROOT = Path("data/raw/tushare/forward_2026")

#: Steps of the published chain, in order.  Run only under ``--refresh``: the
#: sample rebuild is ~25 minutes and re-derives the whole window, so it must not
#: fire as a side effect of asking for today's picks.
PIPELINE = (
    ("fetch", ["-m", "research.experiments.fetch_tushare_forward_2026",
               "--start-date", "2026-01-01"]),
    ("splice", ["-m", "research.experiments.splice_forward_2026_prices"]),
    ("membership", ["-m", "research.experiments.merge_forward_2026_membership"]),
    ("samples", ["-m", "research.experiments.build_forward_2026_samples",
                 "--pools", "hs300,zz1000"]),
    ("score", ["-m", "research.experiments.score_forward_2026",
               "--pools", "hs300,zz1000", "--arms", "frozen", "--require-parity"]),
)


def latest_price_session(forward_root: Path = FORWARD_ROOT) -> Optional[str]:
    """Newest session in the spliced price tree, as ``YYYY-MM-DD``.

    Read from the file *name* (``SYMBOL_first_last.csv``) rather than by opening
    a file: every symbol carries the same window, so one name settles it.
    """

    directory = Path(forward_root) / "price_history" / "raw_hfq_pct"
    ends = {path.stem.rsplit("_", 1)[-1] for path in directory.glob("*_*_*.csv")}
    if not ends:
        return None
    stamp = pd.to_datetime(max(ends), format="%Y%m%d", errors="coerce")
    return None if pd.isna(stamp) else str(stamp.date())


def next_trading_day(session: str, env_file: str = ".env") -> Optional[str]:
    """The session after ``session`` per ``trade_cal``; ``None`` if unreachable.

    Best-effort on purpose: the entry date is a label on the page, and losing it
    must not cost the picks themselves.
    """

    try:
        token = read_token(Path(env_file))
        start = (pd.Timestamp(session) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        end = (pd.Timestamp(session) + pd.Timedelta(days=30)).strftime("%Y%m%d")
        frame = call_api("trade_cal", token,
                         {"exchange": "SSE", "start_date": start, "end_date": end},
                         "cal_date,is_open", retries=3)
    except Exception:
        return None
    if frame.empty:
        return None
    open_days = frame.loc[frame["is_open"].astype(str) == "1", "cal_date"].astype(str)
    if open_days.empty:
        return None
    stamp = pd.to_datetime(open_days.min(), format="%Y%m%d", errors="coerce")
    return None if pd.isna(stamp) else str(stamp.date())


def signal_day_closes(symbols: Sequence[str], session: str, forward_root: Path) -> Dict[str, float]:
    """Signal-day close per symbol, read from the spliced tree.

    Done here rather than in the page so the page stays instant: the ZZ1000
    basket is 90 names and reading their files per render is the difference
    between a snappy tab and a spinner.
    """

    price_dirs = (str(DEFAULT_PRICE_HISTORY_DIR),
                  str(Path(forward_root) / "price_history"))
    closes: Dict[str, float] = {}
    for symbol in symbols:
        try:
            frame = price_window(symbol, price_dirs, session, 1)
        except (FileNotFoundError, ValueError):
            continue
        if len(frame):
            closes[str(symbol)] = float(frame["close"].iloc[-1])
    return closes


def run_pipeline(steps: Sequence[str], cwd: Optional[Path] = None) -> List[Dict[str, object]]:
    """Run named chain steps as subprocesses, stopping at the first failure."""

    records = []
    wanted = [step for step in PIPELINE if step[0] in steps]
    for name, command in wanted:
        print("\n=== refresh: {} ===".format(name), flush=True)
        completed = subprocess.run([sys.executable] + command, cwd=str(cwd) if cwd else None)
        records.append({"step": name, "returncode": int(completed.returncode)})
        if completed.returncode != 0:
            print("refresh stopped: {} exited {}".format(name, completed.returncode))
            break
    return records


def build_pool_selection(pool: str, dataset_dir: Path, session: str,
                         entry_date: Optional[str], forward_root: Path) -> Dict[str, object]:
    model_id = POOLS[pool]["model_id"]
    execution = published_execution(dataset_dir, model_id)
    top_k = float(execution["top_k"])

    scores = load_selection_scores(pool, "frozen", dataset_dir)
    available = sorted(scores["date"].unique())
    if pd.Timestamp(session) not in set(pd.Timestamp(d) for d in available):
        raise SystemExit(
            "score table for {} has no rows for {} (newest is {}). Re-run with --refresh."
            .format(pool, session, str(pd.Timestamp(available[-1]).date())))

    picks = build_picks(scores, top_k, dates=[pd.Timestamp(session)])
    picks = attach_names(picks, load_symbol_names())
    picks.insert(0, "pool", pool)
    picks["entry_date"] = entry_date or ""
    picks["signal_close"] = picks["symbol"].map(
        signal_day_closes(list(picks["symbol"]), session, forward_root))
    # The label is the *next* horizon sessions; it cannot exist for a pick whose
    # entry session has not traded yet. Recorded rather than left to be inferred.
    picks["label_available"] = False

    cross_section = scores.loc[scores["date"] == pd.Timestamp(session)]
    return {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "top_k_source": execution["source"],
        "session": session,
        "entry_date": entry_date,
        "basket": int(len(picks)),
        "cross_section": int(len(cross_section)),
        "picks": picks,
        "mean_predicted_excess": (float(picks["predicted_excess"].mean())
                                  if "predicted_excess" in picks.columns and len(picks) else None),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz1000")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--forward-root", default=str(FORWARD_ROOT))
    parser.add_argument("--session", default=None,
                        help="YYYY-MM-DD signal day; default = newest in the score table")
    parser.add_argument("--refresh", action="store_true",
                        help="run the published chain (fetch/splice/samples/score) first")
    parser.add_argument("--refresh-steps", default="fetch,splice,membership,samples,score")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    forward_root = Path(args.forward_root)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]

    refresh_records: List[Dict[str, object]] = []
    if args.refresh:
        refresh_records = run_pipeline(
            [s.strip() for s in args.refresh_steps.split(",") if s.strip()])
        if any(record["returncode"] != 0 for record in refresh_records):
            return 2

    # A session taken from one pool's table must exist in the others, so resolve
    # it once from the first pool and let every pool assert it.
    reference = load_selection_scores(pools[0], "frozen", dataset_dir)
    newest = pd.Timestamp(reference["date"].max())
    session = str(pd.Timestamp(args.session).date()) if args.session else str(newest.date())
    price_session = latest_price_session(forward_root)
    entry_date = next_trading_day(session, args.env_file)

    print("signal day {}  |  newest scorable session  |  entry (T+1) {}".format(
        session, entry_date or "unresolved"))
    if price_session and price_session > session:
        print("NOTE: prices run to {} but the score table stops at {}. The chain is "
              "stale -- re-run with --refresh for a current list.".format(
                  price_session, session))

    results, frames = [], []
    for pool in pools:
        entry = build_pool_selection(pool, dataset_dir, session, entry_date, forward_root)
        frames.append(entry.pop("picks"))
        results.append(entry)
        print("[{}] {} picks of {} scored names, top_k={} ({})".format(
            pool, entry["basket"], entry["cross_section"], entry["top_k"],
            entry["top_k_source"]))

    combined = pd.concat(frames, ignore_index=True)
    output = dataset_dir / "daily_selection.csv"
    combined.to_csv(output, index=False, encoding="utf-8-sig")

    manifest = {
        "session": session,
        "entry_date": entry_date,
        "latest_price_session": price_session,
        "stale": bool(price_session and price_session > session),
        "horizon": HORIZON,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label_note": ("Picks carry predictions only. The label is the next {} sessions of "
                       "excess return and does not exist yet for these picks, so any "
                       "realized number attached to them would be wrong.".format(HORIZON)),
        "caliber_note": ("Signal caliber: cross-sectional top_k by entry_score on the signal "
                         "day, equal weight. The production chain buys at the next session."),
        "refresh": refresh_records,
        "pools": results,
    }
    manifest_path = forward_root / "daily_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
    print("\n{} rows -> {}".format(len(combined), output))
    print("manifest -> {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
