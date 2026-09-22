"""Is the *direction* of the value factor knowable at t?

``fundamental_feature_study`` found ``bp`` at IC +0.074 in test but -0.071 in
validation, and ``value_regime_diagnostic`` showed the flip is *not* explained
by ``market_state``: every state shares the sign inside each split.  The flip is
yearly / multi-year -- a **style** regime, not a trend regime.

So the gating question for the whole value line is not "is the factor strong"
(it is) but "is its sign forecastable from information available at t".  This
module asks exactly that, using only trailing data:

* ``trailing_ic(t, W)`` -- mean daily Rank IC of the factor over the window
  ``[t - W, t - 10]``.  The 10-day exclusion is required for causality: the
  forward excess label at day ``d`` spans ``d+1 .. d+10``, so an IC computed at
  ``d`` only becomes known at ``d + 10``.
* target -- mean daily Rank IC over ``[t + 1, t + H]``.

A usable timing signal needs ``corr(trailing, forward) > 0`` (factor momentum:
follow the recent direction) or a reliably negative one (mean reversion).  If
neither, the factor's sign is unknowable ex ante and no interaction term,
however clever, can harvest it.

Example::

    python -m research.market_regime.factor_timing_study
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.fundamental_feature_study import FEATURE_NAMES
from research.market_regime.value_regime_diagnostic import load_or_build_panel


LABEL_COLUMN = "future_excess_return_10d"
PANEL_CACHE = "output/datasets/fundamental_feature_panel_excess_hs300_v1.csv"
# The forward label spans 10 sessions, so an IC at day d is only known at d+10.
LABEL_HORIZON = 10
DEFAULT_WINDOWS = (60, 120, 250)
DEFAULT_FORWARD = 60


def daily_ic_series(frame: pd.DataFrame, feature: str, label: str, min_rows: int = 20) -> pd.Series:
    """Per-date cross-sectional Rank IC, indexed by date, NaN-clean.

    Same estimator as ``feature_study_core.daily_rank_ics`` but keeping the date
    index, which the rolling timing analysis needs.
    """

    working = pd.DataFrame({
        "date": pd.to_datetime(frame["date"]),
        "actual": pd.to_numeric(frame[label], errors="coerce"),
        "predicted": pd.to_numeric(frame[feature], errors="coerce"),
    }).dropna()
    if working.empty:
        return pd.Series(dtype=float)
    grouped = working.groupby("date", sort=True)
    keep = (grouped["actual"].transform("size") >= min_rows) & (grouped["actual"].transform("nunique") >= 2) & (
        grouped["predicted"].transform("nunique") >= 2
    )
    working = working[keep]
    if working.empty:
        return pd.Series(dtype=float)
    key = working["date"]
    actual_rank = working["actual"].groupby(key).rank()
    predicted_rank = working["predicted"].groupby(key).rank()
    centered_actual = actual_rank - actual_rank.groupby(key).transform("mean")
    centered_predicted = predicted_rank - predicted_rank.groupby(key).transform("mean")
    covariance = (centered_actual * centered_predicted).groupby(key).sum()
    denominator = np.sqrt(
        (centered_actual ** 2).groupby(key).sum() * (centered_predicted ** 2).groupby(key).sum()
    )
    result = (covariance / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    result.index.name = "date"
    return result


def timing_table(ic_series: pd.Series, windows: Sequence[int], forward: int) -> pd.DataFrame:
    """Correlate a trailing IC window with the next-``forward``-day IC.

    ``signal`` at date ``t`` averages ICs from ``t-window`` to ``t-LABEL_HORIZON``;
    ``target`` averages the ICs of ``t+1 .. t+forward``.  Because the IC series is
    indexed on trading dates, the shifts are in sessions.
    """

    rows = []
    for window in windows:
        trailing = ic_series.shift(LABEL_HORIZON).rolling(int(window), min_periods=int(window) // 2).mean()
        future = ic_series[::-1].rolling(int(forward), min_periods=int(forward) // 2).mean()[::-1].shift(-1)
        paired = pd.DataFrame({"signal": trailing, "target": future}).dropna()
        if len(paired) < 30:
            rows.append({"window": int(window), "forward": int(forward), "n": len(paired)})
            continue
        signal_sign = np.sign(paired["signal"])
        target_sign = np.sign(paired["target"])
        comparable = signal_sign.ne(0) & target_sign.ne(0)
        rows.append({
            "window": int(window),
            "forward": int(forward),
            "n": int(len(paired)),
            "corr": float(paired["signal"].corr(paired["target"])),
            "sign_agreement": float((signal_sign[comparable] == target_sign[comparable]).mean()),
            "trailing_mean_ic": float(paired["signal"].mean()),
            "forward_mean_ic": float(paired["target"].mean()),
        })
    return pd.DataFrame(rows)


def split_bounds(frame: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    bounds = {}
    for split, group in frame.groupby("source_split"):
        bounds[split] = {"start": str(group["date"].min().date()), "end": str(group["date"].max().date()),
                         "rows": int(len(group))}
    return bounds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", default=PANEL_CACHE)
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="factor_timing_v1")
    parser.add_argument("--features", default=",".join(FEATURE_NAMES))
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--forward", default=str(DEFAULT_FORWARD))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    features = [name.strip() for name in args.features.split(",") if name.strip()]
    windows = [int(value) for value in args.windows.split(",") if value.strip()]
    forward = int(args.forward)

    panel = load_or_build_panel(
        args.panel_cache, usecols=["date", "symbol", "source_split", LABEL_COLUMN, *features]
    )

    all_rows = []
    for feature in features:
        series = daily_ic_series(panel, feature, LABEL_COLUMN)
        if series.empty:
            continue
        for split in ("train", "validation", "test", "all"):
            subset = series if split == "all" else series[
                (series.index >= panel.loc[panel["source_split"] == split, "date"].min())
                & (series.index <= panel.loc[panel["source_split"] == split, "date"].max())
            ]
            if len(subset) < 40:
                continue
            table = timing_table(subset, windows, forward)
            table.insert(0, "split", split)
            table.insert(0, "feature", feature)
            table["ic_mean"] = float(subset.mean())
            all_rows.append(table)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "factor_timing_{}.csv".format(args.run_id)
    result.to_csv(table_path, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "is the sign of a fundamental factor forecastable from trailing information",
        "signal": "trailing mean daily Rank IC over [t-window, t-10]",
        "target": "mean daily Rank IC over [t+1, t+forward]",
        "label_horizon_exclusion": LABEL_HORIZON,
        "features": features,
        "windows": windows,
        "forward": forward,
        "split_bounds": split_bounds(panel),
        "path": str(table_path),
    }
    manifest_path = output_dir / "factor_timing_manifest_{}.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(result.round(4).to_string(index=False))
    print("\nmanifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
