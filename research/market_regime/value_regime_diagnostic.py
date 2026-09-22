"""Diagnostic: is ``market_state`` the variable that flips the value factor's sign?

The fundamental study found ``bp`` at IC +0.074 in test but -0.071 in validation.
An interaction with ``market_state`` can only repair that if the *sign* of ``bp``
differs across states *within* a split.  If every state shares one sign inside
each split, the regime variable that matters is something else entirely.

Also caches the full-universe fundamental panel at
``output/datasets/fundamental_feature_panel_excess_hs300_v1.csv``, which
``factor_timing_study`` and ``value_spread_timing_study`` both consume.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.market_regime.feature_study_core import (  # noqa: E402
    daily_rank_ics,
    load_study_panels,
    load_ohlcv_histories,
    merge_market_state,
)
from research.market_regime.fundamental_feature_study import (  # noqa: E402
    DEFAULT_VALUATION_DIR,
    attach_fundamental_features,
    load_valuation_history,
)

DATASET_DIR = Path("output/datasets")
PANEL_CACHE = DATASET_DIR / "fundamental_feature_panel_excess_hs300_v1.csv"
FOCUS = ("bp", "turnover_mv_mean_20", "ep_ttm", "log_amount_20")


def build_panel() -> pd.DataFrame:
    samples, _ = load_study_panels(DATASET_DIR, "excess_hs300_v1", 10)
    histories, _ = load_ohlcv_histories(samples["symbol"].unique())
    valuations = {}
    for symbol in samples["symbol"].unique():
        history = load_valuation_history(symbol, Path(DEFAULT_VALUATION_DIR))
        if history is not None:
            valuations[symbol] = history
    panel = attach_fundamental_features(samples, histories, valuations)
    panel.to_csv(PANEL_CACHE, index=False, encoding="utf-8-sig")
    return panel


def load_or_build_panel(path: str = PANEL_CACHE, usecols=None) -> pd.DataFrame:
    """Read the cached fundamental panel, building it first when it is absent.

    The cache is ~1.1 GB of derived data, so the timing studies read it through
    here rather than duplicating the (2-3 minute) feature computation or silently
    depending on someone having run this diagnostic first.
    """

    path = Path(path)
    if not path.exists():
        build_panel()
    kwargs = {"dtype": {"symbol": str}, "low_memory": False}
    if usecols is not None:
        kwargs["usecols"] = usecols
    panel = pd.read_csv(path, **kwargs)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    return panel


def main() -> int:
    panel = load_or_build_panel()
    panel = merge_market_state(panel, Path("data/benchmarks/000300.csv"))
    label = "future_excess_return_10d"
    print("panel rows={} symbols={}".format(len(panel), panel["symbol"].nunique()))

    for feature in FOCUS:
        print("\n=== {} : daily Rank IC by split x {} ===".format(feature, "market_state"))
        rows = []
        for split in ("train", "validation", "test"):
            subset = panel[panel["source_split"] == split]
            for state, group in subset.groupby("market_state", dropna=True):
                ics = daily_rank_ics(
                    group["date"],
                    pd.to_numeric(group[label], errors="coerce").to_numpy(float),
                    pd.to_numeric(group[feature], errors="coerce").to_numpy(float),
                )
                rows.append({
                    "split": split, "state": state, "ic": float(np.mean(ics)) if ics.size else np.nan,
                    "n_days": int(ics.size),
                })
        table = pd.DataFrame(rows).pivot(index="state", columns="split", values="ic")
        print(table.round(4).to_string())
        print("sign flips across states within a split: {}".format(
            {split: int(np.sign(table[split].dropna()).nunique() > 1) for split in table.columns}))

    print("\n=== bp : daily Rank IC by year (all splits pooled, per-day then mean) ===")
    rows = []
    for year, group in panel.groupby(panel["date"].dt.year):
        ics = daily_rank_ics(
            group["date"],
            pd.to_numeric(group[label], errors="coerce").to_numpy(float),
            pd.to_numeric(group["bp"], errors="coerce").to_numpy(float),
        )
        rows.append({"year": int(year), "bp_ic": float(np.mean(ics)) if ics.size else np.nan,
                     "turnover_ic": float(np.mean(daily_rank_ics(
                         group["date"],
                         pd.to_numeric(group[label], errors="coerce").to_numpy(float),
                         pd.to_numeric(group["turnover_mv_mean_20"], errors="coerce").to_numpy(float),
                     ))) if ics.size else np.nan})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
