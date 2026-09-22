"""Cross-check the label-based zz1000 pool premium against prices.

``pool_return_audit`` reports two descriptions of "buy the pool" that disagree
by 9.3pp after the point-in-time filter is applied (label composition +12.76%,
price buy-and-hold/rebalanced +20.6%/+22.0%), where before the filter they
agreed to 1.9pp.  The price series are built from the *union* of every symbol
that was ever in the PIT-filtered score file, so the price "pool" holds names
for the whole window regardless of whether they were members then, and drops
delisted names from the mean entirely -- both of which re-introduce exactly the
survivorship the PIT filter was meant to remove.

This script rebuilds the price pool with the membership mask applied at each
10-day window start, so the price and label estimators describe the same
portfolio, and reports both the union and the member-only figures.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.index_membership import membership_mask
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


ARMS = {
    "static": "zz1000_excess_v1_outcome",
    "pit": "zz1000_pit_outcome",
}


def pool_price_returns_by_membership(
    scores: pd.DataFrame, pool: str, horizon: int = 10
) -> dict:
    market_data, _ = build_rank_market_data(scores, None, SCORE_COLUMN)
    closes = pd.DataFrame({s: f["close"] for s, f in market_data.items()}).sort_index()
    dates = closes.index

    member = pd.DataFrame(
        {"date": scores["date"], "symbol": scores["symbol"], "member": membership_mask(scores, pool).to_numpy()}
    ).drop_duplicates(["date", "symbol"])
    member_sets = member[member["member"]].groupby("date")["symbol"].apply(set)

    starts = list(range(0, len(dates) - horizon, horizon))
    union, restricted, counts = [], [], []
    for start in starts:
        begin, end = closes.iloc[start], closes.iloc[start + horizon]
        window = (end / begin - 1.0).replace([np.inf, -np.inf], np.nan)
        union.append(float(window.mean(skipna=True)))
        symbols = member_sets.get(dates[start], set())
        columns = [c for c in closes.columns if c in symbols]
        counts.append(len(columns))
        restricted.append(float(window[columns].mean(skipna=True)) if columns else np.nan)

    def compound(values):
        return float(np.prod([1.0 + v for v in values if v == v]) - 1.0)

    return {
        "symbols_in_price_frame": int(closes.shape[1]),
        "windows": len(starts),
        "union_equal_weight_rebalanced": compound(union),
        "member_only_equal_weight_rebalanced": compound(restricted),
        "mean_members_per_window": float(np.mean(counts)),
        "first_window_members": int(counts[0]),
        "last_window_members": int(counts[-1]),
    }


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="pit")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--pool", default="zz1000")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args(argv)

    scores = load_outcome_scores(Path(args.dataset_dir), ARMS[args.arm], "test", args.horizon)
    report = pool_price_returns_by_membership(scores, args.pool, args.horizon)
    print(ARMS[args.arm])
    for key, value in report.items():
        if key in COUNT_KEYS:
            print("  {:<40} {:.1f}".format(key, value))
        else:
            print("  {:<40} {:.4%}".format(key, value))
    return report


COUNT_KEYS = (
    "symbols_in_price_frame",
    "windows",
    "mean_members_per_window",
    "first_window_members",
    "last_window_members",
)


if __name__ == "__main__":
    main()
