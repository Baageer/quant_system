"""Is the price-based value spread forecastable?  (last route to a causal sign)

``value_regime_diagnostic`` closed the trend-regime route (``market_state`` does
not flip ``bp``'s sign -- all three states share a sign inside each split), and
``factor_timing_study`` closed the IC-based route (the trailing-IC signal's
correlation with the forward IC flips sign with the arbitrary choice of window).

One standard route remains: condition on the factor's *realized* return spread
rather than on its IC.  Each date, sort names by ``bp`` into quintiles and take

    spread_t = mean(next-day return | top quintile bp) - (bottom quintile bp)

which is causal at ``t`` (``bp`` and the returns are both known by ``t+1``).  Then
ask whether the trailing cumulative spread forecasts the forward cumulative
spread -- factor momentum (positive) or factor reversal (negative).  If neither,
the factor's sign is unknowable ex ante by every standard route and the value
line is closed.

Also reports the spread's own mean and volatility, because a forecastable but
economically tiny spread is no more useful than an unforecastable one.

Example::

    python -m research.market_regime.value_spread_timing_study
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.data_loader import load_local_price_history
from research.market_regime.value_regime_diagnostic import load_or_build_panel


PANEL_CACHE = "output/datasets/fundamental_feature_panel_excess_hs300_v1.csv"
DEFAULT_TRAILING = (60, 120, 250)
DEFAULT_FORWARD = 60
N_Q = 5


def build_return_panel(symbols: Sequence[str]) -> pd.DataFrame:
    """date × symbol next-day close-to-close returns from the local price cache."""

    series = {}
    for symbol in sorted(set(symbols)):
        try:
            prices = load_local_price_history(symbol)
        except (FileNotFoundError, ValueError):
            continue
        close = pd.to_numeric(prices["close"], errors="coerce")
        series[symbol] = close.pct_change().shift(-1)
    panel = pd.DataFrame(series)
    panel.index.name = "date"
    return panel


def spread_series(factor: pd.DataFrame, returns: pd.DataFrame, n_q: int = N_Q) -> pd.Series:
    """Daily equal-weight top-minus-bottom quintile return spread.

    ``factor`` is long-form (date, symbol, value); ``returns`` is a date × symbol
    frame of next-day returns.  The result is indexed by the signal date, so the
    value at ``t`` is only known at ``t+1``.
    """

    wide_factor = factor.pivot_table(index="date", columns="symbol", values="value", aggfunc="last")
    common_dates = wide_factor.index.intersection(returns.index)
    common_symbols = wide_factor.columns.intersection(returns.columns)
    wide_factor = wide_factor.loc[common_dates, common_symbols]
    next_return = returns.loc[common_dates, common_symbols]
    ranks = wide_factor.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - 1.0 / n_q)
    bottom = ranks.le(1.0 / n_q)
    counts = wide_factor.notnull().sum(axis=1)
    enough = counts >= n_q * 5
    top_return = (next_return.where(top).sum(axis=1) / top.sum(axis=1).replace(0, np.nan))
    bottom_return = (next_return.where(bottom).sum(axis=1) / bottom.sum(axis=1).replace(0, np.nan))
    spread = (top_return - bottom_return).where(enough)
    spread.index.name = "date"
    return spread.dropna()


def _forward_sum(series: pd.Series, window: int) -> pd.Series:
    """Sum of the next ``window`` values, excluding the current one.

    The spread at ``d`` is realized at ``d+1``, so the trailing sum ends at
    ``t-1`` and the forward sum starts at ``t``.
    """

    reversed_rolling = series[::-1].rolling(window, min_periods=max(2, window // 2)).sum()[::-1]
    return reversed_rolling.shift(-1)


def timing_table(spread: pd.Series, trailing: Sequence[int], forward: int) -> pd.DataFrame:
    rows = []
    for window in trailing:
        signal = spread.shift(1).rolling(window, min_periods=max(2, window // 2)).sum()
        target = _forward_sum(spread, forward)
        paired = pd.DataFrame({"signal": signal, "target": target}).dropna()
        if len(paired) < 30:
            rows.append({"trailing": int(window), "n": len(paired)})
            continue
        rows.append({
            "trailing": int(window),
            "forward": int(forward),
            "n": int(len(paired)),
            "corr": float(paired["signal"].corr(paired["target"])),
            "spearman": float(paired["signal"].rank().corr(paired["target"].rank())),
            "sign_agreement": float((np.sign(paired["signal"]) == np.sign(paired["target"])).mean()),
            "signal_mean": float(paired["signal"].mean()),
            "target_mean": float(paired["target"].mean()),
        })
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", default=PANEL_CACHE)
    parser.add_argument("--factor", default="bp")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="value_spread_timing_v1")
    parser.add_argument("--trailing", default=",".join(str(w) for w in DEFAULT_TRAILING))
    parser.add_argument("--forward", default=str(DEFAULT_FORWARD))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    trailing = [int(value) for value in args.trailing.split(",") if value.strip()]
    forward = int(args.forward)

    factor = load_or_build_panel(
        args.panel_cache, usecols=["date", "symbol", args.factor]
    ).rename(columns={args.factor: "value"})
    factor = factor.dropna(subset=["date", "value"])
    returns = build_return_panel(factor["symbol"].unique())
    spread = spread_series(factor, returns)
    if spread.empty:
        raise ValueError("Spread series is empty")

    daily_vol = float(spread.std())
    report: Dict[str, object] = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "is the price-based value spread forecastable from its own past",
        "factor": args.factor,
        "spread_definition": "equal-weight top-minus-bottom bp quintile next-day return",
        "spread_start": str(spread.index.min().date()),
        "spread_end": str(spread.index.max().date()),
        "spread_mean_daily": float(spread.mean()),
        "spread_std_daily": daily_vol,
        "spread_annualised_mean": float(spread.mean() * 252),
        "spread_annualised_vol": float(daily_vol * np.sqrt(252)),
        "spread_positive_share": float((spread > 0).mean()),
    }

    print("=== {} spread: daily mean {:.5f} ({:.2%}/yr), daily vol {:.5f} ({:.2%}/yr), n={} ===".format(
        args.factor, report["spread_mean_daily"], report["spread_annualised_mean"],
        report["spread_std_daily"], report["spread_annualised_vol"], len(spread)))

    tables = []
    for label, subset in (
        ("train", spread[spread.index <= pd.Timestamp("2018-12-14")]),
        ("validation", spread[(spread.index >= pd.Timestamp("2019-07-17")) & (spread.index <= pd.Timestamp("2021-12-17"))]),
        ("test", spread[spread.index >= pd.Timestamp("2022-07-20")]),
        ("all", spread),
    ):
        if len(subset) < 60:
            continue
        table = timing_table(subset, trailing, forward)
        table.insert(0, "split", label)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "value_spread_timing_{}.csv".format(args.run_id)
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    spread.to_frame("spread").to_csv(output_dir / "value_spread_series_{}.csv".format(args.run_id))
    report["path"] = str(csv_path)
    manifest_path = output_dir / "value_spread_timing_manifest_{}.json".format(args.run_id)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(result.round(4).to_string(index=False))
    print("\nmanifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
