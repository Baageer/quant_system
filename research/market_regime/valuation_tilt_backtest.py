"""Does a valuation tilt pay?  A monthly buy-cheap / sell-expensive measurement.

This module exists to answer one specific question with numbers rather than
argument: *if* you ranked a stock universe by valuation every month, bought the
cheapest decile and sold the most expensive, what would that have earned?

It is deliberately an **in-sample descriptive measurement of one fixed rule**,
not a strategy search and not evidence of future profitability.  Four things
have to be said before any number below is quoted:

* **The pool is today's snapshot.**  ``HS300 u ZZ1000`` membership as of 2026 is
  applied to 2018-2025 history, so every symbol that was dropped from an index
  -- or went to zero -- is absent by construction.  This biases *every* decile
  upward, and the cheap decile more than the expensive one.
* **The signal is the published L2 percentile**, read from the
  ``stock_valuation_diagnostic`` artifacts rather than recomputed here.  The
  thing being tested is therefore also the thing being shipped.
* **"Sell the expensive" is not implementable in A-shares.**  The long-short
  spread is reported because it is the clean measurement of whether valuation
  separates returns at all; the implementable arm is long-cheap versus the
  universe.
* **One rebalance phase.**  This repo has already measured that changing only
  the rebalance *phase* moves a validation result by ~39pp
  (``docs/research/backtest_rebalance_phase_noise``).  Monthly month-end is a
  single phase; there is no phase-averaging here, because the valuation source
  is monthly and offers no intra-month snapshots to average over.

The prior on this is strongly negative and on the record: the value *factor*
carries a real cross-sectional IC that flips sign on a 2-3 year style cycle,
and the high-BP-minus-low-BP spread earned **+0.80%/yr at 24.52% vol over the
full sample -- Sharpe 0.03** (``docs/research/03_factor_and_valuation/value_factor_regime_study_summary.md``.
That study used daily ``bp`` on HS300 with no costs.  This one uses monthly
multi-metric valuation state on 1295 symbols with costs, which is a different
enough configuration to be worth measuring rather than assuming.

Example::

    python -m research.market_regime.valuation_tilt_backtest \
        --signal-run stock_valuation_v1 --run-id valuation_tilt_v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR, to_tushare_symbol
from research.market_regime.output_paths import DATASET_DIR

PREFIX = "valuation_tilt"
SIGNAL_PREFIX = "stock_valuation"
DEFAULT_SIGNAL_RUN = "stock_valuation_v1"
DEFAULT_PRICE_DIR = DEFAULT_PRICE_HISTORY_DIR / "raw_hfq_pct"
# The decimals metric set the L2 layer ranks.  Kept in one place so a metric
# added to the diagnostic cannot silently enter a backtest it was never
# validated for.
METRICS = ("pe_ttm", "pb", "ps_ttm", "pcf")
# A composite needs more than one measurement to be a composite; the same veto
# ``composite_own_state`` uses.
MIN_METRICS = 2
# One-way cost as a fraction.  A-share round trip is roughly: commission
# ~0.025% each way, stamp duty 0.05% on the sell, transfer fee ~0.001%, plus
# impact.  0.10% per side is the optimistic end, 0.30% the pessimistic one; the
# headline run uses 0.20% and the sensitivity is reported beside it.
DEFAULT_COST_BPS = 20.0
DEFAULT_GROUPS = 10
# A month with fewer than this many priced names is not a cross-section.
MIN_NAMES = 50


def load_signal(path: Path) -> pd.DataFrame:
    """The published L2 cross-sectional percentile, long format.

    ``cheapness`` is 1 = cheapest among peers *today*, already sign-corrected
    for the metric's direction by ``valuation_state_core.cheapness_score``.  It
    is used as-is: recomputing the direction here would create a second place
    for the convention to live and drift.
    """

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"date", "symbol", "metric", "cheapness"}
    if not required <= set(frame.columns):
        raise ValueError(
            "signal file {} is missing {}".format(path, sorted(required - set(frame.columns)))
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["cheapness"] = pd.to_numeric(frame["cheapness"], errors="coerce")
    return frame.loc[frame["metric"].isin(METRICS) & frame["cheapness"].notna()]


def build_composites(signal: pd.DataFrame, min_metrics: int = MIN_METRICS) -> pd.DataFrame:
    """Median cheapness across a symbol-month's available metrics.

    The median, not the mean: one metric in the extreme tail (a loss-maker's
    PS, a negative-cashflow PCF) should move the verdict by its vote, not by
    its magnitude.  ``n_metrics`` travels with the score so a two-metric month
    can be excluded downstream.
    """

    grouped = signal.groupby(["date", "symbol"])["cheapness"]
    composite = pd.DataFrame({
        "cheapness": grouped.median(),
        "n_metrics": grouped.count(),
    }).reset_index()
    return composite.loc[composite["n_metrics"] >= min_metrics]


def load_monthly_returns(
    price_dir: Path,
    symbols: Sequence[str],
    start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Month-end-to-month-end returns per symbol, from the adjusted close.

    The ``raw_hfq_pct`` series is verified to be a total-return series: its
    close-to-close returns reproduce the vendor's ``pct_change`` column exactly
    (median absolute difference 0.0 over 4700 days on 000001.SZ).  Using raw
    prices here would inject the dividend wedge the passive-benchmark work
    already got bitten by (~2%/yr).
    """

    series: Dict[str, pd.Series] = {}
    missing = []
    for symbol in symbols:
        path = Path(price_dir) / "{}.csv".format(to_tushare_symbol(symbol))
        candidates = sorted(Path(price_dir).glob("{}_*.csv".format(to_tushare_symbol(symbol))))
        if candidates:
            path = candidates[-1]  # latest snapshot window wins
        if not path.is_file():
            missing.append(symbol)
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame.columns = [str(column).strip().lstrip("﻿") for column in frame.columns]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        monthly = close.set_axis(frame["date"]).dropna().resample("ME").last()
        series[symbol] = monthly.pct_change()

    if not series:
        raise ValueError("no price file matched any of the {} symbols".format(len(symbols)))
    returns = pd.DataFrame(series).sort_index()
    if start is not None:
        returns = returns.loc[returns.index >= start]
    return returns, missing


def rank_groups(signal_row: pd.Series, groups: int) -> pd.Series:
    """Decile label 1..N per month, 1 = cheapest.

    ``rank(method="first")`` rather than a quantile cut: a month where a third
    of the universe sits on exactly the same cheapness (a floor of 1.0, say)
    must not put a third of the universe in one bucket or in none.
    """

    valid = signal_row.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    ranks = valid.rank(method="first")
    labels = np.ceil(ranks / len(valid) * groups)
    return labels.astype(int)


def portfolio_returns(
    signal_wide: pd.DataFrame,
    returns: pd.DataFrame,
    groups: int = DEFAULT_GROUPS,
    cost_bps: float = DEFAULT_COST_BPS,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Monthly gross and net returns for every decile, plus the net-weight frame."""

    gross_rows = []
    net_rows = []
    turnover_rows = []
    for date in signal_wide.index:
        # The signal is formed at the *close* of month t; the return it earns is
        # the one that follows.  Shifting the signal, not the return, keeps the
        # timing honest when a month is absent from one of the two frames.
        future = returns.index[returns.index > date]
        if len(future) == 0:
            continue
        hold = future[0]
        forward = returns.loc[hold]
        labels = rank_groups(signal_wide.loc[date], groups)
        names = [name for name in labels.index if pd.notna(forward.get(name, np.nan))]
        if len(names) < MIN_NAMES:
            continue
        labels = labels.loc[names]
        month = pd.DataFrame({"group": labels, "ret": forward.loc[names]})

        row_gross = {"date": date, "hold": hold, "n_names": len(names)}
        row_net = {"date": date, "hold": hold, "n_names": len(names)}
        row_turn = {"date": date, "hold": hold}
        for group in range(1, groups + 1):
            members = month.loc[month["group"].eq(group)]
            if members.empty:
                continue
            row_gross[group] = float(members["ret"].mean())
            row_net[group] = float(members["ret"].mean())
            row_turn[group] = float(len(members))
        # Equal-weight benchmark over the same names -- the like-for-like null.
        row_gross[0] = float(month["ret"].mean())
        row_net[0] = float(month["ret"].mean())
        row_turn[0] = float(len(names))
        gross_rows.append(row_gross)
        net_rows.append(row_net)
        turnover_rows.append(row_turn)

    gross = pd.DataFrame(gross_rows).set_index("date") if gross_rows else pd.DataFrame()
    net = pd.DataFrame(net_rows).set_index("date") if net_rows else pd.DataFrame()
    turnover = pd.DataFrame(turnover_rows).set_index("date") if turnover_rows else pd.DataFrame()

    if gross.empty:
        return gross, net, turnover, pd.DataFrame()

    # Charge costs on the realised one-way turnover of each decile.  Membership
    # is recomputed from scratch each month, so turnover is the share of the
    # sleeve that changed -- measured from the name sets, not assumed to be
    # 1/N.
    costs = _turnover_costs(signal_wide, returns, groups, cost_bps)
    for column in net.columns:
        if column in costs.columns:
            net[column] = net[column] - costs[column]
    return gross, net, turnover, costs


def _turnover_costs(
    signal_wide: pd.DataFrame,
    returns: pd.DataFrame,
    groups: int,
    cost_bps: float,
) -> pd.DataFrame:
    """One-way turnover cost per decile per month, as a fraction of the sleeve."""

    cost_rows = []
    previous: Optional[Dict[int, set]] = None
    for date in signal_wide.index:
        future = returns.index[returns.index > date]
        if len(future) == 0:
            continue
        forward = returns.loc[future[0]]
        labels = rank_groups(signal_wide.loc[date], groups)
        names = [name for name in labels.index if pd.notna(forward.get(name, np.nan))]
        if len(names) < MIN_NAMES:
            continue
        labels = labels.loc[names]
        current = {
            group: set(labels.index[labels.eq(group)])
            for group in range(1, groups + 1)
        }
        current[0] = set(labels.index)
        row = {"date": date}
        for group, members in current.items():
            if not members:
                continue
            if previous is None or not previous.get(group):
                # First month is a full build from cash: 100% one-way.
                row[group] = cost_bps / 1e4
            else:
                left = members - previous[group]
                row[group] = len(left) / len(members) * cost_bps / 1e4
        cost_rows.append(row)
        previous = current
    return pd.DataFrame(cost_rows).set_index("date") if cost_rows else pd.DataFrame()


def summarize(monthly: pd.DataFrame, groups: int, label: str) -> Dict[str, float]:
    """Annualised return / vol / Sharpe / max drawdown / hit rate for one sleeve.

    ``t_stat`` is the one to read for significance, and it is computed on the
    **arithmetic** monthly mean, not on the annualised compound return.  The two
    disagree sharply on a path like the cheap decile's (compound -4.7%/yr,
    arithmetic -0.8%/yr): compounding punishes the drawdown twice, so a Sharpe
    built from the compound figure overstates how far from zero the mean is.
    """

    values = monthly.dropna()
    if values.empty:
        return {}
    equity = (1.0 + values).cumprod()
    years = len(values) / 12.0
    annual = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = values.std() * np.sqrt(12)
    standard_error = values.std() / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return {
        "sleeve": label,
        "months": int(len(values)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": float(annual),
        "mean_monthly": float(values.mean()),
        "annual_vol": float(vol),
        "sharpe": float(annual / vol) if vol > 0 else np.nan,
        "t_stat": float(values.mean() / standard_error) if standard_error else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "hit_rate": float((values > 0).mean()),
    }


def evaluate(
    signal_path: Path,
    price_dir: Path = DEFAULT_PRICE_DIR,
    groups: int = DEFAULT_GROUPS,
    cost_bps: float = DEFAULT_COST_BPS,
    limit_symbols: int = 0,
) -> Dict[str, object]:
    """Signal, prices, deciles.  Pure with respect to the filesystem."""

    signal = load_signal(signal_path)
    composite = build_composites(signal)
    symbols = sorted(composite["symbol"].unique())
    if limit_symbols:
        symbols = symbols[:limit_symbols]
        composite = composite.loc[composite["symbol"].isin(symbols)]
    returns, missing = load_monthly_returns(price_dir, symbols)

    results: Dict[str, object] = {"missing_prices": missing, "blocks": {}, "deciles": {}}
    # One block per metric plus the composite, on the same price frame, so the
    # decile columns are comparable instead of each metric defining its own
    # universe.
    # One block per metric plus the composite, all on the same price frame, so
    # the decile columns are comparable instead of each metric defining its own
    # universe.
    blocks = {"composite": composite}
    for metric in METRICS:
        subset = signal.loc[signal["metric"].eq(metric)]
        if subset.empty:
            continue
        blocks[metric] = subset[["date", "symbol", "cheapness"]]

    for name, block in blocks.items():
        wide = block.pivot_table(
            index="date", columns="symbol", values="cheapness", aggfunc="last"
        )
        common = [column for column in wide.columns if column in returns.columns]
        if len(common) < MIN_NAMES:
            continue
        # Restrict the price frame to the months the signal covers: a decile
        # measured on 2018-2025 must not be diluted by 2005-2017 price history.
        window = returns.loc[
            (returns.index >= wide.index.min()) & (returns.index <= wide.index.max()),
            common,
        ]
        gross, net, turnover, costs = portfolio_returns(
            wide.loc[:, common], window, groups=groups, cost_bps=cost_bps
        )
        if gross.empty:
            continue
        results["deciles"][name] = gross
        row = []
        row.append(summarize(net[0], groups, "{}: universe (equal weight)".format(name)))
        for group in range(1, groups + 1):
            if group in net.columns:
                row.append(summarize(net[group], groups, "{}: D{}".format(name, group)))
        row.append(summarize(net[1] - net[groups], groups, "{}: D1-D{} gross-of-nothing".format(name, groups)))
        results["blocks"][name] = pd.DataFrame([item for item in row if item])
        results["blocks"][name + "_gross"] = pd.DataFrame([
            summarize(gross[1], groups, "{}: D1 gross".format(name)),
            summarize(gross[groups], groups, "{}: D{} gross".format(name, groups)),
            summarize(gross[1] - gross[groups], groups, "{}: D1-D{} gross".format(name, groups)),
            summarize(gross[0], groups, "{}: universe gross".format(name)),
        ])
        results["blocks"][name + "_turnover"] = costs
        results[name + "_monthly_net"] = net
        results[name + "_monthly_gross"] = gross

    return results


def yearly(net: pd.DataFrame, groups: int) -> pd.DataFrame:
    """Calendar-year return of each sleeve -- where a style flip becomes visible."""

    if net.empty:
        return net
    rows = []
    for year, block in net.groupby(net.index.year):
        row = {"year": int(year)}
        for column in [0] + list(range(1, groups + 1)):
            if column in block.columns:
                row["universe" if column == 0 else "D{}".format(column)] = float(
                    (1.0 + block[column].dropna()).prod() - 1.0
                )
        if 1 in block.columns and groups in block.columns:
            row["D1-D{}".format(groups)] = float(
                (1.0 + (block[1] - block[groups]).dropna()).prod() - 1.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-run", default=DEFAULT_SIGNAL_RUN,
                        help="stock_valuation run id whose cross_section table holds the signal")
    parser.add_argument("--signal-file", default="",
                        help="explicit cross_section CSV, overriding --signal-run")
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_DIR))
    parser.add_argument("--out-dir", default=str(DATASET_DIR))
    parser.add_argument("--run-id", default="valuation_tilt_v1")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="one-way cost in basis points")
    parser.add_argument("--limit-symbols", type=int, default=0, help="smoke test on the first N")
    args = parser.parse_args(argv)

    signal_path = Path(args.signal_file) if args.signal_file else Path(args.out_dir) / (
        "{}_{}_{}.csv".format(SIGNAL_PREFIX, "cross_section", args.signal_run)
    )
    if not signal_path.is_file():
        raise SystemExit("signal file not found: {}".format(signal_path))

    results = evaluate(
        signal_path,
        price_dir=Path(args.price_dir),
        groups=args.groups,
        cost_bps=args.cost_bps,
        limit_symbols=args.limit_symbols,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in results["blocks"].items():
        path = out_dir / "{}_{}_{}.csv".format(PREFIX, name, args.run_id)
        # The per-month frames carry the month in the index; dropping it would
        # leave a cost table with no way to tell which month each row is.
        table.to_csv(path, index=table.index.name is not None, encoding="utf-8-sig")
        written[name] = str(path)
    composite_net = results.get("composite_monthly_net")
    if composite_net is not None and not composite_net.empty:
        path = out_dir / "{}_composite_monthly_{}.csv".format(PREFIX, args.run_id)
        composite_net.to_csv(path, index=True, encoding="utf-8-sig")
        written["composite_monthly"] = str(path)
        path = out_dir / "{}_composite_yearly_{}.csv".format(PREFIX, args.run_id)
        yearly(composite_net, args.groups).to_csv(path, index=False, encoding="utf-8-sig")
        written["composite_yearly"] = str(path)

    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "月度按估值分位调仓（买最便宜十分位、卖最贵十分位）历史上收益如何",
        "descriptive_only": True,
        "in_sample": True,
        "not_a_strategy_search": True,
        "signal_source": str(signal_path),
        "signal_run": args.signal_run,
        "price_dir": args.price_dir,
        "price_adjustment": "raw_hfq_pct (total-return; close-to-close == vendor pct_change)",
        "metrics": list(METRICS),
        "min_metrics_for_composite": MIN_METRICS,
        "groups": args.groups,
        "cost_bps_one_way": args.cost_bps,
        "min_names_per_month": MIN_NAMES,
        "rebalance": "month-end, single phase (no phase averaging available: monthly source)",
        "caveats": [
            "pool is the 2026 HS300 u ZZ1000 snapshot applied to history: survivorship bias, "
            "upward on every decile and most on the cheap one",
            "shorting the expensive decile is not implementable in A-shares",
            "single rebalance phase; this repo measured ~39pp validation spread from phase alone",
            "in-sample: no train/validation/test split was used, by design -- this measures the "
            "rule that was asked about, it does not select one",
        ],
        "prior_evidence": "docs/research/03_factor_and_valuation/value_factor_regime_study_summary.md (bp spread Sharpe 0.03)",
        "paths": written,
    }
    manifest_path = out_dir / "{}_manifest_{}.json".format(PREFIX, args.run_id)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    print()
    for name in ("composite",) + METRICS:
        key_gross = name + "_gross"
        if key_gross not in results["blocks"]:
            continue
        print("=== {} ===".format(name))
        print(results["blocks"][key_gross].round(4).to_string(index=False))
        print(results["blocks"][name].round(4).to_string(index=False))
        print()
    if composite_net is not None and not composite_net.empty:
        print("=== 分年（composite，净成本）===")
        print(yearly(composite_net, args.groups).round(4).to_string(index=False))
    print()
    print("valuation tilt backtest done; manifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
