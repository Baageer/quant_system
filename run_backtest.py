"""
Generic backtest entry script.

Supports:
- loading timing and stop strategies from YAML
- single-strategy and multi-strategy signal combinations
- exporting trade records after a backtest run
"""

import argparse
import os
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_max_drawdown, calculate_sharpe_ratio
from backtest.performance import PerformanceAnalyzer
from data.data_api import DataAPI
from signals.signal_engine import SignalEngine
from signals.strategy_loader import StrategyLoader
from utils.logger import setup_logger


def create_strategy_function(trade_amount):
    """Create the broker-facing strategy callback from per-symbol signals."""

    def strategy_func(date, data, positions):
        signals = {}

        for symbol, df in data.items():
            if date not in df.index:
                continue

            current_signal = df.loc[date, "signal"]
            if pd.isna(current_signal):
                continue

            current_pos = positions.get(symbol, 0)
            current_price = df.loc[date, "close"]
            shares = int(trade_amount / current_price)

            if current_signal == 1 and current_pos == 0:
                signals[symbol] = {"action": "buy", "shares": shares}
            elif current_signal == -1 and current_pos > 0:
                signals[symbol] = {"action": "sell", "shares": current_pos}

        return signals

    return strategy_func


def run_backtest(
    strategy_name: Union[str, List[str]],
    start_date: str,
    end_date: str,
    config_path: str = "./config/settings.yaml",
    strategy_config_path: str = "./config/strategies.yaml",
    stock_file: str = None,
    initial_capital: float = None,
    trade_amount: float = None,
    enable_stop_loss: bool = True,
    enable_stop_profit: bool = True,
    signal_combination: str = "weighted",
    signal_weights: Optional[List[float]] = None,
    signal_threshold: float = 0.5,
):
    """Run a configured backtest."""

    logger = setup_logger()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    strategy_loader = StrategyLoader(strategy_config_path)

    if isinstance(strategy_name, str):
        strategy_names = [strategy_name]
    else:
        strategy_names = strategy_name

    strategies, strategy_infos = strategy_loader.build_timing_strategies(strategy_names)
    signal_engine = SignalEngine()

    if signal_weights is None:
        signal_weights = [1.0 / len(strategies)] * len(strategies)
    elif len(signal_weights) != len(strategies):
        raise ValueError(
            f"Signal weight count ({len(signal_weights)}) does not match "
            f"strategy count ({len(strategies)})."
        )

    min_data_length = max(info.get("min_data_length", 20) for info in strategy_infos)
    backtest_config = strategy_loader.config.get("backtest", {})

    if initial_capital is None:
        initial_capital = backtest_config.get(
            "initial_capital", config["backtest"]["initial_capital"]
        )
    if trade_amount is None:
        trade_amount = backtest_config.get("trade_amount", 100000)
    if stock_file is None:
        stock_file = config["data"].get("stock_file", "./data/test1.txt")

    (
        stop_loss_strategy,
        stop_loss_info,
        stop_profit_strategy,
        stop_profit_info,
    ) = strategy_loader.build_stop_strategies(
        enable_stop_loss=enable_stop_loss,
        enable_stop_profit=enable_stop_profit,
    )

    logger.info("=" * 60)
    if len(strategies) == 1:
        logger.info(f"Strategy: {strategy_infos[0]['name']}")
        logger.info(f"Params: {strategy_infos[0]['params']}")
    else:
        logger.info(f"Strategy combination ({len(strategies)} strategies):")
        for i, (info, weight) in enumerate(zip(strategy_infos, signal_weights), 1):
            logger.info(f"  {i}. {info['name']} - weight: {weight:.2f}")
            logger.info(f"     Params: {info['params']}")
        logger.info(f"Signal combination mode: {signal_combination}")
        if signal_combination == "threshold":
            logger.info(f"Signal threshold: {signal_threshold}")

    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Initial capital: {initial_capital:,.2f}")
    logger.info(f"Trade amount: {trade_amount:,.2f}")
    logger.info(f"Stock file: {stock_file}")

    if stop_loss_strategy:
        logger.info(f"Stop loss enabled: {stop_loss_info['params']}")
    else:
        logger.info("Stop loss disabled")

    if stop_profit_strategy:
        logger.info(f"Stop profit enabled: {stop_profit_info['params']}")
    else:
        logger.info("Stop profit disabled")
    logger.info("=" * 60)

    data_api = DataAPI(
        source="akshare",
        stock_file=stock_file,
        cache_dir=config["data"]["cache_dir"],
        processed_dir=config["data"]["processed_dir"],
    )

    stock_list = data_api.get_stock_list()
    stock_list = stock_list[:1]
    logger.info(f"Stock count: {len(stock_list)}")

    date_iterator = tqdm(stock_list, desc="Data loading", unit="symbol", disable=False)
    data = {}

    for symbol in date_iterator:
        df = data_api.get_price_history_data(symbol, start_date, end_date)
        df.columns = [
            "date",
            "code",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_change",
            "change",
            "turnover",
        ]

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.sort_index()

        if len(df) < min_data_length:
            df["signal"] = np.nan
            data[symbol] = df
            continue

        if len(strategies) == 1:
            df["signal"] = strategies[0].generate_signal(df)
        else:
            signals_list = [strategy.generate_signal(df) for strategy in strategies]
            combined_signal = signal_engine.combine_signals(signals_list, signal_weights)

            if signal_combination == "weighted":
                df["signal"] = combined_signal.apply(
                    lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0)
                )
            elif signal_combination == "voting":
                vote_signal = pd.Series(0, index=df.index)
                for signal in signals_list:
                    vote_signal += signal
                df["signal"] = vote_signal.apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
            elif signal_combination == "unanimous":
                print(combined_signal.describe())
                df["signal"] = combined_signal.apply(
                    lambda x: (
                        1
                        if x >= (len(strategies) - 0.5)
                        else (-1 if x <= -(len(strategies) - 0.5) else 0)
                    )
                )
            else:
                df["signal"] = combined_signal.apply(
                    lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0)
                )

        data[symbol] = df
        date_iterator.set_postfix({"loading": symbol})

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=config["backtest"]["commission_rate"],
        slippage=config["backtest"]["slippage"],
    )

    engine.set_stop_strategies(
        stop_loss_strategy=stop_loss_strategy,
        stop_profit_strategy=stop_profit_strategy,
    )

    logger.info("Backtest engine initialized")

    strategy_func = create_strategy_function(trade_amount)

    logger.info("Running backtest...")
    results = engine.run(data, strategy_func, start_date, end_date, show_progress=True)

    logger.info("Backtest completed, analyzing results...")

    results["returns"] = results["portfolio_value"].pct_change()
    results["cumulative_returns"] = (1 + results["returns"]).cumprod()

    total_cash = results["portfolio_value"].iloc[-1]
    total_return = (total_cash / initial_capital - 1) * 100
    sharpe = calculate_sharpe_ratio(results["returns"])
    max_drawdown = calculate_max_drawdown(results["portfolio_value"])
    position_counts = results["positions"].apply(len)
    max_position = position_counts.max()

    trades = engine.get_trades()
    if len(trades) == 0:
        logger.info("No trade records generated")
        return results

    total_trades = len(trades[trades["action"] == "buy"])
    total_trades_done = len(trades[trades["action"] == "sell"])
    win_trades = len(trades[trades["profit"] > 0]) if "profit" in trades.columns else 0
    win_rate = (win_trades / total_trades_done * 100) if total_trades_done > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("Backtest summary")
    logger.info("=" * 60)
    logger.info(f"Initial capital: {initial_capital:,.2f}")
    logger.info(f"Final portfolio value: {total_cash:,.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    logger.info(f"Sharpe ratio: {sharpe:.4f}")
    logger.info(f"Max drawdown: {max_drawdown:.2f}%")
    logger.info(f"Trade count: {total_trades}")
    logger.info(f"Win rate: {win_rate:.2f}%")
    logger.info(f"Max concurrent positions: {max_position}")
    logger.info(f"Current positions: {len(results['positions'].iloc[-1].keys())}")

    if total_trades > 0 and "profit" in trades.columns:
        avg_profit = trades["profit"].mean()
        avg_profit_pct = trades["profit_pct"].mean()
        logger.info(f"Average profit: {avg_profit:,.2f}")
        logger.info(f"Average profit pct: {avg_profit_pct:.2f}%")

        if "reason" in trades.columns:
            stop_loss_count = len(trades[trades["reason"] == "stop_loss"])
            stop_profit_count = len(trades[trades["reason"] == "stop_profit"])
            strategy_count = len(
                trades[(trades["action"] == "sell") & (trades["reason"] == "strategy")]
            )
            logger.info(f"Stop-loss exits: {stop_loss_count}")
            logger.info(f"Stop-profit exits: {stop_profit_count}")
            logger.info(f"Strategy exits: {strategy_count}")

    print_trades = False
    if len(trades) > 0 and print_trades:
        logger.info("\n" + "-" * 60)
        logger.info("Trade records")
        logger.info("-" * 60)
        for _, trade in trades.iterrows():
            reason = trade.get("reason", "strategy")
            reason_map = {
                "strategy": "strategy",
                "stop_loss": "stop_loss",
                "stop_profit": "stop_profit",
            }
            reason_str = reason_map.get(reason, reason)

            if trade["action"] == "buy":
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"buy {trade['shares']} @ {trade['price']:.2f} | "
                    f"cost: {trade['cost']:,.2f}"
                )
            else:
                profit_str = (
                    f"profit: {trade['profit']:,.2f} ({trade['profit_pct']:.2f}%)"
                    if "profit" in trade
                    else ""
                )
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"sell {trade['shares']} @ {trade['price']:.2f} | "
                    f"{profit_str} | {reason_str}"
                )

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name_str = "_".join(strategy_names) if len(strategy_names) > 1 else strategy_names[0]
    results_file = f"{output_dir}/backtest_{strategy_name_str}_{timestamp}.csv"
    trades_file = f"{output_dir}/trades_{strategy_name_str}_{timestamp}.csv"

    # results.to_csv(results_file)
    trades.to_csv(trades_file)

    logger.info("\nResults saved")
    logger.info(f"  Backtest results: {results_file}")
    logger.info(f"  Trade records: {trades_file}")

    return results, trades


def main():
    parser = argparse.ArgumentParser(
        description="Generic quant backtest runner with multi-signal support."
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="ma_cross",
        help="Strategy name. Use commas to pass multiple strategies, for example ma_cross,rsi",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config/settings.yaml",
        help="Path to the main config file",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="./config/strategies.yaml",
        help="Path to the strategy config file",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital override",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=None,
        help="Per-trade cash amount override",
    )
    parser.add_argument(
        "--stock-file",
        type=str,
        default=None,
        help="Path to the stock universe file",
    )
    parser.add_argument(
        "--no-stop-loss",
        action="store_true",
        help="Disable stop-loss strategy",
    )
    parser.add_argument(
        "--no-stop-profit",
        action="store_true",
        help="Disable stop-profit strategy",
    )
    parser.add_argument(
        "--signal-combination",
        type=str,
        default="weighted",
        choices=["weighted", "voting", "unanimous"],
        help="How to combine multiple strategy signals",
    )
    parser.add_argument(
        "--signal-weights",
        type=str,
        default=None,
        help="Comma-separated weights for multiple signals, for example 0.6,0.4",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert combined signals into buy or sell actions",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available strategies",
    )

    args = parser.parse_args()

    if args.list:
        loader = StrategyLoader(args.strategy_config)
        loader.list_strategies()
        return

    strategy_names = [s.strip() for s in args.strategy.split(",")]

    signal_weights = None
    if args.signal_weights:
        signal_weights = [float(w.strip()) for w in args.signal_weights.split(",")]

    run_backtest(
        strategy_name=strategy_names if len(strategy_names) > 1 else strategy_names[0],
        start_date=args.start,
        end_date=args.end,
        config_path=args.config,
        strategy_config_path=args.strategy_config,
        stock_file=args.stock_file,
        initial_capital=args.capital,
        trade_amount=args.trade_amount,
        enable_stop_loss=not args.no_stop_loss,
        enable_stop_profit=not args.no_stop_profit,
        signal_combination=args.signal_combination,
        signal_weights=signal_weights,
        signal_threshold=args.signal_threshold,
    )


if __name__ == "__main__":
    main()
