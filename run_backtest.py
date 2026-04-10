"""
通用回测入口脚本
支持通过配置文件选择策略和参数
支持多信号组合
"""
import argparse
import yaml
import importlib
from datetime import datetime
from utils.logger import setup_logger
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from data.data_api import DataAPI
from signals.signal_engine import SignalEngine
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional


class StrategyLoader:
    """策略加载器"""
    
    def __init__(self, strategy_config_path: str = "./config/strategies.yaml"):
        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_strategy(self, strategy_name: str):
        """动态加载策略类"""
        strategies = self.config.get('timing_strategies', {})
        
        if strategy_name not in strategies:
            raise ValueError(f"未找到策略: {strategy_name}，可用策略: {list(strategies.keys())}")
        
        strategy_info = strategies[strategy_name]
        module_name = strategy_info['module']
        class_name = strategy_info['class']
        
        module = importlib.import_module(module_name)
        strategy_class = getattr(module, class_name)
        
        return strategy_class, strategy_info
    
    def get_stop_strategy(self, strategy_name: str):
        """动态加载止盈止损策略类"""
        strategies = self.config.get('stop_strategies', {})
        
        if strategy_name not in strategies:
            return None, None
        
        strategy_info = strategies[strategy_name]
        module_name = strategy_info['module']
        class_name = strategy_info['class']
        
        module = importlib.import_module(module_name)
        strategy_class = getattr(module, class_name)
        
        return strategy_class, strategy_info
    
    def list_strategies(self):
        """列出所有可用策略"""
        strategies = self.config.get('timing_strategies', {})
        print("\n可用策略列表:")
        print("-" * 60)
        for name, info in strategies.items():
            print(f"  {name}: {info['name']}")
            print(f"    描述: {info['description']}")
            print(f"    参数: {info['params']}")
            print()
        
        stop_strategies = self.config.get('stop_strategies', {})
        if stop_strategies:
            print("\n止盈止损策略列表:")
            print("-" * 60)
            for name, info in stop_strategies.items():
                print(f"  {name}: {info['name']}")
                print(f"    描述: {info['description']}")
                print(f"    参数: {info['params']}")
                print()


def create_strategy_function(trade_amount):
    """创建策略函数"""
    
    def strategy_func(date, data, positions):
        """策略函数"""
        signals = {}
        
        for symbol, df in data.items():
            if date not in df.index:
                continue
            
            current_signal = df.loc[date, 'signal']
            if pd.isna(current_signal):
                continue
            
            current_pos = positions.get(symbol, 0)
            current_price = df.loc[date, 'close']
            shares = int(trade_amount / current_price)
            
            if current_signal == 1 and current_pos == 0:
                signals[symbol] = {
                    'action': 'buy',
                    'shares': shares
                }
            elif current_signal == -1 and current_pos > 0:
                signals[symbol] = {
                    'action': 'sell',
                    'shares': current_pos
                }
        
        
        
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
    signal_combination: str = 'weighted',
    signal_weights: Optional[List[float]] = None,
    signal_threshold: float = 0.5
):
    """运行回测"""
    logger = setup_logger()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    strategy_loader = StrategyLoader(strategy_config_path)
    
    if isinstance(strategy_name, str):
        strategy_names = [strategy_name]
    else:
        strategy_names = strategy_name
    
    strategies = []
    strategy_infos = []
    for name in strategy_names:
        strategy_class, strategy_info = strategy_loader.get_strategy(name)
        strategies.append(strategy_class(**strategy_info['params']))
        strategy_infos.append(strategy_info)
    
    signal_engine = SignalEngine()
    
    if signal_weights is None:
        signal_weights = [1.0 / len(strategies)] * len(strategies)
    elif len(signal_weights) != len(strategies):
        raise ValueError(f"权重数量({len(signal_weights)})与策略数量({len(strategies)})不匹配")

    min_data_length = max([info.get('min_data_length', 20) for info in strategy_infos])
    
    backtest_config = strategy_loader.config.get('backtest', {})
    
    if initial_capital is None:
        initial_capital = backtest_config.get('initial_capital', config['backtest']['initial_capital'])
    if trade_amount is None:
        trade_amount = backtest_config.get('trade_amount', 100000)
    if stock_file is None:
        stock_file = config['data'].get('stock_file', './data/test1.txt')
    
    stop_loss_strategy = None
    stop_profit_strategy = None
    
    if enable_stop_loss:
        stop_loss_class, stop_loss_info = strategy_loader.get_stop_strategy('stop_loss')
        if stop_loss_class and stop_loss_info:
            stop_loss_strategy = stop_loss_class(**stop_loss_info['params'])
    
    if enable_stop_profit:
        stop_profit_class, stop_profit_info = strategy_loader.get_stop_strategy('stop_profit')
        if stop_profit_class and stop_profit_info:
            stop_profit_strategy = stop_profit_class(**stop_profit_info['params'])
    
    logger.info("=" * 60)
    if len(strategies) == 1:
        logger.info(f"策略: {strategy_infos[0]['name']}")
        logger.info(f"参数: {strategy_infos[0]['params']}")
    else:
        logger.info(f"多策略组合 ({len(strategies)}个策略):")
        for i, (info, weight) in enumerate(zip(strategy_infos, signal_weights), 1):
            logger.info(f"  {i}. {info['name']} - 权重: {weight:.2f}")
            logger.info(f"     参数: {info['params']}")
        logger.info(f"信号组合方式: {signal_combination}")
        if signal_combination == 'threshold':
            logger.info(f"信号阈值: {signal_threshold}")
    logger.info(f"回测区间: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info(f"单次交易金额: {trade_amount:,.2f}")
    logger.info(f"股票文件: {stock_file}")
    if stop_loss_strategy:
        logger.info(f"止损策略: 已启用 - {stop_loss_info['params']}")
    else:
        logger.info("止损策略: 未启用")
    if stop_profit_strategy:
        logger.info(f"止盈策略: 已启用 - {stop_profit_info['params']}")
    else:
        logger.info("止盈策略: 未启用")
    logger.info("=" * 60)
    
    data_api = DataAPI(
        source='akshare',
        stock_file=stock_file,
        cache_dir=config['data']['cache_dir'],
        processed_dir=config['data']['processed_dir']
    )

    stock_list = data_api.get_stock_list()
    stock_list = stock_list[:1]
    logger.info(f"股票数量: {len(stock_list)}")
    
    date_iterator = tqdm(stock_list, desc="数据加载进度", unit="个", disable=False)
    data = {}
    for symbol in date_iterator:
        df = data_api.get_price_history_data(symbol, start_date, end_date)
        
        df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 
                        'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        if len(df) < min_data_length:
            df['signal'] = np.nan
            data[symbol] = df
            continue

        if len(strategies) == 1:
            signal = strategies[0].generate_signal(df)
            df['signal'] = signal
        else:
            signals_list = []
            for strategy in strategies:
                signal = strategy.generate_signal(df)
                signals_list.append(signal)
            
            combined_signal = signal_engine.combine_signals(signals_list, signal_weights)
            
            if signal_combination == 'weighted':
                df['signal'] = combined_signal.apply(lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0))
            elif signal_combination == 'voting':
                vote_signal = pd.Series(0, index=df.index)
                for signal in signals_list:
                    vote_signal += signal
                df['signal'] = vote_signal.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            elif signal_combination == 'unanimous':
                print(combined_signal.describe())
                df['signal'] = combined_signal.apply(
                    lambda x: 1 if x >= (len(strategies) - 0.5) else (-1 if x <= -(len(strategies) - 0.5) else 0)
                )
            else:
                df['signal'] = combined_signal.apply(lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0))
        
        data[symbol] = df

        date_iterator.set_postfix({
                '加载数据': f'{symbol}',
            })
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=config['backtest']['commission_rate'],
        slippage=config['backtest']['slippage']
    )
    
    engine.set_stop_strategies(
        stop_loss_strategy=stop_loss_strategy,
        stop_profit_strategy=stop_profit_strategy
    )
    
    logger.info("回测引擎初始化完成")

    
    strategy_func = create_strategy_function(trade_amount)
    
    logger.info("开始执行回测...")
    results = engine.run(data, strategy_func, start_date, end_date, show_progress=True)
    
    logger.info("回测执行完成，开始分析结果...")
    
    results['returns'] = results['portfolio_value'].pct_change()
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    total_cash = results['portfolio_value'].iloc[-1]
    total_return = (total_cash / initial_capital - 1) * 100
    sharpe = calculate_sharpe_ratio(results['returns'])
    max_drawdown = calculate_max_drawdown(results['portfolio_value'])
    position_counts = results['positions'].apply(len)
    max_position = position_counts.max()
    
    trades = engine.get_trades()
    if len(trades) == 0:
        logger.info("没有交易记录")
        return results
    
    total_trades = len(trades[trades['action'] == 'buy'])
    total_trades_done = len(trades[trades['action'] == 'sell'])
    win_trades = len(trades[trades['profit'] > 0]) if 'profit' in trades.columns else 0
    win_rate = (win_trades / total_trades_done * 100) if total_trades_done > 0 else 0
    
    logger.info("\n" + "=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info(f"最终资金: {total_cash:,.2f}")
    logger.info(f"总收益率: {total_return:.2f}%")
    logger.info(f"夏普比率: {sharpe:.4f}")
    logger.info(f"最大回撤: {max_drawdown:.2f}%")
    logger.info(f"交易次数: {total_trades}")
    logger.info(f"胜率: {win_rate:.2f}%")
    logger.info(f"最大持仓个数: {max_position}")
    logger.info(f"当前持仓个数: {len(results['positions'].iloc[-1].keys())}")
    
    if total_trades > 0 and 'profit' in trades.columns:
        avg_profit = trades['profit'].mean()
        avg_profit_pct = trades['profit_pct'].mean()
        logger.info(f"平均盈亏: {avg_profit:,.2f} 元")
        logger.info(f"平均盈亏百分比: {avg_profit_pct:.2f}%")
        
        if 'reason' in trades.columns:
            stop_loss_count = len(trades[trades['reason'] == 'stop_loss'])
            stop_profit_count = len(trades[trades['reason'] == 'stop_profit'])
            strategy_count = len(trades[(trades['action'] == 'sell') & (trades['reason'] == 'strategy')])
            logger.info(f"止损卖出: {stop_loss_count} 次")
            logger.info(f"止盈卖出: {stop_profit_count} 次")
            logger.info(f"策略卖出: {strategy_count} 次")
    
    print_trades = False
    if len(trades) > 0 and print_trades:
        logger.info("\n" + "-" * 60)
        logger.info("交易记录")
        logger.info("-" * 60)
        for i, trade in trades.iterrows():
            reason = trade.get('reason', 'strategy')
            reason_map = {
                'strategy': '策略信号',
                'stop_loss': '止损',
                'stop_profit': '止盈'
            }
            reason_str = reason_map.get(reason, reason)
            
            if trade['action'] == 'buy':
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"买入 {trade['shares']}股 @ {trade['price']:.2f} | 成本: {trade['cost']:,.2f}"
                )
            else:
                profit_str = f"盈亏: {trade['profit']:,.2f} ({trade['profit_pct']:.2f}%)" if 'profit' in trade else ""
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"卖出 {trade['shares']}股 @ {trade['price']:.2f} | {profit_str} | {reason_str}"
                )
    
    output_dir = "./output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name_str = "_".join(strategy_names) if len(strategy_names) > 1 else strategy_names[0]
    results_file = f"{output_dir}/backtest_{strategy_name_str}_{timestamp}.csv"
    trades_file = f"{output_dir}/trades_{strategy_name_str}_{timestamp}.csv"
    
    # results.to_csv(results_file)
    trades.to_csv(trades_file)
    
    logger.info(f"\n结果已保存:")
    logger.info(f"  回测结果: {results_file}")
    logger.info(f"  交易记录: {trades_file}")
    
    return results, trades


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统通用回测 - 支持多信号组合')
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='ma_cross',
        help='策略名称，多个策略用逗号分隔 (例如: ma_cross,rsi_strategy) (默认: ma_cross)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config/settings.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--strategy-config',
        type=str,
        default='./config/strategies.yaml',
        help='策略配置文件路径'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='回测开始日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2023-12-31',
        help='回测结束日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=None,
        help='初始资金'
    )
    parser.add_argument(
        '--trade-amount',
        type=float,
        default=None,
        help='单次交易金额'
    )
    parser.add_argument(
        '--stock-file',
        type=str,
        default=None,
        help='股票文件路径'
    )
    parser.add_argument(
        '--no-stop-loss',
        action='store_true',
        help='禁用止损策略'
    )
    parser.add_argument(
        '--no-stop-profit',
        action='store_true',
        help='禁用止盈策略'
    )
    parser.add_argument(
        '--signal-combination',
        type=str,
        default='weighted',
        choices=['weighted', 'voting', 'unanimous'],
        help='信号组合方式: weighted(加权平均), voting(投票), unanimous(一致同意) (默认: weighted)'
    )
    parser.add_argument(
        '--signal-weights',
        type=str,
        default=None,
        help='信号权重，多个权重用逗号分隔 (例如: 0.6,0.4)'
    )
    parser.add_argument(
        '--signal-threshold',
        type=float,
        default=0.5,
        help='信号阈值，用于判断买卖信号 (默认: 0.5)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用策略'
    )
    
    args = parser.parse_args()
    
    if args.list:
        loader = StrategyLoader(args.strategy_config)
        loader.list_strategies()
        return
    
    strategy_names = [s.strip() for s in args.strategy.split(',')]
    
    signal_weights = None
    if args.signal_weights:
        signal_weights = [float(w.strip()) for w in args.signal_weights.split(',')]
    
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
        signal_threshold=args.signal_threshold
    )


if __name__ == "__main__":
    main()
