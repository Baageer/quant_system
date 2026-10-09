"""
Simple performance test for optimized backtest engine using mock data.
"""
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.engine import BacktestEngine


def create_mock_data(num_stocks, num_days, start_date='2023-01-01'):
    """Create mock price data for testing."""
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    data = {}
    for i in range(num_stocks):
        symbol = f"STOCK{i:04d}"
        
        # Generate random walk prices
        np.random.seed(i)
        returns = np.random.normal(0.001, 0.02, num_days)
        prices = 100 * np.cumprod(1 + returns)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, num_days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, num_days)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, num_days)),
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, num_days),
        }, index=dates)
        
        # Add simple moving average signals
        df['signal'] = 0
        ma_short = df['close'].rolling(5).mean()
        ma_long = df['close'].rolling(20).mean()
        df.loc[df.index[20:], 'signal'] = np.where(
            ma_short.iloc[20:] > ma_long.iloc[20:],
            1, -1
        )
        
        data[symbol] = df
    
    return data


def simple_strategy_func(date, data, positions):
    """Simple strategy for testing."""
    signals = {}
    trade_amount = 50000
    
    for symbol, df in data.items():
        if date not in df.index:
            continue
        
        current_signal = df.loc[date, "signal"]
        if pd.isna(current_signal):
            continue
        
        current_pos = positions.get(symbol, 0)
        current_price = df.loc[date, "close"]
        if pd.isna(current_price) or current_price <= 0:
            continue
        shares = int(trade_amount / current_price)
        
        if current_signal == 1 and current_pos == 0:
            signals[symbol] = {"action": "buy", "shares": shares}
        elif current_signal == -1 and current_pos > 0:
            signals[symbol] = {"action": "sell", "shares": current_pos}
    
    return signals


def test_performance():
    """Test backtest performance with different data sizes."""
    
    print("=" * 60)
    print("Optimized Backtest Engine Performance Test")
    print("=" * 60)
    
    test_configs = [
        {
            "name": "Small Dataset",
            "num_stocks": 10,
            "num_days": 100,
            "description": "10 stocks × 100 days"
        },
        {
            "name": "Medium Dataset", 
            "num_stocks": 100,
            "num_days": 250,
            "description": "100 stocks × 250 days (1 year)"
        },
        {
            "name": "Large Dataset",
            "num_stocks": 500, 
            "num_days": 500,
            "description": "500 stocks × 500 days (2 years)"
        },
        {
            "name": "Very Large Dataset",
            "num_stocks": 1000,
            "num_days": 500, 
            "description": "1000 stocks × 500 days (2 years)"
        },
        {
            "name": "Very Very Large Dataset",
            "num_stocks": 1000,
            "num_days": 1500, 
            "description": "1000 stocks × 1500 days (6 years)"
        }
    ]
    
    results_summary = []
    
    for config in test_configs:
        print(f"\n{'=' * 60}")
        print(f"Test: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'=' * 60}")
        
        try:
            # Create mock data
            print("Creating mock data...")
            data = create_mock_data(config['num_stocks'], config['num_days'])
            
            # Initialize engine
            print("Initializing backtest engine...")
            engine = BacktestEngine(
                initial_capital=1000000,
                commission_rate=0.0003,
                slippage=0.0001
            )
            
            # Run backtest
            print("Running backtest...")
            start_time = time.time()
            
            results = engine.run(
                data=data,
                strategy_func=simple_strategy_func,
                start_date='2023-01-01',
                end_date=None,
                show_progress=True
            )
            
            elapsed_time = time.time() - start_time
            
            # Calculate statistics
            total_return = (results['portfolio_value'].iloc[-1] / 1000000 - 1) * 100
            num_trades = len([t for t in engine.trades if t.get('status') == 'filled'])
            
            result_summary = {
                'test': config['name'],
                'stocks': config['num_stocks'],
                'days': config['num_days'],
                'total_cells': config['num_stocks'] * config['num_days'],
                'time_seconds': elapsed_time,
                'cells_per_second': (config['num_stocks'] * config['num_days']) / elapsed_time,
                'total_return': total_return,
                'num_trades': num_trades
            }
            results_summary.append(result_summary)
            
            print(f"\n{'=' * 60}")
            print(f"Results: {config['name']}")
            print(f"{'=' * 60}")
            print(f"Execution time: {elapsed_time:.2f} seconds")
            print(f"Processing speed: {result_summary['cells_per_second']:,.0f} cells/second")
            print(f"Total return: {total_return:.2f}%")
            print(f"Number of trades: {num_trades}")
            print(f"Final portfolio value: {results['portfolio_value'].iloc[-1]:,.2f}")
            
        except Exception as e:
            print(f"Error in test {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Test':<20} {'Stocks':<10} {'Days':<10} {'Time(s)':<12} {'Speed(cells/s)':<15} {'Return%':<10}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['test']:<20} {result['stocks']:<10} {result['days']:<10} "
              f"{result['time_seconds']:<12.2f} {result['cells_per_second']:<15,.0f} "
              f"{result['total_return']:<10.2f}")
    
    print(f"\n{'=' * 60}")
    print("OPTIMIZATIONS IMPLEMENTED")
    print(f"{'=' * 60}")
    print("1. Pre-built price matrix for O(1) price lookup")
    print("2. Date-to-symbols mapping to reduce iterations") 
    print("3. Cached position data in stop signal checks")
    print("4. Filtered data passed to strategy function")
    print("5. Reduced redundant DataFrame.loc operations")
    
    print(f"\n{'=' * 60}")
    print("EXPECTED PERFORMANCE IMPROVEMENTS")
    print(f"{'=' * 60}")
    print("- Price retrieval: 10-50x faster (matrix vs loop)")
    print("- Strategy signals: 5-20x faster (filtered data)")
    print("- Stop checks: 2-5x faster (cached data)")
    print("- Overall: 3-10x faster for large stock pools")
    
    # Performance analysis
    if len(results_summary) >= 2:
        print(f"\n{'=' * 60}")
        print("SCALABILITY ANALYSIS")
        print(f"{'=' * 60}")
        
        for i in range(1, len(results_summary)):
            prev = results_summary[i-1]
            curr = results_summary[i]
            
            cells_ratio = curr['total_cells'] / prev['total_cells']
            time_ratio = curr['time_seconds'] / prev['time_seconds']
            efficiency = cells_ratio / time_ratio
            
            print(f"{prev['test']} → {curr['test']}:")
            print(f"  Data size increased: {cells_ratio:.1f}x")
            print(f"  Time increased: {time_ratio:.1f}x") 
            print(f"  Efficiency: {efficiency:.2f}x (ideal = 1.0)")
            
            if efficiency > 0.8:
                print(f"  ✓ Excellent scalability!")
            elif efficiency > 0.5:
                print(f"  ✓ Good scalability")
            else:
                print(f"  ⚠ Scalability could be improved")


if __name__ == "__main__":
    test_performance()