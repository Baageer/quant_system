"""
测试 vwap_distance 和 rsrs_beta 策略
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from signals.timing.vwap_distance import VWAPDistanceStrategy
from signals.timing.rsrs_beta import RSRSBetaStrategy
from signals.indicators import vwap_distance


def create_test_data(n_days=200):
    """创建测试数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    # 生成随机价格数据
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    high = close + np.random.rand(n_days) * 2
    low = close - np.random.rand(n_days) * 2
    open_price = close + np.random.randn(n_days) * 0.5
    volume = np.random.randint(1000000, 10000000, n_days)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    data.set_index('date', inplace=True)
    
    return data


def test_vwap_distance_strategy():
    """测试 VWAP 距离策略"""
    print("=" * 60)
    print("测试 VWAP Distance 策略")
    print("=" * 60)
    
    data = create_test_data(200)
    
    # 测试 VWAP 距离计算
    vwap_dist = vwap_distance(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        volume=data['volume'],
        window=20
    )
    
    print(f"VWAP 距离统计:")
    print(f"  均值: {vwap_dist.mean():.4f}")
    print(f"  标准差: {vwap_dist.std():.4f}")
    print(f"  最大值: {vwap_dist.max():.4f}")
    print(f"  最小值: {vwap_dist.min():.4f}")
    print(f"  缺失值数量: {vwap_dist.isna().sum()}")
    
    # 测试策略
    strategy = VWAPDistanceStrategy(
        window=20,
        threshold=0.02,
        use_negative=True,
        use_trend_filter=False,
        use_volume_filter=False
    )
    
    signal = strategy.generate_signal(data)
    
    print(f"\n策略信号统计:")
    print(f"  做多次数: {(signal == 1).sum()}")
    print(f"  做空次数: {(signal == -1).sum()}")
    print(f"  空仓次数: {(signal == 0).sum()}")
    print(f"  换手率: {((signal.diff() != 0).sum() / len(signal)):.2%}")
    
    # 测试带过滤的策略
    strategy_with_filters = VWAPDistanceStrategy(
        window=20,
        threshold=0.02,
        use_negative=True,
        use_trend_filter=True,
        trend_window=60,
        use_volume_filter=True,
        volume_window=20
    )
    
    signal_filtered = strategy_with_filters.generate_signal(data)
    
    print(f"\n带过滤的策略信号统计:")
    print(f"  做多次数: {(signal_filtered == 1).sum()}")
    print(f"  做空次数: {(signal_filtered == -1).sum()}")
    print(f"  空仓次数: {(signal_filtered == 0).sum()}")
    print(f"  换手率: {((signal_filtered.diff() != 0).sum() / len(signal_filtered)):.2%}")
    
    # 测试可视化方法
    vwap_values = strategy.get_vwap_distance_values(data)
    print(f"\n可视化数据:")
    print(f"  VWAP 距离序列长度: {len(vwap_values['vwap_distance'])}")
    print(f"  阈值: {vwap_values['vwap_distance_threshold'].iloc[0]}")
    
    print("\n✓ VWAP Distance 策略测试通过")


def test_rsrs_beta_strategy():
    """测试 RSRS beta 策略"""
    print("\n" + "=" * 60)
    print("测试 RSRS Beta 策略")
    print("=" * 60)
    
    data = create_test_data(200)
    
    # 测试策略
    strategy = RSRSBetaStrategy(
        window=20,
        zscore_window=90,
        min_valid_window=12,
        entry_zscore=0.5,
        exit_zscore=0.0,
        use_negative=True,
        use_trend_filter=False,
        use_volume_filter=False
    )
    
    signal = strategy.generate_signal(data)
    
    print(f"策略信号统计:")
    print(f"  做多次数: {(signal == 1).sum()}")
    print(f"  做空次数: {(signal == -1).sum()}")
    print(f"  空仓次数: {(signal == 0).sum()}")
    print(f"  换手率: {((signal.diff() != 0).sum() / len(signal)):.2%}")
    
    # 测试带过滤的策略
    strategy_with_filters = RSRSBetaStrategy(
        window=20,
        zscore_window=90,
        min_valid_window=12,
        entry_zscore=0.5,
        exit_zscore=0.0,
        use_negative=True,
        use_trend_filter=True,
        trend_window=60,
        use_volume_filter=True,
        volume_window=20
    )
    
    signal_filtered = strategy_with_filters.generate_signal(data)
    
    print(f"\n带过滤的策略信号统计:")
    print(f"  做多次数: {(signal_filtered == 1).sum()}")
    print(f"  做空次数: {(signal_filtered == -1).sum()}")
    print(f"  空仓次数: {(signal_filtered == 0).sum()}")
    print(f"  换手率: {((signal_filtered.diff() != 0).sum() / len(signal_filtered)):.2%}")
    
    # 测试可视化方法
    rsrs_values = strategy.get_rsrs_beta_values(data)
    print(f"\n可视化数据:")
    print(f"  RSRS beta 序列长度: {len(rsrs_values['rsrs_beta'])}")
    print(f"  RSRS zscore 均值: {rsrs_values['rsrs_zscore'].mean():.4f}")
    print(f"  RSRS zscore 标准差: {rsrs_values['rsrs_zscore'].std():.4f}")
    print(f"  入场阈值: {rsrs_values['entry_zscore'].iloc[0]}")
    
    print("\n✓ RSRS Beta 策略测试通过")


def test_strategy_compatibility():
    """测试策略与回测系统的兼容性"""
    print("\n" + "=" * 60)
    print("测试策略兼容性")
    print("=" * 60)
    
    data = create_test_data(200)
    
    # 测试 VWAP Distance 策略
    vwap_strategy = VWAPDistanceStrategy(
        window=20,
        threshold=0.02,
        use_negative=True
    )
    
    vwap_signal = vwap_strategy.generate_signal(data)
    print(f"VWAP Distance 信号类型: {type(vwap_signal)}")
    print(f"VWAP Distance 信号范围: [{vwap_signal.min()}, {vwap_signal.max()}]")
    print(f"VWAP Distance 信号值: {vwap_signal.unique()}")
    
    # 测试 RSRS Beta 策略
    rsrs_strategy = RSRSBetaStrategy(
        window=20,
        zscore_window=90,
        entry_zscore=0.5,
        use_negative=True
    )
    
    rsrs_signal = rsrs_strategy.generate_signal(data)
    print(f"\nRSRS Beta 信号类型: {type(rsrs_signal)}")
    print(f"RSRS Beta 信号范围: [{rsrs_signal.min()}, {rsrs_signal.max()}]")
    print(f"RSRS Beta 信号值: {rsrs_signal.unique()}")
    
    # 验证信号格式
    assert isinstance(vwap_signal, pd.Series), "VWAP Distance 信号应为 pd.Series"
    assert isinstance(rsrs_signal, pd.Series), "RSRS Beta 信号应为 pd.Series"
    assert set(vwap_signal.unique()).issubset({-1, 0, 1}), "VWAP Distance 信号值应为 -1, 0, 1"
    assert set(rsrs_signal.unique()).issubset({-1, 0, 1}), "RSRS Beta 信号值应为 -1, 0, 1"
    
    print("\n✓ 策略兼容性测试通过")


if __name__ == "__main__":
    try:
        test_vwap_distance_strategy()
        test_rsrs_beta_strategy()
        test_strategy_compatibility()
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)