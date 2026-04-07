"""
量化交易系统主入口
"""
import yaml
from utils.logger import setup_logger
from data.data_api import DataAPI
from factors.factor_engine import FactorEngine
from signals.signal_engine import SignalEngine
from portfolio.optimizer import PortfolioOptimizer
from portfolio.position import PositionManager
from portfolio.rebalance import Rebalancer


def load_config(config_path: str = "./config/settings.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    logger = setup_logger()
    logger.info("启动量化交易系统")
    
    config = load_config()
    logger.info("配置加载完成")
    
    data_api = DataAPI(
        cache_dir=config['data']['cache_dir'],
        processed_dir=config['data']['processed_dir']
    )
    logger.info("数据接口初始化完成")
    
    factor_engine = FactorEngine()
    logger.info("因子引擎初始化完成")
    
    signal_engine = SignalEngine()
    logger.info("信号引擎初始化完成")
    
    optimizer = PortfolioOptimizer(method='mean_variance')
    position_manager = PositionManager(
        initial_capital=config['backtest']['initial_capital'],
        max_position_pct=config['strategy']['max_position_pct'],
        min_position_pct=config['strategy']['min_position_pct']
    )
    rebalancer = Rebalancer(rebalance_freq=config['strategy']['rebalance_freq'])
    logger.info("组合管理模块初始化完成")
    
    logger.info("系统启动完成，准备运行策略...")
    
    try:
        pass
    except Exception as e:
        logger.error(f"策略运行失败: {e}")
        raise
    
    logger.info("量化交易系统运行结束")


if __name__ == "__main__":
    main()
