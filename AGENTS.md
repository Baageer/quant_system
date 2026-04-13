# AGENTS.md

## 项目定位

这是一个以 A 股为主的量化研究与回测仓库，当前主要服务于三类工作：

- 开发和调试择时策略
- 运行批量或组合信号回测
- 用 Streamlit 可视化行情、指标和回测结果

当前真正的主工作流集中在 `run_backtest.py`、`app_streamlit.py` 和 `app_backtest.py`。`main.py` 更像初始化骨架，不是日常开发入口。

## 推荐入口

- `run_backtest.py`
  - 通用回测 CLI 入口
  - 从 `config/strategies.yaml` 动态加载策略类
  - 支持单策略、多策略加权组合、止损止盈开关
- `app_streamlit.py`
  - 行情与技术指标可视化界面
  - 适合验证指标计算、观察 K 线与信号
- `app_backtest.py`
  - 回测结果可视化界面
  - 依赖 `output/` 目录中的 `backtest_*.csv` 和 `trades_*.csv`
- `main.py`
  - 仅保留系统初始化骨架
  - 不建议作为新增功能的首选入口

## 目录地图

- `backtest/`
  - 回测核心逻辑
  - `engine.py` 是组合净值、持仓和交易执行的核心
  - `metrics.py` / `performance.py` 负责绩效分析
- `signals/`
  - 信号与指标主目录
  - `indicators.py` 是技术指标基础库
  - `timing/` 放择时策略
  - `stop/` 放止损止盈策略
  - `signal_engine.py` 负责多信号组合
- `data/`
  - 数据接口与缓存
  - `data_api.py` 负责 AkShare/TuShare 数据读取
  - `HS300.txt`、`test1.txt` 等是股票列表样本
- `config/`
  - `settings.yaml` 放全局配置
  - `strategies.yaml` 是策略注册表和默认参数来源
- `portfolio/`
  - 仓位、调仓、组合优化模块
  - 当前主回测流程对它们的直接使用相对有限
- `factors/`
  - 因子框架预留
  - 当前主工作流以 `signals/` 为主
- `research/`
  - 实验脚本与原型代码
  - 新想法先放这里，稳定后再迁移到正式模块
- `tests/`
  - `pytest` 测试目录
  - 已覆盖 `indicators`、`ma_cross`、`macd`、`rsi`
- `output/`
  - 回测导出结果
  - 默认放交易记录和结果 CSV
- `logs/`
  - 运行日志

## 运行命令

在仓库根目录执行。

```powershell
pip install -r requirements.txt
python run_backtest.py --list
python run_backtest.py --strategy ma_cross --start 2023-01-01 --end 2023-12-31
python run_backtest.py --strategy ma_cross,rsi --signal-combination weighted --signal-weights 0.6,0.4
streamlit run app_streamlit.py
streamlit run app_backtest.py
pytest -q
```

## 策略扩展约定

`run_backtest.py` 通过 `config/strategies.yaml` 动态导入策略类，因此新增择时策略时请遵循下面的约定。

### 择时策略约定

- 文件通常放在 `signals/timing/`
- 类需要能被 `importlib.import_module()` 导入
- 类构造参数应能直接由 YAML 序列化
- 需要实现 `generate_signal(data: pd.DataFrame) -> pd.Series`
- 返回信号语义保持一致
  - `1` 表示做多或买入状态
  - `-1` 表示做空、卖出或反向状态
  - `0` 表示空仓或中性

### 数据输入约定

- 索引通常应是日期索引
- 最少需要 `close`
- 布林带、SuperTrend、ATR、KDJ 等逻辑通常还需要 `high`、`low`
- 某些可视化和交易逻辑默认还会使用 `open`、`volume`

### 新增策略的最小步骤

1. 在 `signals/timing/` 下实现策略模块和类。
2. 在 `config/strategies.yaml` 注册模块路径、类名、默认参数、`min_data_length`。
3. 在 `tests/` 新增对应测试文件。
4. 如果需要 UI 辅助展示，补充类似 `get_ma_values()`、`get_macd_values()` 这样的辅助读取方法。

## 开发时的实用判断

- 想改“指标公式”时，优先看 `signals/indicators.py`
- 想改“买卖信号逻辑”时，优先看 `signals/timing/*.py`
- 想改“组合回测行为”时，优先看 `backtest/engine.py`
- 想改“可选策略和默认参数”时，优先看 `config/strategies.yaml`
- 想改“股票池、手续费、滑点、初始资金”时，优先看 `config/settings.yaml`
- 想改“结果展示”时，优先看 `app_streamlit.py` 或 `app_backtest.py`

## 当前仓库的几个重要注意点

这些不是抽象建议，而是当前扫描仓库后确认存在的现实约束。

- `main.py` 目前更像骨架文件，不是主业务入口。
- `run_backtest.py` 内部目前有 `stock_list = stock_list[:1]`，默认会把股票池截成单只股票，做批量回测前要先确认这是不是有意为之。
- `run_backtest.py` 当前会导出 `trades_*.csv`，但 `results.to_csv(results_file)` 被注释掉了；这会导致 `app_backtest.py` 可能找不到成对的 `backtest_*.csv`。
- 若在 Windows PowerShell 里看到中文注释乱码，优先怀疑终端显示编码，不一定是源码本身损坏。

## 测试现状

截至 2026-04-13，在当前环境直接执行：

```powershell
pytest -q
```

结果不是全绿，现状为：

- `64 passed`
- `59 failed`

失败主要集中在两类问题：

- 多处代码和测试使用 `pd.isna`，当前环境下会抛出 `AttributeError: module 'pandas' has no attribute 'isna'`
- `rolling(...).apply(..., raw=True)` 在当前环境中触发 `TypeError`

这意味着：

- 当前仓库不能把“测试已通过”视作默认前提
- 修改指标与信号逻辑后，建议优先跑定向测试，而不是默认依赖全量测试通过

## 建议的工作方式

- 小实验先放 `research/`，验证稳定后再迁移到 `signals/` 或 `backtest/`
- 修改策略时，同步检查 `config/strategies.yaml`
- 修改指标时，同步检查 `tests/test_indicators.py`
- 修改回测导出逻辑时，同步检查 `app_backtest.py` 对文件名模式的假设
- 默认从仓库根目录运行脚本，很多相对路径都是按根目录写的

## 非源码目录说明

这些目录通常视为运行产物或缓存，不建议手工维护业务逻辑：

- `output/`
- `logs/`
- `.pytest_cache/`

## 与当前工作区相处的规则

- 当前仓库存在未跟踪文件，尤其是若干测试文件和额外回测脚本
- 没有明确需求时，不要删除或重置用户现有改动
- 生成新结果文件时，默认放在 `output/`
