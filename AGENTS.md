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

## Python 解释器（固定）

本仓库统一使用下面这个解释器，不要用裸 `python`：

```powershell
D:\ProgramData\miniconda3\envs\llm\python.exe        # PowerShell / cmd
```

```bash
/d/ProgramData/miniconda3/envs/llm/python.exe        # Git Bash
```

裸 `python` 在 Git Bash 下会解析到 `C:\Users\<用户>\AppData\Local\Microsoft\WindowsApps\python`（WindowsApps 存根，静默失败或返回退出码 49），base 环境也没有 pandas。

`.vscode/settings.json` 已与上面对齐（`python.defaultInterpreterPath` = `D:\ProgramData\miniconda3\envs\llm\python.exe`，2026-09-15 修正；此前指向不存在的 `D:\Anaconda3\envs\llm\python.exe`）。同时开启了 `python.testing.pytestEnabled`，因此仓库根的 `pytest.ini` 决定 VS Code 测试面板的发现范围——它把范围限定在 `tests/`，原因见「测试现状」。

该环境的依赖现状（2026-09-15 实测）：

- **`pytest` 9.1.1 已装**，`python -m pytest tests -q` 可跑（注意必须限定 `tests/`，原因见「测试现状」）。
- **已安装 `streamlit` 1.63.0**，`streamlit run` 可以启动；`plotly` 原本缺失，已补装 7.0.0（`requirements.txt:19` 声明的是 `plotly>=5.0.0`，应用在顶层 import 它）。
- 仓库根目录另有一个名为 `streamlit/` 的目录（内含 `industry_index_daily_monitor_app.py`）。**它不会遮蔽真包**：`site-packages/streamlit/` 是常规包（有 `__init__.py`），常规包优先于路径上更靠前的命名空间包片段，所以在仓库根 `import streamlit` 拿到的是真库，`streamlit.testing.v1.AppTest` 可用。只有在 streamlit 未安装时，`import streamlit` 才会解析到这个目录——看到 `streamlit.__file__ is None` 是「没装」的信号，不是「装坏了」的信号。

从仓库根调用模块（`-m` 形式依赖仓库根在 `sys.path` 上）：

```bash
/d/ProgramData/miniconda3/envs/llm/python.exe -m research.market_regime.index_valuation_diagnostic --run-id index_valuation_v1
```

## 运行命令

在仓库根目录执行。下面的 `python` 均指上一节固定的解释器。

```powershell
D:\ProgramData\miniconda3\envs\llm\python.exe -m pip install -r requirements.txt
python run_backtest.py --list
python run_backtest.py --strategy ma_cross --start 2023-01-01 --end 2023-12-31
python run_backtest.py --strategy ma_cross,rsi --signal-combination weighted --signal-weights 0.6,0.4
streamlit run app_streamlit.py
streamlit run app_backtest.py
```

```bash
/d/ProgramData/miniconda3/envs/llm/python.exe -m pytest tests -q     # 测试（必须限定 tests/）
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

**`pytest` 已装（9.1.1），用它。** 仓库自带的 `tests/run_tests.py` 请不要再当作全量测试用——它只覆盖 402 个里的 165 个（见下）。

```bash
<python.exe> -m pytest -q                                        # 全量：402 passed（2026-09-15）
<python.exe> -m pytest tests/test_strategy_loader.py -q          # 定向
<python.exe> -m pytest -q -p no:warnings                         # 关掉 pandas FutureWarning 噪声
```

**必须在仓库根目录执行。** 测试里有 `./config/settings.yaml` 这类相对 cwd 的路径，在 `tests/` 里跑会得到 17 个 `FileNotFoundError`（既有约定，不是坏测试）。

发现范围由仓库根的 `pytest.ini` 限定为 `testpaths = tests`。没有它的话，裸 `pytest` 会收进 `research/experiments/bollinger/test_bollinger.py`，而那个文件在**导入期**就 `os.listdir` 一个硬编码的 `J:\workspace\2_quant\quant_system\data\raw\akshare\price_history`，整轮收集中断、一个测试都跑不了。VS Code 测试面板受同一个设置约束。

### 为什么废弃 `tests/run_tests.py` 作全量入口

它是个零参数调用器，**只会调用模块级函数**，类内测试（`TestXxx`）一个都不跑。实测差距：

| 运行器 | 收集 | 结果 |
|---|---|---|
| `python -m pytest tests -q` | 402 | 402 passed |
| `python tests/run_tests.py` | 165 | `total 165 fails 6` |

192 个类内测试方法从未被它执行；并且它**掩盖过一个真实失败**——`test_build_timing_strategies` 断言 `ma_cross` 加载后是裸 `MACrossStrategy`，但 `config/strategies.yaml` 给 `ma_cross` 配了 `use_volume_filter: true`，加载器会把它包进 `FilteredTimingStrategy`。该断言是 common-filters 那次提交留下的陈述，已于 2026-09-15 修正。

它剩下的 6 个失败全是 `TypeError: ... missing 1 required positional argument: 'tmpdir'`（fixture 参数被零参调用），在 pytest 下全部通过。**这些失败不代表仓库有 6 个坏测试，只代表那个运行器不注入 fixture。**

- 新增测试请按 pytest 写，可以正常使用 `tmpdir` 等内置 fixture
- 如果仍要用 `tests/run_tests.py` 跑单个模块，则该模块的函数不能带 fixture 参数

## 建议的工作方式

- 小实验先放 `research/`，验证稳定后再迁移到 `signals/` 或 `backtest/`
- 修改策略时，同步检查 `config/strategies.yaml`
- 修改指标时，同步检查 `tests/test_indicators.py`
- 修改回测导出逻辑时，同步检查 `app_backtest.py` 对文件名模式的假设
- 新增或修改 Streamlit 页面时，使用 `width="stretch"` 或 `width="content"`，不要使用将被移除的 `use_container_width` 参数
- 改完 Streamlit 页面要冒烟时，不必开浏览器：`from streamlit.testing.v1 import AppTest` 可以无头跑整个脚本并遍历控件（`at.selectbox[i].set_value(...).run()`），异常、报错和渲染出的 `dataframe` / `plotly_chart` 都能读到。这是本项目发现「图表构造函数读了不存在的列」这类渲染期缺陷的唯一有效手段——语法检查和单测都盖不到
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
