"""当期选股链路的测试：会话缓存、窗口端点、产物读取。

覆盖 ``fetch_tushare_forward_2026``、``splice_forward_2026_prices`` 与
``stock_selection`` 的当日读取部分。都不联网：网络路径靠抓取脚本自己的重试与
manifest 记录，测试只钉纯逻辑与磁盘状态。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.experiments import score_forward_2026 as sf

from research.experiments.fetch_tushare_forward_2026 import (
    cached_session_rows,
    parse_args as fetch_parse_args,
)
from research.experiments.splice_forward_2026_prices import (
    FIRST_SESSION,
    forward_sessions,
    latest_published_session,
    parse_args as splice_parse_args,
)
from research.market_regime import stock_selection as ss


def _session_file(directory, day, rows):
    """写一个 ``daily/{day}.csv``；``rows=0`` 就是只有表头的未发布形态。"""

    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"ts_code": ["000001.SZ"] * rows,
                          "trade_date": [day] * rows,
                          "close": [10.0] * rows})
    frame.to_csv(directory / "{}.csv".format(day), index=False)


class TestSessionCache:
    """会话文件缓存的坑：空文件被当成有效缓存，于是那次会话永远不再重抓。"""

    def test_counts_data_rows_not_the_header(self, tmp_path):
        path = tmp_path / "20260917.csv"
        _session_file(tmp_path, "20260916", 3)
        assert cached_session_rows(tmp_path / "20260916.csv") == 3
        _session_file(tmp_path, "20260917", 0)
        assert cached_session_rows(tmp_path / "20260917.csv") == 0
        assert not path.exists() or cached_session_rows(path) == 0

    def test_missing_file_is_zero_rather_than_an_error(self, tmp_path):
        assert cached_session_rows(tmp_path / "nope.csv") == 0

    def test_unreadable_file_is_zero_rather_than_an_error(self, tmp_path):
        # 内容是二进制垃圾：读取应降级为 0，而不是把整轮抓取掀翻
        path = tmp_path / "20260918.csv"
        path.write_bytes(b"\xff\xfe\x00\x01\x02")
        assert cached_session_rows(path) == 0


class TestWindowEndpoint:
    def test_latest_published_session_ignores_header_only_files(self, tmp_path):
        """未发布的会话在磁盘上就是「只有表头」，不能被当成窗口末点。"""

        daily = tmp_path / "daily"
        for day in ("20260915", "20260916"):
            _session_file(daily, day, 2)
        _session_file(daily, "20260917", 0)  # 已抓取但 tushare 未发布
        assert latest_published_session(tmp_path) == "20260916"

    def test_latest_published_session_advances_once_rows_appear(self, tmp_path):
        """同一天补上数据后，窗口末点必须前移——这正是原先的静默失败。"""

        daily = tmp_path / "daily"
        _session_file(daily, "20260916", 2)
        _session_file(daily, "20260917", 0)
        assert latest_published_session(tmp_path) == "20260916"
        _session_file(daily, "20260917", 5)
        assert latest_published_session(tmp_path) == "20260917"

    def test_latest_published_session_is_none_when_nothing_usable(self, tmp_path):
        assert latest_published_session(tmp_path) is None
        _session_file(tmp_path / "daily", "20260917", 0)
        assert latest_published_session(tmp_path) is None

    def test_forward_sessions_respects_the_resolved_endpoint(self, tmp_path):
        daily = tmp_path / "daily"
        for day in ("20260915", "20260916", "20260917"):
            _session_file(daily, day, 2)
        assert forward_sessions(tmp_path, "20260916") == ["20260915", "20260916"]
        assert forward_sessions(tmp_path, "20260917") == [
            "20260915", "20260916", "20260917"]

    def test_forward_sessions_never_precedes_the_fixed_start(self, tmp_path):
        daily = tmp_path / "daily"
        _session_file(daily, "20251231", 2)  # 锚点会话，早于窗口
        _session_file(daily, FIRST_SESSION, 2)
        assert forward_sessions(tmp_path, "20260105") == [FIRST_SESSION]

    def test_non_session_filenames_are_ignored(self, tmp_path):
        daily = tmp_path / "daily"
        _session_file(daily, "20260916", 2)
        _session_file(daily, "readme", 2)
        _session_file(daily, "2026091", 2)
        assert latest_published_session(tmp_path) == "20260916"


class _PerRowRegressor:
    """Stands in for a frozen Ridge: ``predict`` returns a per-row function.

    Reading the first feature column makes the output genuinely row-dependent,
    so a positional misalignment shows up as a wrong value rather than passing by
    luck on a constant.
    """

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        return features[:, 0] * 0.01


class TestRawPredictionsAlignWithRows:
    """裸预测列必须与同一行的 ``_10d`` 列相等——它们是同一个数组。

    ``build_outcome_predictions`` 返回的帧按 ``(date, symbol)`` 排过序，而
    ``predict`` 的输出是 ``samples`` 的顺序。样本文件是**逐批拼接**的，全局并非
    有序，所以按位置贴回去会把两个裸列**贴到错行上**，且窗口一变长、排序置换就变，
    错法也跟着变 —— 一个会在每次延长窗口时"悄悄换一批错值"的 bug。这两个列目前
    没有下游读者（生产链路只读 ``entry_score`` 与 ``_10d`` 列），但躺在已发布产物
    里、名字又起得跟真的一样，迟早会被当成真的引用。
    """

    def _samples(self):
        # 故意乱序：拼接产物就是这个形态
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-09-17", "2026-09-16", "2026-09-16",
                                    "2026-09-17", "2026-09-16", "2026-09-17"]),
            "symbol": ["600519", "000002", "000001", "000001", "600519", "000002"],
            "feature_a": [5.0, 1.0, 3.0, 2.0, 4.0, 6.0],
            "future_excess_return_10d": [np.nan] * 6,
            "future_mae_10d": [np.nan] * 6,
        })

    def _model(self):
        return {
            "preprocessor": {
                "feature_columns": ["feature_a"],
                "medians": [0.0], "means": [0.0], "scales": [1.0],
            },
            "excess_model": _PerRowRegressor(),
            "mae_model": _PerRowRegressor(),
        }

    def test_bare_columns_equal_their_10d_twins(self):
        scored = sf.apply_model(self._samples(), self._model(), 0.5, "unit_test")
        assert scored["predicted_excess_return"].to_numpy() == pytest.approx(
            scored["predicted_excess_return_10d"].to_numpy())
        assert scored["predicted_mae"].to_numpy() == pytest.approx(
            scored["predicted_mae_10d"].to_numpy())

    def test_output_is_sorted_and_per_row_values_are_right(self):
        samples = self._samples()
        scored = sf.apply_model(samples, self._model(), 0.5, "unit_test")
        keys = list(zip(scored["date"], scored["symbol"]))
        assert keys == sorted(keys), "输出没有按 (date, symbol) 排序"

        # 预测是 feature_a * 0.01，必须落在**同一个 (date, symbol)** 上
        expected = samples.loc[:, ["date", "symbol"]].assign(
            expected=pd.to_numeric(samples["feature_a"]) * 0.01)
        merged = scored.merge(expected, on=["date", "symbol"], how="left")
        assert len(merged) == len(samples), "键不唯一，对不上行"
        assert merged["predicted_excess_return_10d"].to_numpy() == pytest.approx(
            merged["expected"].to_numpy())

    def test_row_order_of_the_input_does_not_matter(self):
        shuffled = self._samples().sample(frac=1.0, random_state=7).reset_index(drop=True)
        a = sf.apply_model(self._samples(), self._model(), 0.5, "unit_test")
        b = sf.apply_model(shuffled, self._model(), 0.5, "unit_test")
        pd.testing.assert_frame_equal(a, b)


class TestNoPinnedWindowEnd:
    """窗口末点不许是写死的日期。

    两个脚本都栽过同一个坑：常量在写下的当天是对的，第二天就悄悄过期，而失败
    形态是「照常成功、只是停在上一天」。所以末点必须自动检测：抓取侧默认今天
    （``trade_cal`` 只会给已开市的交易日，未发布的今日会落在 0 行文件里，而
    :func:`cached_session_rows` 保证下一轮会补），拼接侧默认取 ``daily/`` 里
    最新的非空会话。
    """

    def test_fetch_defaults_to_today_not_a_literal(self):
        assert fetch_parse_args([]).end_date is None

    def test_fetch_still_honours_an_explicit_end_date(self):
        assert fetch_parse_args(["--end-date", "2026-01-30"]).end_date == "2026-01-30"

    def test_splice_defaults_to_autodetect(self):
        assert splice_parse_args([]).last_session is None

    def test_splice_still_honours_an_explicit_end_session(self):
        assert splice_parse_args(["--last-session", "20260130"]).last_session == "20260130"


class TestDailySelectionReaders:
    def _write(self, output_dir, forward_root, picks, manifest=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        picks.to_csv(output_dir / "daily_selection.csv", index=False, encoding="utf-8-sig")
        if manifest is not None:
            forward_root.mkdir(parents=True, exist_ok=True)
            (forward_root / "daily_selection_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    def _picks(self):
        return pd.DataFrame({
            "pool": ["hs300", "hs300", "zz1000"],
            "date": pd.to_datetime(["2026-09-17"] * 3),
            "symbol": ["600519", "1", "000002"],
            "rank": [1, 2, 1],
            "entry_score": [0.01, 0.005, 0.02],
            "predicted_excess": [0.03, 0.02, 0.04],
            "predicted_mae": [-0.05, -0.04, -0.06],
            "entry_date": ["2026-09-18"] * 3,
            "signal_close": [1500.0, 11.6, 3.02],
        })

    def test_missing_artifact_degrades_to_empty(self, tmp_path):
        payload = ss.load_daily_selection(tmp_path / "out", tmp_path / "fwd")
        assert payload["picks"].empty
        assert payload["manifest"] == {}

    def test_reads_picks_and_normalises_the_symbol(self, tmp_path):
        output, forward = tmp_path / "out", tmp_path / "fwd"
        self._write(output, forward, self._picks(), {"session": "2026-09-17"})
        payload = ss.load_daily_selection(output, forward)
        # "1" 必须补成 "000001"，否则名称映射与行情查找都会落空
        assert set(payload["picks"]["symbol"]) == {"600519", "000001", "000002"}
        assert payload["manifest"]["session"] == "2026-09-17"

    def test_corrupt_manifest_does_not_take_down_the_picks(self, tmp_path):
        output, forward = tmp_path / "out", tmp_path / "fwd"
        self._write(output, forward, self._picks())
        forward.mkdir(parents=True, exist_ok=True)
        (forward / "daily_selection_manifest.json").write_text("{not json", encoding="utf-8")
        payload = ss.load_daily_selection(output, forward)
        assert not payload["picks"].empty
        assert payload["manifest"] == {}

    def test_pool_entry_lookup(self):
        manifest = {"pools": [{"pool": "hs300", "top_k": 0.05}]}
        assert ss.daily_pool_entry(manifest, "hs300")["top_k"] == 0.05
        assert ss.daily_pool_entry(manifest, "zz1000") == {}

    def test_artifact_paths_are_distinct_per_caller(self, tmp_path):
        paths = ss.daily_selection_paths(tmp_path / "out", tmp_path / "fwd")
        assert paths["picks"].name == "daily_selection.csv"
        assert paths["manifest"].name == "daily_selection_manifest.json"
        assert paths["picks"] != paths["manifest"]


# --- integration against the real artifacts ---------------------------------
#
# Extension-invariance -- "appending a session must not rewrite earlier rows" --
# is deliberately *not* tested here.  It is already covered at the unit level by
# ``test_causal_panel_is_unchanged_when_future_rows_change`` in
# ``test_market_regime_feature_event_study.py``, on synthetic data, in
# milliseconds.  Re-asserting it against the real forward tree costs ~16s and
# tests nothing extra, because ``build_causal_feature_panel`` computes every
# factor per symbol from trailing rows only.  (Verified once by hand on
# 2026-09-18: 2 symbols x 3 session cuts, max |delta| = 0.000e+00 over all 80
# feature columns.  Recorded in docs/research/stock_selection_2026.md §5.2.)

_DAILY = ss.daily_selection_paths(ss.DEFAULT_DATASET_DIR, ss.FORWARD_ROOT)

needs_daily = pytest.mark.skipif(
    not _DAILY["picks"].exists(),
    reason="当日选股产物未生成（先跑 daily_selection_2026）")


@needs_daily
class TestRealDailyArtifact:
    def _payload(self):
        return ss.load_daily_selection()

    def test_every_pick_is_unlabelled(self):
        """最新交易日的选股不可能有已实现收益——有就是 bug。"""

        picks = self._payload()["picks"]
        assert "label_available" in picks.columns
        assert not picks["label_available"].any()

    def test_picks_live_on_the_manifest_session(self):
        payload = self._payload()
        picks, manifest = payload["picks"], payload["manifest"]
        assert set(pd.to_datetime(picks["date"]).unique()) == {
            pd.Timestamp(manifest["session"])}

    def test_basket_width_matches_published_top_k(self):
        payload = self._payload()
        picks, manifest = payload["picks"], payload["manifest"]
        for entry in manifest.get("pools", []):
            block = picks.loc[picks["pool"] == entry["pool"]]
            assert len(block) == entry["basket"]
            # 名额规则是 ceil(n × top_k)，用已发布的 top_k
            import math
            assert entry["basket"] == math.ceil(entry["cross_section"] * entry["top_k"])

    def test_entry_date_is_strictly_after_the_signal_day(self):
        payload = self._payload()
        manifest = payload["manifest"]
        if not manifest.get("entry_date"):
            pytest.skip("trade_cal 不可达，entry_date 未解析")
        assert pd.Timestamp(manifest["entry_date"]) > pd.Timestamp(manifest["session"])

    def test_signal_closes_are_populated_for_the_hs300_basket(self):
        """名称与信号日收盘都应取到；取不到说明拼价树与打分表对不上。"""

        picks = self._payload()["picks"]
        block = picks.loc[picks["pool"] == "hs300"]
        assert block["signal_close"].notna().all()
        assert (block["signal_close"] > 0).all()


# --- the page itself --------------------------------------------------------
#
# The repo rule is that UI modules are never imported by tests; the page is
# covered by driving it headlessly instead.  Everything else here tests the
# logic behind the page, which cannot catch a defect that only exists while
# rendering.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_APP = PROJECT_ROOT / "research/market_regime/app_stock_selection_live.py"
REVIEW_APP = PROJECT_ROOT / "research/market_regime/app_stock_selection.py"


def _app_test(path):
    pytest.importorskip("streamlit", reason="streamlit 未安装")
    from streamlit.testing.v1 import AppTest
    # Absolute: ``from_file`` resolves relative paths against the *calling* file
    # (this one, i.e. tests/), not the working directory.
    return AppTest.from_file(str(path), default_timeout=120)


@pytest.mark.parametrize("path", [LIVE_APP, REVIEW_APP])
class TestPagesRender:
    """两个页面都必须能无头跑完，且不抛异常。"""

    def test_first_render_raises_nothing(self, path):
        app = _app_test(path)
        app.run()
        assert not app.exception, [str(e) for e in app.exception]

    def test_title_is_set(self, path):
        app = _app_test(path)
        app.run()
        if app.exception:
            pytest.skip("页面依赖的产物不可用")
        assert app.title or app.subheader, "页面没有渲染出任何标题"


@needs_daily
class TestLivePageAgainstRealArtifacts:
    """当日页的控件必须真的能切换。

    ``AppTest`` 的坑：带 ``format_func`` 的控件里，``options`` 给的是**格式化后**的
    标签（``沪深300（hs300）``），而 ``value`` / ``set_value`` 用的是**原始值**
    （``hs300``）。拿 ``options`` 去 ``set_value`` 不会报错，只会把一个双重格式化的
    值塞进去、页面照常渲染 —— 也就是说那种写法**看起来通过了、其实没测到东西**。
    所以这里一律用 ``POOLS`` 的原始键驱动，并断言渲染出来的确实是那一个池。
    """

    def _open(self):
        app = _app_test(LIVE_APP)
        app.run()
        assert not app.exception, [str(e) for e in app.exception]
        assert app.radio, "没有股票池控件"
        return app

    def test_radio_options_are_formatted_labels(self):
        app = self._open()
        labels = list(app.radio[0].options)
        assert len(labels) == 2, labels
        assert all("（" in label and "）" in label for label in labels), labels
        # 原始值仍是 POOLS 的键
        assert app.radio[0].value in ss.POOLS

    def test_switching_pool_rerenders_that_pools_basket(self):
        app = self._open()
        for pool in ss.POOLS:
            app.radio[0].set_value(pool).run()
            assert not app.exception, "切到 {} 时崩溃: {}".format(
                pool, [str(e) for e in app.exception])
            assert app.radio[0].value == pool, "控件没有切到 {}".format(pool)
            titles = [str(s.value) for s in app.subheader]
            expected = "{} · 当日名单".format(ss.POOLS[pool]["index_name"])
            assert expected in titles, "{} 没有渲染出 {}，实际 {}".format(
                pool, expected, titles)

    def test_freshness_metrics_are_shown(self):
        """信号日/拟成交日/数据截至/生成时间——陈旧与否必须可见。"""

        app = self._open()
        labels = [m.label for m in app.metric]
        assert "信号日（T）" in labels
        assert "拟成交日（T+1）" in labels
        assert "行情数据截至" in labels

    def test_no_realized_return_is_ever_displayed(self):
        """这批选股不可能有已实现收益；页面若显示一个就是 bug。"""

        app = self._open()
        payload = ss.load_daily_selection()
        picks = payload["picks"]
        assert not picks["realized_excess"].notna().any()
        assert not picks["label_available"].any()
        # 页面上也不该出现「已实现」字样
        for frame in app.dataframe:
            columns = list(getattr(frame.value, "columns", []))
            assert not any("已实现" in str(c) for c in columns), columns
