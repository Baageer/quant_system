"""选股展示层的纯函数测试。

被测对象是 ``research.market_regime.stock_selection``——按仓库约定，UI 模块
（``app_stock_selection``）**不被测试导入**，所有能出错的计算都放在这个无 streamlit
依赖的模块里。页面本身靠 ``AppTest`` 无头遍历来盖。

测试分两层：合成数据的单元测试，加一组跑在真实产物上的集成断言（产物不存在时跳过，
因为它们是 D: 盘上的数据文件，不是仓库内容）。
"""

import pandas as pd
import pytest

from research.market_regime import stock_selection as ss


def _scores(rows):
    """``[(date, symbol, score, excess, predicted)]`` -> 打分帧。"""

    frame = pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "entry_score": score,
          "future_excess_return_10d": excess, "future_mae_10d": 0.0,
          "predicted_excess_return_10d": predicted, "predicted_mae_10d": 0.0}
         for d, s, score, excess, predicted in rows])
    return frame


class TestRebalanceCadence:
    def test_strikes_first_session_of_each_iso_week(self):
        # 2026-08-03 是周一；08-04..08-07 同周，必须被吃掉
        sessions = pd.to_datetime([
            "2026-08-03", "2026-08-04", "2026-08-07",
            "2026-08-10", "2026-08-11",
        ])
        assert ss.rebalance_dates(sessions) == [
            pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-10")]

    def test_week_grouping_is_by_iso_week_not_by_month(self):
        """跨月的一周只打一次——7 月的周五与 8 月的周一属于不同 ISO 周。"""

        sessions = pd.to_datetime(["2026-07-31", "2026-08-03", "2026-08-04"])
        assert ss.rebalance_dates(sessions) == [
            pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-03")]

    def test_week_whose_first_session_is_not_monday(self):
        """节后首日可能是周二——按「周一」过滤会静默丢掉整周。

        2026 春节假期吃掉 02-16（周一），该周首个交易日是 02-17（周二）。
        """

        sessions = pd.to_datetime([
            "2026-02-17", "2026-02-18",                 # 春节假期后的周内首日
            "2026-02-23", "2026-02-24",
        ])
        assert ss.rebalance_dates(sessions) == [
            pd.Timestamp("2026-02-17"), pd.Timestamp("2026-02-23")]

    def test_matches_generalised_weekday_cadence(self):
        """与生产通用实现 ``weekday_strike_dates(dates, 0)`` 钉在一起。"""

        from research.experiments.validation_null_phase_check import weekday_strike_dates

        sessions = pd.bdate_range("2026-01-01", "2026-03-31")
        assert ss.rebalance_dates(sessions) == sorted(weekday_strike_dates(sessions, 0))

    def test_single_session_week(self):
        sessions = pd.to_datetime(["2026-08-03", "2026-08-14"])
        assert ss.rebalance_dates(sessions) == [
            pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-14")]


class TestBuildPicks:
    def test_boundary_is_ceil_of_count_times_top_k(self):
        """名额用 ``ceil(n * top_k)``，与 ``select_top_k`` 同一规则。"""

        rows = [("2026-08-03", "{:06d}".format(i), 1.0 - i / 100.0, -0.01 + i / 100.0, 0.01)
                for i in range(10)]
        picks = ss.build_picks(_scores(rows), 0.05)
        # ceil(10 * 0.05) = 1
        assert len(picks) == 1
        assert picks["symbol"].tolist() == ["000000"]

    def test_rank_is_descending_within_date(self):
        rows = [("2026-08-03", "000001", 0.5, 0.0, 0.02),
                ("2026-08-03", "000002", 0.9, 0.0, 0.03),
                ("2026-08-03", "000003", 0.7, 0.0, 0.01)]
        picks = ss.build_picks(_scores(rows), 1.0)
        assert picks["symbol"].tolist() == ["000002", "000003", "000001"]
        assert picks["rank"].tolist() == [1, 2, 3]

    def test_dates_filter_restricts_to_requested_rebalances(self):
        rows = [("2026-08-03", "000001", 0.5, 0.0, 0.02),
                ("2026-08-04", "000002", 0.9, 0.0, 0.03)]
        picks = ss.build_picks(_scores(rows), 1.0, dates=[pd.Timestamp("2026-08-03")])
        assert picks["date"].unique().tolist() == [pd.Timestamp("2026-08-03")]

    def test_missing_label_marks_pick_open_instead_of_dropping_it(self):
        rows = [("2026-09-14", "000001", 0.5, float("nan"), 0.02),
                ("2026-09-14", "000002", 0.4, 0.01, 0.01)]
        picks = ss.build_picks(_scores(rows), 1.0)
        assert len(picks) == 2
        assert picks.set_index("symbol")["label_available"].to_dict() == {
            "000001": False, "000002": True}

    def test_empty_input_returns_empty_frame(self):
        picks = ss.build_picks(_scores([]), 0.05)
        assert picks.empty


class TestSummarizePicks:
    def _frame(self):
        # 池内 4 只，篮子是分最高的 2 只
        rows = [("2026-08-03", "000001", 0.9, 0.10, 0.05),
                ("2026-08-03", "000002", 0.8, 0.06, 0.04),
                ("2026-08-03", "000003", 0.2, -0.02, 0.01),
                ("2026-08-03", "000004", 0.1, -0.06, 0.00)]
        return _scores(rows)

    def test_edge_is_basket_minus_pool(self):
        scores = self._frame()
        picks = ss.build_picks(scores, 0.5)
        summary = ss.summarize_picks(picks, scores)
        row = summary.iloc[0]
        assert row["basket_excess"] == pytest.approx((0.10 + 0.06) / 2)
        assert row["pool_excess"] == pytest.approx((0.10 + 0.06 - 0.02 - 0.06) / 4)
        assert row["edge"] == pytest.approx(row["basket_excess"] - row["pool_excess"])

    def test_unlabelled_date_keeps_row_with_nan_not_dropped(self):
        """开放持仓是一个状态，不是缺口——行必须在，值为空。"""

        rows = [("2026-09-14", "000001", 0.9, float("nan"), 0.05),
                ("2026-09-14", "000002", 0.8, float("nan"), 0.04)]
        scores = _scores(rows)
        # top_k=1.0 才能把两只都选进来：2 只 × 0.5 的边界是 ceil(1.0)=1
        summary = ss.summarize_picks(ss.build_picks(scores, 1.0), scores)
        assert len(summary) == 1
        assert not summary.iloc[0]["label_available"]
        assert pd.isna(summary.iloc[0]["edge"])
        assert summary.iloc[0]["basket_symbols"] == 2


class TestParsePositions:
    def test_parses_dict_literal_and_zfills(self):
        assert ss.parse_positions("{'1': 100, '600519': 200}") == {
            "000001": 100, "600519": 200}

    def test_accepts_a_real_dict(self):
        assert ss.parse_positions({"600519": 200}) == {"600519": 200}

    def test_empty_book_is_empty_not_an_error(self):
        assert ss.parse_positions("{}") == {}
        assert ss.parse_positions(None) == {}
        assert ss.parse_positions(float("nan")) == {}

    def test_garbage_blank_row_rather_than_kill_the_page(self):
        assert ss.parse_positions("{not a dict") == {}
        assert ss.parse_positions("[1, 2, 3]") == {}

    def test_non_numeric_share_count_is_skipped(self):
        assert ss.parse_positions("{'600519': 'many', '000001': 100}") == {"000001": 100}


class TestBookAsOf:
    def _equity(self):
        frame = pd.DataFrame(
            {"portfolio_value": [100.0, 101.0, 102.0, 103.0],
             "cash": [100.0, 50.0, 50.0, 50.0],
             "positions": ["{}", "{'600519': 100}", "{}", "{'000001': 200}"]},
            index=pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]))
        return frame

    def test_returns_book_at_as_of(self):
        book = ss.book_as_of(self._equity(), "2026-08-04")
        assert book["symbol"].tolist() == ["600519"]
        assert book["shares"].tolist() == [100]

    def test_falls_back_to_most_recent_non_empty_book(self):
        """窗口结束时可能是空仓日，不能因此显示「没有持仓」。"""

        book = ss.book_as_of(self._equity(), "2026-08-05")
        assert book["symbol"].tolist() == ["600519"]
        assert book["date"].iloc[0] == pd.Timestamp("2026-08-04")

    def test_empty_equity_is_empty_book(self):
        assert ss.book_as_of(pd.DataFrame(), None).empty


class TestBookWithValues:
    def _book(self):
        return pd.DataFrame({"date": pd.Timestamp("2026-08-06"),
                             "symbol": ["600519", "000001"],
                             "shares": [100, 200]})

    def test_weights_are_share_of_portfolio_value(self):
        valued = ss.book_with_values(self._book(), {"600519": 10.0, "000001": 5.0},
                                     portfolio_value=2000.0)
        assert valued["market_value"].tolist() == [1000.0, 1000.0]
        assert valued["weight"].tolist() == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_missing_quote_keeps_the_row(self):
        """行情缓存里没有的持仓是事实，不该从表里消失。"""

        valued = ss.book_with_values(self._book(), {"600519": 10.0},
                                     portfolio_value=2000.0)
        assert len(valued) == 2
        assert pd.isna(valued.set_index("symbol").loc["000001", "market_value"])

    def test_empty_book_gains_the_columns(self):
        valued = ss.book_with_values(pd.DataFrame(), {})
        assert valued.empty
        assert {"name", "market_value", "weight"} <= set(valued.columns)


class TestArtifactReaders:
    def test_missing_manifest_is_empty_dict(self, tmp_path):
        assert ss.load_selection_manifest(tmp_path) == {}

    def test_manifest_pool_absent_is_empty_dict(self):
        assert ss.manifest_pool({"pools": [{"pool": "hs300"}]}, "zz1000") == {}

    def test_extension_trustworthy_requires_an_explicit_pass(self):
        assert ss.extension_trustworthy({"extension_check": {"status": "PASS"}})
        assert not ss.extension_trustworthy({"extension_check": {"status": "FAIL"}})
        # 缺 manifest 不是通过——那正是会误引一个未审计数字的状态
        assert not ss.extension_trustworthy({})

    def test_missing_artifacts_degrade_to_empty_frames(self, tmp_path):
        frames = ss.load_selection_artifacts("hs300", tmp_path, {})
        assert set(frames) == {"signals", "summary", "equity", "trades", "holdings"}
        assert all(frame.empty for frame in frames.values())

    def test_artifact_paths_are_pool_scoped(self, tmp_path):
        paths = ss.selection_artifact_paths(tmp_path, "hs300")
        assert all("hs300" in path.name for path in paths.values())

    def test_load_symbol_names_missing_file_is_empty_dict(self, tmp_path):
        assert ss.load_symbol_names(tmp_path / "nope.csv") == {}

    def test_attach_names_falls_back_to_the_code(self):
        frame = pd.DataFrame({"symbol": ["600519", "999999"]})
        out = ss.attach_names(frame, {"600519": "贵州茅台"})
        # 查不到的标的显示代码本身，而不是 NaN
        assert out["name"].tolist() == ["贵州茅台", "999999"]


class TestDraftSummary:
    def test_splits_open_from_graded(self):
        rows = [("2026-08-03", "000001", 0.9, 0.05, 0.02),
                ("2026-09-14", "000002", 0.8, float("nan"), 0.02)]
        picks = ss.build_picks(_scores(rows), 1.0)
        open_, graded = ss.draft_summary(picks)
        assert open_["symbol"].tolist() == ["000002"]
        assert graded["symbol"].tolist() == ["000001"]


# --- integration against the real artifacts ---------------------------------
#
# These read what ``build_stock_selection_2026`` wrote.  The artifacts live on
# the D: volume, not in the checkout, so a fresh machine skips them rather than
# failing.

_ARTIFACTS = ss.selection_artifact_paths(ss.DEFAULT_DATASET_DIR, "hs300")["signals"]
_MANIFEST = ss.FORWARD_ROOT / "stock_selection_manifest.json"

needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS.exists(), reason="选股产物未生成（先跑 build_stock_selection_2026）")


@needs_artifacts
class TestRealArtifacts:
    POOLS = {"hs300": 11, "zz1000": 90}

    @pytest.mark.parametrize("pool", ["hs300", "zz1000"])
    def test_basket_width_matches_published_top_k(self, pool):
        """每期篮子宽度 = ceil(池内只数 × top_k)，用已发布的 top_k。"""

        picks = ss.load_selection_artifacts(pool, None, {})["signals"]
        counts = picks.groupby("date").size()
        assert set(counts.unique()) == {self.POOLS[pool]}
        assert len(counts) == 7

    @pytest.mark.parametrize("pool", ["hs300", "zz1000"])
    def test_extension_check_passed(self, pool):
        """组合数字只有在扩展一致性检查通过时才可引用。"""

        entry = ss.manifest_pool(ss.load_selection_manifest(), pool)
        assert ss.extension_trustworthy(entry), entry.get("extension_check")

    @pytest.mark.parametrize("pool", ["hs300", "zz1000"])
    def test_last_two_rebalances_are_still_open(self, pool):
        """窗口以开放持仓收尾——这是状态，必须被如实展示。"""

        picks = ss.load_selection_artifacts(pool, None, {})["signals"]
        tail = picks.loc[picks["date"] >= pd.Timestamp("2026-09-07")]
        assert not tail.empty
        assert not tail["label_available"].any()

    @pytest.mark.parametrize("pool", ["hs300", "zz1000"])
    def test_book_at_window_end_is_not_empty(self, pool):
        equity = ss.load_selection_artifacts(pool, None, {})["equity"]
        book = ss.book_as_of(equity, pd.Timestamp("2026-09-15"))
        assert not book.empty
        assert book["symbol"].is_unique
