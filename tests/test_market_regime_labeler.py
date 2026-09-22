import numpy as np
import pandas as pd
import pytest

from research.market_regime import MarketRegimeLabeler, RegimeLabelerConfig, audit_price_data


def make_three_regime_prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=180)
    close = np.concatenate(
        [
            np.linspace(10.0, 18.0, 60),
            18.0 + 0.15 * np.sin(np.linspace(0, 6 * np.pi, 60)),
            np.linspace(18.0, 10.0, 60),
        ]
    )
    return pd.DataFrame({"close": close}, index=index)


def test_labeler_outputs_complete_offline_tables_for_clear_regimes():
    data = make_three_regime_prices()
    labeler = MarketRegimeLabeler(
        RegimeLabelerConfig(
                min_segment_length=15,
                min_return=0.04,
                volatility_multiplier=0.5,
            )
    )

    segments, daily, pivots = labeler.label(data, "000001")

    assert {"up", "sideways", "down"}.issubset(set(segments["regime"]))
    assert len(daily) == len(data)
    assert daily["date"].is_unique
    assert daily["segment_id"].nunique() == len(segments)
    assert segments["open_segment"].tolist()[-1] is True
    assert segments["open_segment"].iloc[:-1].eq(False).all()
    assert set(pivots["pivot_type"]).issubset({"top", "bottom"})
    confirmed = pivots.dropna(subset=["confirm_date"])
    assert (confirmed["confirm_date"] >= confirmed["pivot_date"]).all()


def test_labeler_rejects_invalid_close_and_date_inputs():
    labeler = MarketRegimeLabeler(RegimeLabelerConfig(min_segment_length=3))
    data = pd.DataFrame({"close": [10.0, -1.0, 11.0]}, index=pd.bdate_range("2024-01-02", periods=3))

    with pytest.raises(ValueError, match="正数"):
        labeler.label(data, "000001")

    unsorted = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0]},
        index=pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04"]),
    )
    with pytest.raises(ValueError, match="升序"):
        labeler.label(unsorted, "000001")


def test_audit_price_data_reports_quality_issues_without_mutating_data():
    data = pd.DataFrame(
        {"close": [10.0, np.nan, -2.0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-04"]),
    )

    report = audit_price_data(data)

    assert report["row_count"] == 3
    assert report["duplicate_date_count"] == 1
    assert report["missing_close_count"] == 1
    assert report["non_positive_close_count"] == 1
    assert report["is_date_sorted"] is True


def test_v11_keeps_flat_market_segments_within_maximum_length():
    data = pd.DataFrame({"close": np.repeat(10.0, 180)}, index=pd.bdate_range("2024-01-02", periods=180))
    labeler = MarketRegimeLabeler(
        RegimeLabelerConfig(min_segment_length=15, max_segment_length=45, smoothing_span=1)
    )

    segments, _, _ = labeler.label(data, "000001")

    assert set(segments["regime"]) == {"sideways"}
    assert segments["duration"].max() <= 45


def test_v11_pivot_confirmation_uses_first_reversal_threshold_crossing():
    prices = np.array([10.0, 10.5, 11.0, 10.8, 10.4, 10.2])

    confirmation = MarketRegimeLabeler._find_confirmation_position(prices, 2, "top", 0.05)

    assert confirmation == 4
