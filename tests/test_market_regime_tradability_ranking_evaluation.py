import pandas as pd

from research.market_regime.tradability_ranking_evaluation import ranking_quality, select_top_k


def make_scores():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01"] * 4 + ["2021-01-04"] * 4),
            "symbol": list("ABCDEFGH"),
            "p_tradable": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.3, 0.2],
            "tradability_target_10d": ["tradable_up", "not_tradable", "not_tradable", "not_tradable", "tradable_up", "not_tradable", "not_tradable", "not_tradable"],
            "sample_weight": [1.0] * 8,
        }
    )


def test_top_k_selects_each_daily_cross_section():
    scores = make_scores()
    selected = select_top_k(scores, 0.25)

    assert selected.tolist() == [True, False, False, False, True, False, False, False]


def test_top_k_accepts_an_explicit_score_column():
    scores = make_scores().assign(entry_score=[0.1, 0.9, 0.2, 0.3, 0.7, 0.6, 0.8, 0.2])

    selected = select_top_k(scores, 0.25, "entry_score")

    assert selected.tolist() == [False, True, False, False, False, False, True, False]


def test_ranking_quality_reports_weighted_precision_and_recall():
    scores = make_scores()
    quality = ranking_quality(scores, select_top_k(scores, 0.25))

    assert quality["selected_count"] == 2
    assert quality["weighted_precision"] == 1.0
    assert quality["weighted_recall"] == 1.0
