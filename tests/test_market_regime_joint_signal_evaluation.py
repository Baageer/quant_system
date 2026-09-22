import pandas as pd

from research.market_regime.joint_signal_evaluation import select_joint_signals


def test_joint_selection_requires_both_cross_sectional_ranks():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01"] * 4),
            "p_up": [0.9, 0.8, 0.2, 0.1],
            "p_tradable": [0.1, 0.8, 0.9, 0.2],
        }
    )

    selected = select_joint_signals(scores, p_up_top_k=0.5, p_tradable_top_k=0.5)

    assert selected.tolist() == [False, True, False, False]
