import contextlib

import numpy as np
import pandas as pd

from research.market_regime.logistic_model import truncate_train_start


@contextlib.contextmanager
def raises(exception, match=None):
    """Minimal stand-in for ``pytest.raises`` -- pytest is not installed here."""

    try:
        yield
    except exception as exc:
        if match is not None and match not in str(exc):
            raise AssertionError("{} not found in {!r}".format(match, str(exc)))
    else:
        raise AssertionError("{} was not raised".format(exception.__name__))


def samples(dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "symbol": list("abcdefghij")[:len(dates)],
                         "value": range(len(dates))})


DATES = ["2014-10-16", "2014-10-17", "2014-10-18", "2019-01-02"]


def test_truncate_keeps_rows_on_and_after_the_boundary():
    """The boundary is inclusive: a run starting at the benchmark's first date
    must keep that date's rows, or the training set loses its first session."""

    kept = truncate_train_start(samples(DATES), "2014-10-17")

    assert kept["date"].dt.strftime("%Y-%m-%d").tolist() == ["2014-10-17", "2014-10-18", "2019-01-02"]


def test_truncate_with_no_boundary_returns_the_frame_untouched():
    original = samples(DATES)

    assert truncate_train_start(original, None) is original
    assert truncate_train_start(original, "") is original


def test_truncate_resets_the_index_so_downstream_positional_use_stays_valid():
    kept = truncate_train_start(samples(DATES), "2014-10-17")

    assert kept.index.tolist() == [0, 1, 2]


def test_truncate_drops_nothing_when_the_boundary_precedes_all_rows():
    original = samples(DATES)

    kept = truncate_train_start(original, "2005-01-01")

    assert len(kept) == len(original)


def test_truncate_rejects_a_boundary_that_empties_the_training_set():
    """Silently training on nothing would be worse than failing."""

    with raises(ValueError, match="leaves no training rows"):
        truncate_train_start(samples(DATES), "2030-01-01")


def test_truncate_reports_the_earliest_available_date_when_it_refuses():
    with raises(ValueError, match="2014-10-16"):
        truncate_train_start(samples(DATES), "2030-01-01")


def test_truncate_requires_a_date_column():
    with raises(ValueError, match="no date column"):
        truncate_train_start(pd.DataFrame({"symbol": ["a"], "value": [1]}), "2014-10-17")


def test_truncate_only_touches_the_training_split():
    """The contract that keeps arms comparable: validation and test must be
    scored over the same period regardless of training history."""

    train = samples(DATES)
    validation = samples(["2019-07-17", "2021-12-17"])
    test = samples(["2022-07-20", "2025-12-17"])

    truncate_train_start(train, "2014-10-17")

    assert len(validation) == 2 and len(test) == 2
    assert validation["date"].min() == pd.Timestamp("2019-07-17")
    assert test["date"].min() == pd.Timestamp("2022-07-20")


def test_truncate_is_idempotent():
    once = truncate_train_start(samples(DATES), "2014-10-17")
    twice = truncate_train_start(once, "2014-10-17")

    assert once["date"].tolist() == twice["date"].tolist()
    assert np.array_equal(once["value"].to_numpy(), twice["value"].to_numpy())
