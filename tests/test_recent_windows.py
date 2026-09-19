import numpy as np
import pandas as pd
import pytest

from scripts.recent_windows import ProbabilityEnsemble, recent_training_indices


def test_calendar_window_restricts_only_training():
    df = pd.DataFrame({'mes_rank': np.repeat(np.arange(1, 81), 2)})
    original = np.flatnonzero(df.mes_rank <= 50)
    for months in [24, 36, 48]:
        selected = recent_training_indices(df, original, months)
        assert df.iloc[selected].mes_rank.min() == 50 - months + 1
        assert df.iloc[selected].mes_rank.max() == 50
        assert len(selected) == 2 * months
    np.testing.assert_array_equal(recent_training_indices(df, original, None), original)


class DummyModel:
    classes_ = np.array([0, 1])

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def test_ensemble_normalizes_weights_and_preserves_class_order():
    model = ProbabilityEnsemble([DummyModel(.2), DummyModel(.8)], [1, 3])
    np.testing.assert_allclose(model.predict_proba([0, 1]), [[.35, .65], [.35, .65]])
    np.testing.assert_array_equal(model.predict([0, 1]), [1, 1])


@pytest.mark.parametrize('weights', [[-1, 2], [0, 0], [1], [np.nan, 1]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        ProbabilityEnsemble([DummyModel(.2), DummyModel(.8)], weights)
