"""Small reusable helpers for notebook 07: calendar windows and ensemble inference."""
import numpy as np


def recent_training_indices(df, indices, months):
    """Restrict already-purged training rows, without changing their historical features."""
    indices = np.asarray(indices)
    if months is None:
        return indices
    if months < 1:
        raise ValueError("months must be positive")
    last = df.iloc[indices].mes_rank.max()
    return indices[df.iloc[indices].mes_rank.to_numpy() >= last - months + 1]


class ProbabilityEnsemble:
    """Serializable average of raw-input pipelines; scores are not calibrated probabilities."""

    def __init__(self, models, weights=None):
        if not models:
            raise ValueError("At least one model is required")
        self.models = models
        weights = np.ones(len(models)) if weights is None else np.asarray(weights, dtype=float)
        if (len(weights) != len(models) or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("Invalid ensemble weights")
        self.weights = weights / weights.sum()
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probabilities = []
        for model in self.models:
            if not np.array_equal(model.classes_, self.classes_):
                raise ValueError("Model class order must be [0, 1]")
            probabilities.append(model.predict_proba(X))
        return np.average(np.stack(probabilities), axis=0, weights=self.weights)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
