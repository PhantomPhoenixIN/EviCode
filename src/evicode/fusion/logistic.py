"""Evidence fusion models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticEvidenceFusion:
    """Logistic-regression fusion over evidence vectors."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the fusion model."""
        self.model = LogisticRegression(random_state=seed, max_iter=1000)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the fusion model."""
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict semantic-correctness probabilities."""
        return self.model.predict_proba(features)[:, 1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        return self.model.predict(features)
