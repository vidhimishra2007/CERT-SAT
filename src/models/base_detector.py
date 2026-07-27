"""
src/models/base_detector.py

Generic model interface for CERT-SAT anomaly detectors.

Every detector plugged into the prediction pipeline (Isolation Forest, LOF,
One-Class SVM, LSTM, Autoencoder, ...) conforms to `BaseDetector`. This lets
`src.models.multi_channel.MultiChannelDetector` train/score/save/load any of
them through one uniform code path instead of special-casing each algorithm.

BACKWARD COMPATIBILITY
-----------------------
`fit()`, `predict()`, and `score()` keep their original, model-native
signatures and return values unchanged (e.g. IsolationForestDetector.predict()
still returns sklearn's {-1, 1} labels) so none of the already-shipped
benchmarking/evaluation code (src/evaluation/metrics.py,
src/models/isolation_forest.py's run_isolation_forest, etc.) needs to change.

What's new is ADDITIVE, on top of that existing contract:
  - MODEL_TYPE / VERSION / SCORE_DIRECTION class metadata
  - is_anomaly()     : model's own predict(), normalized to a plain bool array
  - anomaly_score()  : model's own score(), normalized so HIGHER = more
                        anomalous, regardless of the underlying model's
                        native convention
  - get_metadata()   : small dict describing the fitted detector, used when
                        persisting model artifacts (see multi_channel.py)

See docs/MODEL_INTEGRATION_GUIDE.md for the full contract new models must
satisfy (required input/output shapes) and a step-by-step guide for adding
a new detector type (e.g. LSTM) to this pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np

# Valid values for SCORE_DIRECTION. Documents which direction of the
# subclass's OWN score() output means "more anomalous" -- this differs by
# algorithm family (e.g. sklearn IF/LOF: lower = more anomalous;
# reconstruction-error models like LSTM/Autoencoder: higher = more anomalous).
LOWER_IS_ANOMALOUS = "lower_is_anomalous"
HIGHER_IS_ANOMALOUS = "higher_is_anomalous"
_VALID_SCORE_DIRECTIONS = (LOWER_IS_ANOMALOUS, HIGHER_IS_ANOMALOUS)


class BaseDetector(ABC):
    """
    Common interface for single-channel anomaly detectors.

    Required methods (must be implemented by every subclass)
    -----------------------------------------------------------
    fit(X_train)          : train on one channel's nominal-only training
                             data. Must return `self`.
    predict(X_test)       : model-native anomaly predictions. Convention is
                             documented per-subclass (e.g. sklearn {-1, 1}).
    score(X_test)          : model-native continuous anomaly scores.
                             Direction (higher/lower = more anomalous) is
                             documented via SCORE_DIRECTION.
    is_anomaly(X_test)     : same decision as predict(), normalized to a
                             plain numpy bool array (True = anomaly). New
                             pipeline code should call this instead of
                             interpreting predict() directly.

    Required class attributes
    --------------------------
    MODEL_TYPE       : short string id, e.g. "isolation_forest", "lof",
                        "one_class_svm", "lstm". Used in saved-model metadata.
    VERSION          : version string for this detector implementation, e.g.
                        "1.0.0". Bump when fit/predict/score behavior changes
                        in a way that would make old saved artifacts stale.
    SCORE_DIRECTION  : one of LOWER_IS_ANOMALOUS / HIGHER_IS_ANOMALOUS --
                        which direction of score() means "more anomalous".

    Provided for free (do not override unless you need custom behavior)
    ---------------------------------------------------------------------
    anomaly_score(X_test) : score(X_test) normalized so HIGHER always means
                             more anomalous, using SCORE_DIRECTION.
    is_fitted              : bool property, True after fit() has run.
    get_metadata()          : dict describing this detector instance, used
                             when the manager persists the model artifact.
    """

    MODEL_TYPE: str = "base"
    VERSION: str = "1.0.0"
    SCORE_DIRECTION: str = HIGHER_IS_ANOMALOUS

    def __init__(self, ch_id: str):
        self.ch_id = ch_id
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Required contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_anomaly(self, X_test: np.ndarray) -> np.ndarray:
        """Return a plain bool ndarray, shape (n_windows,), True = anomaly."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Provided (normalized) helpers
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def anomaly_score(self, X_test: np.ndarray) -> np.ndarray:
        """
        score(X_test) normalized so HIGHER ALWAYS means more anomalous,
        regardless of the subclass's native SCORE_DIRECTION. Use this (not
        score()) in any new code that compares/ranks scores across detector
        types, e.g. an ensemble or a cross-model comparison report.
        """
        if self.SCORE_DIRECTION not in _VALID_SCORE_DIRECTIONS:
            raise ValueError(
                f"{type(self).__name__}.SCORE_DIRECTION must be one of "
                f"{_VALID_SCORE_DIRECTIONS}, got {self.SCORE_DIRECTION!r}"
            )
        raw = np.asarray(self.score(X_test), dtype=float)
        return -raw if self.SCORE_DIRECTION == LOWER_IS_ANOMALOUS else raw

    def get_metadata(self) -> dict:
        """Small dict describing this fitted detector, for artifact metadata."""
        return {
            "ch_id": self.ch_id,
            "model_type": self.MODEL_TYPE,
            "detector_class": type(self).__name__,
            "version": self.VERSION,
            "score_direction": self.SCORE_DIRECTION,
            "is_fitted": self.is_fitted,
        }

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"Detector for channel '{self.ch_id}' ({type(self).__name__}) "
                f"is not fitted yet. Call fit(X_train) first."
            )
