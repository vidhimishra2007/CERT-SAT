"""
src/models/multi_channel.py

Generic multi-channel prediction pipeline for CERT-SAT.

MultiChannelIsolationForest (src/models/isolation_forest.py) is the model
currently deployed to the backend and is left untouched so nothing already
integrated by Rupali breaks. `MultiChannelDetector` here is the same idea,
generalized to work with ANY BaseDetector subclass -- Isolation Forest, LOF,
One-Class SVM, LSTM, Autoencoder, whatever comes next -- through one code
path, so new models don't need a new hand-written manager class.

Usage
-----
    from src.models.multi_channel import MultiChannelDetector
    from src.models.one_class_svm import OneClassSVMDetector

    manager = MultiChannelDetector(OneClassSVMDetector)
    manager.fit_channel("P-1", X_train, contamination=0.05)
    manager.scalers["P-1"] = scaler        # persist for live inference
    result = manager.predict("P-1", X_test)
    manager.save("models_saved/ocsvm_all_channels.joblib")

    # later, anywhere (backend, notebook, ...):
    manager = MultiChannelDetector.load("models_saved/ocsvm_all_channels.joblib")
    result = manager.predict_raw_window("P-1", raw_buffer)

See docs/MODEL_INTEGRATION_GUIDE.md for the full input/output contract and
a step-by-step guide for plugging in a new model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, Union

import joblib
import numpy as np

from src.models.base_detector import BaseDetector
from src.models.versioning import build_artifact_metadata, check_interface_version
from src.utils.config import WINDOW_SIZE


class MultiChannelDetector:
    """
    Owns one BaseDetector-conforming detector per channel, and exposes a
    single, model-agnostic predict(ch_id, X) / predict_raw_window(ch_id,
    raw_buffer) call -- the caller never needs to know or care which
    algorithm is behind a given saved artifact.

    IMPORTANT -- input to predict() must already be windowed + scaled the
    same way training data was (see src.data.preprocessing.scale_channel_
    with_scaler and src.data.windowing.create_windows_stats, or pass raw
    data to predict_raw_window() instead, which does this for you using
    the channel's saved scaler).
    """

    def __init__(self, detector_cls: Type[BaseDetector],
                 window_size: int = WINDOW_SIZE):
        """
        Parameters
        ----------
        detector_cls : a BaseDetector subclass, e.g. IsolationForestDetector,
            LOFDetector, OneClassSVMDetector, or a new model you've added.
            NOT an instance -- this class is instantiated once per channel
            inside fit_channel().
        window_size : must match the window size used to build the features
            passed to fit_channel()/predict(). Only used for metadata and by
            predict_raw_window(); does not affect fit()/predict() directly.
        """
        if not (isinstance(detector_cls, type) and issubclass(detector_cls, BaseDetector)):
            raise TypeError(
                f"detector_cls must be a subclass of BaseDetector, got {detector_cls!r}. "
                f"See docs/MODEL_INTEGRATION_GUIDE.md for how to implement one."
            )
        self.detector_cls = detector_cls
        self.model_type = detector_cls.MODEL_TYPE
        self.window_size = window_size
        self.detectors: dict[str, BaseDetector] = {}
        self.scalers: dict = {}
        self.meta: dict = {}

    def fit_channel(self, ch_id: str, X_train: np.ndarray, **detector_kwargs) -> BaseDetector:
        """
        Instantiate and fit one detector for `ch_id`, using this manager's
        detector_cls. Extra keyword args (e.g. contamination=0.05,
        n_neighbors=20) are passed straight through to the detector's
        __init__ -- see that class's constructor for what it accepts.
        """
        det = self.detector_cls(ch_id, **detector_kwargs)
        det.fit(X_train)
        self.detectors[ch_id] = det
        return det

    def predict(self, ch_id: str, X_test: np.ndarray) -> dict:
        """
        Standardized prediction entrypoint. Works identically no matter
        which BaseDetector subclass is behind this manager.

        Parameters
        ----------
        ch_id : channel id, e.g. "P-1" -- must have been fit already.
        X_test : ndarray, shape (n_windows, n_features)
            Windowed, scaled features -- same shape/preprocessing as
            X_train passed to fit_channel().

        Returns
        -------
        dict:
            "ch_id"          : str
            "model_type"     : str, e.g. "one_class_svm"
            "is_anomaly"     : bool[], shape (n_windows,) -- True = anomaly
            "anomaly_score"  : float[], shape (n_windows,) -- HIGHER = more
                                anomalous, ALWAYS, regardless of model type
                                (see BaseDetector.anomaly_score)
            "n_samples"      : int
        """
        if ch_id not in self.detectors:
            raise KeyError(
                f"No trained '{self.model_type}' model for channel '{ch_id}'. "
                f"Trained channels: {list(self.detectors)}"
            )
        det = self.detectors[ch_id]
        return {
            "ch_id": ch_id,
            "model_type": self.model_type,
            "is_anomaly": det.is_anomaly(X_test),
            "anomaly_score": det.anomaly_score(X_test),
            "n_samples": int(np.asarray(X_test).shape[0]),
        }

    def predict_raw_window(self, ch_id: str, raw_buffer: np.ndarray) -> dict:
        """
        Score ONE raw (unscaled) rolling window directly -- scales and
        windows it using this channel's saved scaler first, then predicts.
        This is what a live backend integration should call per packet
        batch, rather than reimplementing scaling/windowing itself.

        Parameters
        ----------
        ch_id : channel id, e.g. "P-1" -- must have been fit already.
        raw_buffer : ndarray, shape (window_size, n_raw_features)
            The most recent `window_size` RAW (unscaled) telemetry
            timesteps for this channel, chronological order.
        """
        # Local import avoids a circular import (live_window -> this module).
        from src.inference.live_window import preprocess_live_window

        X_test = preprocess_live_window(ch_id, raw_buffer, self, window_size=self.window_size)
        return self.predict(ch_id, X_test)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save every channel's fitted detector into one .joblib file, plus a
        `meta` block (model type, channel list, library versions, save
        timestamp). See docs/MODEL_INTEGRATION_GUIDE.md, "Versioning and
        loading saved models".
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "detector_cls_name": self.detector_cls.__name__,
            "detector_module": self.detector_cls.__module__,
            "detectors": self.detectors,
            "scalers": self.scalers,
            "window_size": self.window_size,
            "meta": build_artifact_metadata(
                model_type=self.model_type,
                channel_ids=list(self.detectors.keys()),
                window_size=self.window_size,
            ),
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path],
              detector_cls: Optional[Type[BaseDetector]] = None) -> "MultiChannelDetector":
        """
        Load a saved artifact.

        Parameters
        ----------
        path : path to the .joblib file written by save().
        detector_cls : optional override. Normally unnecessary -- the
            correct class is resolved automatically from what save()
            recorded (detector_module / detector_cls_name), as long as that
            module is importable in the current environment. Pass this
            explicitly only if you're loading an artifact whose original
            detector class has moved/been renamed.
        """
        payload = joblib.load(Path(path))

        if detector_cls is None:
            detector_cls = _resolve_detector_cls(
                payload.get("detector_module"), payload.get("detector_cls_name")
            )

        obj = cls(detector_cls, window_size=payload.get("window_size", WINDOW_SIZE))
        obj.detectors = payload["detectors"]
        obj.scalers = payload.get("scalers", {})
        obj.meta = payload.get("meta", {})
        check_interface_version(obj.meta, source=str(path))
        return obj


def _resolve_detector_cls(module_name: Optional[str], cls_name: Optional[str]) -> Type[BaseDetector]:
    if not module_name or not cls_name:
        raise ValueError(
            "Saved artifact does not record its detector class (older format?). "
            "Pass detector_cls=... explicitly to MultiChannelDetector.load()."
        )
    import importlib

    module = importlib.import_module(module_name)
    detector_cls = getattr(module, cls_name)
    if not (isinstance(detector_cls, type) and issubclass(detector_cls, BaseDetector)):
        raise TypeError(f"Resolved {module_name}.{cls_name}, which is not a BaseDetector subclass.")
    return detector_cls
