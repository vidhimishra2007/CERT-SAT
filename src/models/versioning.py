"""
src/models/versioning.py

Shared helper for the metadata block stamped onto every saved CERT-SAT
model artifact (.joblib file), by both the legacy MultiChannelIsolationForest
(src/models/isolation_forest.py) and the generic MultiChannelDetector
(src/models/multi_channel.py).

See docs/MODEL_INTEGRATION_GUIDE.md, section "Versioning and loading saved
models", for what each field means and how the backend should check it.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

import sklearn

# Bump this when the *shape* of the saved artifact itself changes (e.g. keys
# added/removed from the joblib payload, meta schema changes) -- NOT every
# time a model is retrained. Backends can use this to decide whether their
# loading code is compatible with a given artifact.
INTERFACE_VERSION = "1.0"


def build_artifact_metadata(model_type: str, channel_ids: list[str], **extra) -> dict:
    """
    Build the `meta` dict stamped into a saved artifact.

    Parameters
    ----------
    model_type : e.g. "isolation_forest", "lof", "one_class_svm"
    channel_ids : channel ids included in this artifact
    **extra : any additional fields the caller wants recorded (e.g.
        window_size, stride, contamination settings)
    """
    return {
        "interface_version": INTERFACE_VERSION,
        "model_type": model_type,
        "channel_count": len(channel_ids),
        "channel_ids": sorted(channel_ids),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        **extra,
    }


def check_interface_version(meta: dict, source: str = "artifact") -> None:
    """
    Warn (do not raise) if a loaded artifact's interface_version doesn't
    match what this codebase expects. Missing meta (pre-versioning
    artifacts) is treated as version "unknown", not an error.
    """
    found = meta.get("interface_version", "unknown (pre-versioning artifact)")
    if found != INTERFACE_VERSION:
        print(
            f"[versioning] Warning: {source} was saved with interface_version="
            f"{found!r}, but this codebase expects {INTERFACE_VERSION!r}. "
            f"The artifact may still load fine -- but if predict() shapes or "
            f"keys look unexpected, retrain and re-save with the current code."
        )
