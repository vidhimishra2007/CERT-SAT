# CERT-SAT Backend Contract

Use this contract for raw telemetry integration with:

```text
models_saved/isoforest_all_channels_with_scalers.joblib
```

## 1. What exactly should be passed to `predict()`

Do not pass raw telemetry directly to `predict()`.

For raw telemetry, call the raw-window helper:

```python
result = manager.predict_raw_window(ch_id, raw_buffer)
```

`predict()` receives:

```text
ch_id: string
X_window_features: numpy.ndarray, shape (1, n_raw_features * 5)
```

Example for channel `P-1`:

```text
raw_buffer shape: (120, 25)
X_window_features shape: (1, 125)
```

The 5 generated summary features per raw feature are:

```text
mean, std, min, max, last_value - first_value
```

## 2. How should the rolling buffer be maintained?

Maintain one rolling buffer per channel.

```python
from collections import defaultdict, deque

import numpy as np

from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE


manager = MultiChannelIsolationForest.load(
    "models_saved/isoforest_all_channels_with_scalers.joblib"
)

buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


def handle_packet(ch_id: str, telemetry_vector):
    buffers[ch_id].append(telemetry_vector)

    if len(buffers[ch_id]) < WINDOW_SIZE:
        return {
            "ch_id": ch_id,
            "status": "warming_up",
            "buffer_size": len(buffers[ch_id]),
            "required_buffer_size": WINDOW_SIZE,
            "prediction": None,
        }

    raw_buffer = np.asarray(buffers[ch_id], dtype=float)
    result = manager.predict_raw_window(ch_id, raw_buffer)

    return {
        "ch_id": result["ch_id"],
        "status": "scored",
        "is_anomaly": bool(result["is_anomaly"][0]),
        "label": int(result["label"][0]),
        "score": float(result["score"][0]),
        "n_samples": int(result["n_samples"]),
    }
```

Rules:

- Keep packets in chronological order.
- Keep a separate buffer for every channel, such as `P-1`, `S-1`, `E-1`.
- Each buffer must contain exactly `120` latest packets before scoring.
- Each packet must have the same raw feature count used during training for that channel.
- Do not fit or refit scalers in backend. The fitted scalers are already saved inside the model artifact.

## 3. What is the final output format?

The model returns a Python dictionary:

```python
{
    "ch_id": "P-1",
    "is_anomaly": array([True]),
    "label": array([-1]),
    "score": array([-0.16101739]),
    "n_samples": 1
}
```

Recommended JSON API output:

```json
{
  "ch_id": "P-1",
  "status": "scored",
  "is_anomaly": true,
  "label": -1,
  "score": -0.16101739,
  "n_samples": 1
}
```

Warm-up output before 120 packets:

```json
{
  "ch_id": "P-1",
  "status": "warming_up",
  "buffer_size": 37,
  "required_buffer_size": 120,
  "prediction": null
}
```

Meaning:

```text
is_anomaly = true means anomaly detected
label = -1 means anomaly
label = 1 means normal
score lower means more anomalous
n_samples = 1 because one rolling window is scored at a time
```
