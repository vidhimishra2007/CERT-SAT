"""
generate_regression_fixtures.py

This is the exact script used to create tests/fixtures/regression_fixtures.json.
Run it from your project root (the "Anomaly Detection" folder):

    python generate_regression_fixtures.py

It does 4 simple things, repeated for several channels:
  1. Open a real .npy data file (already in your project).
  2. Cut out a 120-row chunk from it.
  3. Ask your already-trained model what it thinks of that chunk.
  4. Write down the question + the model's answer into a JSON file.

Nothing here trains a new model or invents data (except 3 clearly-labeled
"synthetic" test cases at the end, which are plain arrays of zeros/ones/fives
used to test extreme inputs -- not real satellite readings).
"""

import sys, os, json, hashlib
import numpy as np
import sklearn, joblib

# Make sure Python can find your src/ folder no matter where this is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.isolation_forest import MultiChannelIsolationForest
from src.inference.live_window import preprocess_live_window
from src.utils.config import WINDOW_SIZE, STRIDE

MODEL_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"
TEST_DIR = "archive/data/data/test"
OUTPUT_PATH = "tests/fixtures/regression_fixtures.json"

# Which channels to pull real data from. These are the actual NASA telemetry
# .npy files that were already sitting in your archive/data/data/test folder.
CHANNELS = ["P-1", "E-1", "A-1", "T-1", "G-1", "M-6", "C-1"]


def main():
    # --- Step 1: open your real, already-trained model -----------------
    print(f"Loading model from {MODEL_PATH} ...")
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    print(f"Loaded {len(manager.detectors)} channel models.\n")

    fixtures = {}

    # --- Step 2: for each channel, grab 3 real chunks of real data -----
    for ch_id in CHANNELS:
        # Open the real .npy file for this channel -- this data already
        # existed in your project, we are only reading it.
        test_arr = np.load(os.path.join(TEST_DIR, f"{ch_id}.npy"))
        n_rows = test_arr.shape[0]

        # Pick 3 starting points: the very beginning, ~1/3 through, ~2/3
        # through. This just gives us variety instead of only testing the
        # start of the file every time.
        offsets = sorted(set([0, n_rows // 3, (2 * n_rows) // 3]))
        offsets = [o for o in offsets if o + WINDOW_SIZE <= n_rows]

        for offset in offsets:
            # Cut out exactly 120 rows starting at this offset -- this
            # is the real chunk of telemetry we're going to ask the
            # model about.
            raw_window = test_arr[offset : offset + WINDOW_SIZE]

            # --- Step 3: ask the real model what it thinks -------------
            X = preprocess_live_window(ch_id, raw_window, manager)
            result = manager.predict(ch_id, X)

            label = int(result["label"][0])          # 1 = normal, -1 = anomaly
            score = float(result["score"][0])         # lower = more anomalous

            # A "fingerprint" of the exact data we used, so later we can
            # check we're re-reading the same real data, not something
            # that quietly changed.
            data_fingerprint = hashlib.sha256(raw_window.tobytes()).hexdigest()

            # --- Step 4: write down the question + the model's answer --
            key = f"{ch_id}_offset{offset}"
            fixtures[key] = {
                "ch_id": ch_id,
                "offset": offset,
                "synthetic": None,  # None = this is real data, not made up
                "window_shape": list(raw_window.shape),
                "raw_window_sha256": data_fingerprint,
                "expected_label": label,
                "expected_is_anomaly": bool(label == -1),
                "expected_score": score,
            }
            print(f"  {key}: label={label}  score={score:.6f}")

    # --- Extra: a few made-up (synthetic) edge cases, clearly labeled --
    # These are NOT real satellite data. They're plain arrays we build by
    # hand, just to see how the model reacts to extreme/unusual input.
    print("\nAdding synthetic edge cases (not real data)...")
    ch_id = "P-1"
    n_features = manager.detectors[ch_id].model.n_features_in_ // 5

    synthetic_inputs = {
        "zeros": np.zeros((WINDOW_SIZE, n_features)),
        "ones": np.ones((WINDOW_SIZE, n_features)),
        "constant_5": np.full((WINDOW_SIZE, n_features), 5.0),
    }

    for name, raw_window in synthetic_inputs.items():
        X = preprocess_live_window(ch_id, raw_window, manager)
        result = manager.predict(ch_id, X)
        label = int(result["label"][0])
        score = float(result["score"][0])

        key = f"{ch_id}_synthetic_{name}"
        fixtures[key] = {
            "ch_id": ch_id,
            "offset": None,
            "synthetic": name,  # marks this clearly as made-up, not real
            "window_shape": list(raw_window.shape),
            "raw_window_sha256": hashlib.sha256(raw_window.tobytes()).hexdigest(),
            "expected_label": label,
            "expected_is_anomaly": bool(label == -1),
            "expected_score": score,
        }
        print(f"  {key}: label={label}  score={score:.6f}")

    # --- Save everything to the answer-key JSON file --------------------
    model_sha256 = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    output = {
        "meta": {
            "model_path": MODEL_PATH,
            "model_sha256": model_sha256,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "generated_with": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "score_tolerance": 1e-6,
        },
        "cases": fixtures,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(fixtures)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()