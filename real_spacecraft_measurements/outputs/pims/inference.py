"""Portable NumPy inference for the accompanying NASA ISS MAMS forecast model.

Input: eight observed [ax_g, ay_g, az_g] rows in chronological order,
exactly 16 seconds apart. Output: the next three-axis acceleration in g.
These are already filtered ISS acceleration measurements, not actuator commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_model(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("format") != "iss-mams-forecast-v1":
        raise ValueError("Unsupported model format")
    return model


def predict_batch(model, history_g):
    """Predict from (batch, 8, 3) histories. This function never reads truth."""
    history = np.asarray(history_g, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (8, 3):
        raise ValueError("Expected history shape (batch, 8, 3), oldest first")
    if not np.isfinite(history).all():
        raise ValueError("History must contain finite measured values")
    last = history[:, -1, :]
    if model["kind"] == "persistence":
        return last.copy()
    scaler = model["feature_scaler"]
    x = (history.reshape(len(history), 24) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if model["kind"] == "ridge_residual":
        delta_standardized = x @ np.asarray(model["coefficients"]).T + np.asarray(model["intercept"])
    elif model["kind"] == "mlp_residual":
        for index, layer in enumerate(model["layers"]):
            x = x @ np.asarray(layer["weights"]) + np.asarray(layer["bias"])
            if index < len(model["layers"]) - 1:
                if model["activation"] == "tanh":
                    x = np.tanh(x)
                elif model["activation"] == "relu":
                    x = np.maximum(x, 0.0)
                else:
                    raise ValueError("Unsupported hidden activation")
        delta_standardized = x
    else:
        raise ValueError("Unknown model kind")
    target_scaler = model["residual_scaler"]
    delta_g = delta_standardized * np.asarray(target_scaler["scale"]) + np.asarray(target_scaler["mean"])
    return last + delta_g


def predict_one(model, history_g):
    """Predict one next sample in g from eight chronological measured samples."""
    return predict_batch(model, np.asarray(history_g)[None, ...])[0]


def rollout(model, history_g, steps):
    """Autoregressive prediction: only the initial history is supplied.

    Returns (steps, 3) for one history, or (batch, steps, 3) for batches.
    Subsequent inputs contain model predictions, with no future observations.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    values = np.asarray(history_g, dtype=np.float64)
    single = values.ndim == 2
    state = values[None, ...].copy() if single else values.copy()
    predictions = []
    for _ in range(steps):
        next_value = predict_batch(model, state)
        predictions.append(next_value)
        state = np.concatenate([state[:, 1:, :], next_value[:, None, :]], axis=1)
    result = np.stack(predictions, axis=1)
    return result[0] if single else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("selected_model.json"))
    parser.add_argument("--history", type=Path, required=True, help="JSON array of 8 x 3 measured g values")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    print(json.dumps({"units": "g", "dt_s": 16, "forecast": rollout(load_model(args.model), history, args.steps).tolist()}, indent=2))
