"""Portable NumPy inference for measured ISS MARS-S RPM forecasting.

Input: eight real, consecutive 1-second RPM samples in chronological order.
Output: observed closed-loop rotor RPM forecast, NOT a motor command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_features(history_rpm):
    history = np.asarray(history_rpm, dtype=np.float64)
    if history.ndim == 1:
        history = history[None, :]
    if history.ndim != 2 or history.shape[1] != 8:
        raise ValueError("Expected eight chronological RPM samples per history.")
    if not np.isfinite(history).all():
        raise ValueError("History contains a missing or nonfinite RPM sample.")
    return np.column_stack((history[:, -1], np.diff(history, axis=1)))


def predict(model, history_rpm):
    history = np.asarray(history_rpm, dtype=np.float64)
    if history.ndim == 1:
        history = history[None, :]
    features = make_features(history)
    kind = model["kind"]
    if kind == "persistence":
        return history[:, -1].copy()
    if kind == "moving_mean":
        return history.mean(axis=1)
    value = (features - np.asarray(model["x_mean"])) / np.asarray(model["x_scale"])
    if kind == "ridge_residual":
        residual = value @ np.asarray(model["coef"]) + model["intercept"]
    elif kind == "mlp_residual":
        for index, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
            value = value @ np.asarray(weight) + np.asarray(bias)
            if index < len(model["weights"]) - 1:
                value = np.maximum(value, 0)
        residual = value[:, 0]
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return history[:, -1] + residual * model["y_scale"] + model["y_mean"]


def rollout(model, history_rpm, steps=1):
    """Diagnostic recursion. The MLP failed long-rollout evaluation; use predict for normal inference."""
    if not isinstance(steps, int) or not 1 <= steps <= 600:
        raise ValueError("steps must be an integer from 1 to 600.")
    history = np.asarray(history_rpm, dtype=np.float64)
    if history.ndim == 1:
        history = history[None, :]
    make_features(history)
    predictions = []
    for _ in range(steps):
        forecast = predict(model, history)
        predictions.append(forecast)
        history = np.column_stack((history[:, 1:], forecast))
    return np.column_stack(predictions)


def nominal_radial_g(rpm, radius_m=0.15):
    """Derived centrifugal acceleration at a nominal radius, not a measured label."""
    return radius_m * (np.asarray(rpm, dtype=float) * 2 * np.pi / 60) ** 2 / 9.80665


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent / "outputs" / "rpm"
    parser.add_argument("--model", type=Path, default=default / "selected_model_with_usage_notes.json")
    parser.add_argument("--history", type=Path, default=default / "example_history_rpm.json")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--diagnostic-rollout", action="store_true",
                        help="Explicitly reproduce unsupported recursive forecasts for analysis.")
    args = parser.parse_args()
    if args.steps > 1 and not args.diagnostic_rollout:
        parser.error("The RPM MLP failed recursive stability evaluation. Use --steps 1 with fresh measurements. To reproduce the failure, add --diagnostic-rollout.")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    example = json.loads(args.history.read_text(encoding="utf-8"))
    history = example["history_rpm"] if isinstance(example, dict) else example
    result = rollout(model, history, args.steps)[0]
    print(json.dumps({"model": model["name"], "sample_interval_seconds": 1,
        "forecast_rpm": result.tolist(),
        "nominal_derived_radial_g": [float(nominal_radial_g(value)) if value >= 0 else None for value in result],
        "invalid_negative_rpm_prediction": (result < 0).tolist(),
        "scope": "MARS-S observed RPM forecast; derived g is not a direct measurement; no control output",
        "warning": "Unknown future commands are unobserved. The MLP is for one-step measured-history prediction; recursive forecasts are unstable."}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
