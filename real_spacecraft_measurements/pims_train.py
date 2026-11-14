"""Train a supplemental forecast of real NASA ISS MAMS measured acceleration.

Fixed UTC splits: Nov 1-7, 2026 train; Nov 8-9 validation; Nov 10-11 test.
Eight past 3-axis samples at 16 seconds predict the next measured sample.
The candidate budget is fixed: persistence, Ridge(alpha=10), and one small
MLP residual model with explicit chronological-validation early stopping.
Never call this spacecraft actuator-response or artificial-gravity training.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pims"
AXES = ["ax_g", "ay_g", "az_g"]
LOOKBACK = 8
DT = 16
HORIZONS = [1, 5, 15]
SEED = 20261114
SPLITS = {
    "train": ("2026-11-01T00:00:00Z", "2026-11-08T00:00:00Z"),
    "validation": ("2026-11-08T00:00:00Z", "2026-11-10T00:00:00Z"),
    "test": ("2026-11-10T00:00:00Z", "2026-11-12T00:00:00Z"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inference():
    spec = importlib.util.spec_from_file_location("pims_portable_inference", OUT / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_data():
    path = ROOT / "data" / "processed" / "iss_mams_acceleration.csv"
    manifest_path = ROOT / "data" / "raw" / "pims" / "source_manifest.json"
    split_path = ROOT / "data" / "processed" / "iss_mams_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    checksum = sha256(path)
    if checksum != manifest["processed"]["sha256"] or checksum != split_manifest["dataset_sha256"]:
        raise AssertionError("Processed data checksum does not match both manifests")
    for split, bounds in SPLITS.items():
        if [split_manifest[split]["start_inclusive"], split_manifest[split]["end_exclusive"]] != list(bounds):
            raise AssertionError("UTC split differs from the preregistered split")
    frame = pd.read_csv(path)
    required = ["timestamp_utc", *AXES, "segment_id", "source_file", "quality"]
    if frame.columns.tolist() != required:
        raise ValueError(f"Unexpected CSV schema: {frame.columns.tolist()}")
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    if not frame.timestamp_utc.is_monotonic_increasing or frame.timestamp_utc.duplicated().any():
        raise AssertionError("Timestamps must be unique and chronologically sorted")
    if not np.isfinite(frame[AXES].to_numpy()).all() or not (frame.quality == "Valid").all():
        raise AssertionError("Input contains nonfinite or nonvalid measurements")
    if manifest["dt_s"] != DT:
        raise AssertionError("Unexpected sample cadence")
    return frame, manifest, checksum


def windows_for_split(frame, split, horizon):
    start, end = map(pd.Timestamp, SPLITS[split])
    subset = frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc < end)].copy()
    # Break on every source segment OR timestamp gap. No padding or interpolation.
    breaks = subset.segment_id.ne(subset.segment_id.shift()) | subset.timestamp_utc.diff().dt.total_seconds().ne(DT)
    subset["chunk"] = breaks.cumsum()
    histories, targets, times, chunks = [], [], [], []
    for chunk_id, group in subset.groupby("chunk", sort=False):
        values = group[AXES].to_numpy(dtype=np.float64)
        n = len(values) - LOOKBACK - horizon + 1
        if n <= 0:
            continue
        offsets = np.arange(n)[:, None]
        histories.append(values[offsets + np.arange(LOOKBACK)[None, :]])
        targets.append(values[offsets + LOOKBACK + np.arange(horizon)[None, :]])
        times.append(group.timestamp_utc.iloc[LOOKBACK:LOOKBACK + n].to_numpy())
        chunks.append(np.repeat(f"{split}:{group.segment_id.iloc[0]}:{int(chunk_id)}", n))
    if not histories:
        raise ValueError(f"No valid windows for {split}")
    return {
        "history": np.concatenate(histories),
        "target": np.concatenate(targets),
        "time": np.concatenate(times),
        "chunk": np.concatenate(chunks),
        "input_rows": len(subset),
        "chunks": int(subset.chunk.nunique()),
        "segments": int(subset.segment_id.nunique()),
        "first_utc": subset.timestamp_utc.iloc[0].isoformat(),
        "last_utc": subset.timestamp_utc.iloc[-1].isoformat(),
    }


def metrics(truth, predicted):
    error = np.asarray(predicted) - np.asarray(truth)
    rmse = np.sqrt(np.mean(error * error, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    bias = np.mean(error, axis=0)
    overall = float(np.sqrt(np.mean(error * error)))
    return {
        "samples": len(error), "overall_rmse_g": overall,
        "overall_rmse_micro_g": overall * 1e6,
        "overall_mae_g": float(np.mean(np.abs(error))),
        "overall_mae_micro_g": float(np.mean(np.abs(error)) * 1e6),
        "per_axis": {axis: {
            "rmse_g": float(rmse[i]), "rmse_micro_g": float(rmse[i] * 1e6),
            "mae_g": float(mae[i]), "mae_micro_g": float(mae[i] * 1e6),
            "bias_g": float(bias[i]), "bias_micro_g": float(bias[i] * 1e6),
        } for i, axis in enumerate(AXES)},
    }


def model_document(kind, dataset_hash, x_scaler=None, y_scaler=None, fitted=None):
    doc = {
        "format": "iss-mams-forecast-v1", "kind": kind,
        "data_type": "MEASURED_PROCESSED_ISS_ACCELERATION_FORECAST",
        "scope": "Forecast filtered ISS disturbance acceleration; no actuator actions, RPM, habitat or gravity-control outputs.",
        "units": "g", "dt_s": DT, "lookback_samples": LOOKBACK,
        "input_shape": [LOOKBACK, 3], "axis_order": AXES,
        "history_order": "oldest to newest; last sample is forecast origin",
        "history_span_between_first_and_last_samples_s": (LOOKBACK - 1) * DT,
        "feature_order": [f"{axis}_at_origin_minus_{lag * DT}s" for lag in range(LOOKBACK - 1, -1, -1) for axis in AXES],
        "forecast_horizon_s": DT, "output_order": AXES,
        "output_formula": "last_measured_acceleration_g + predicted_delta_g" if kind != "persistence" else "last_measured_acceleration_g",
        "processed_data_sha256": dataset_hash,
        "normalization_fit_partition": "train: 2026-11-01 through 2026-11-07 UTC only",
        "tested_max_recursive_steps": max(HORIZONS),
    }
    if kind != "persistence":
        doc["feature_scaler"] = {"mean": x_scaler.mean_.tolist(), "scale": x_scaler.scale_.tolist()}
        doc["residual_scaler"] = {"mean": y_scaler.mean_.tolist(), "scale": y_scaler.scale_.tolist()}
    if kind == "ridge_residual":
        doc["alpha"] = float(fitted.alpha)
        doc["coefficients"] = fitted.coef_.tolist()
        doc["intercept"] = fitted.intercept_.tolist()
    if kind == "mlp_residual":
        doc["activation"] = fitted.activation
        doc["output_activation"] = fitted.out_activation_
        doc["layers"] = [{"weights": coef.tolist(), "bias": intercept.tolist()}
                         for coef, intercept in zip(fitted.coefs_, fitted.intercepts_)]
    return doc


def plot_results(report, history, validation_scores, test_windows, selected_pred, baseline_pred, rollout_metrics, out):
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["validation_rmse_micro_g"] for row in history], color="#1764ab", label="MLP validation")
    axes[0].axhline(validation_scores["persistence"]["overall_rmse_micro_g"], ls="--", color="#767676", label="Persistence")
    axes[0].axvline(report["training"]["mlp_best_epoch"], ls=":", color="#ab6517", label="Selected MLP epoch")
    axes[0].set(xlabel="Training epoch", ylabel="Validation RMSE (micro-g)", title="Chronological validation; no test tuning")
    axes[0].legend(fontsize=8)
    kinds = list(report["one_step_test"])
    values = [report["one_step_test"][kind]["overall_rmse_micro_g"] for kind in kinds]
    bars = axes[1].bar(kinds, values, color=["#777777", "#329471", "#1764ab"])
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(np.arange(len(kinds)), ["Persistence", "Ridge residual", "MLP residual"])
    axes[1].set(ylabel="Test RMSE (micro-g)", title="Held-out Nov 10–11: 16-second forecast")
    axes[1].set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(out / "validation_and_test.png")
    plt.close(fig)

    # Deterministic illustrative excerpt: first at most 225 contiguous test origins.
    first_chunk = test_windows["chunk"][0]
    count = min(225, int(np.sum(test_windows["chunk"] == first_chunk)))
    ts = pd.to_datetime(test_windows["time"][:count], utc=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for j, axis in enumerate(AXES):
        axes[j].plot(ts, test_windows["target"][:count, 0, j] * 1e6, color="#20242b", lw=1.4, label="Measured next sample")
        axes[j].plot(ts, selected_pred[:count, j] * 1e6, color="#1764ab", lw=1.0, label=report["selection"]["winner"])
        axes[j].plot(ts, baseline_pred[:count, j] * 1e6, color="#aa6747", lw=0.8, alpha=0.75, label="Persistence")
        axes[j].set_ylabel(f"{axis[1]} (micro-g)")
        axes[j].grid(alpha=0.18)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Actual held-out measurements and one-step forecasts — first contiguous test hour")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axes[-1].set_xlabel("Forecast target time on 10 November 2026 (UTC)")
    fig.tight_layout()
    fig.savefig(out / "held_out_forecast.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, color in [("persistence", "#777777"), (report["selection"]["winner"], "#1764ab")]:
        if name == "persistence" and report["selection"]["winner"] == "persistence":
            continue
        ax.plot([h * DT for h in HORIZONS], [rollout_metrics[name][str(h)]["overall_rmse_micro_g"] for h in HORIZONS], marker="o", label=name, color=color)
    ax.set(xlabel="Recursive forecast horizon (seconds)", ylabel="Test RMSE (micro-g)", title="Open-loop recursive forecast: no future observations")
    ax.set_xticks([h * DT for h in HORIZONS])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "recursive_forecast.png")
    plt.close(fig)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.max_epochs > 80 or args.patience < 1:
        raise ValueError("Training is bounded to 1–80 epochs")
    OUT.mkdir(parents=True, exist_ok=True)
    portable = load_inference()
    frame, source, dataset_hash = read_data()
    train = windows_for_split(frame, "train", 1)
    validation = windows_for_split(frame, "validation", 1)
    # Test windows and labels are not used until after the selected model is frozen.
    x = train["history"].reshape(-1, LOOKBACK * 3)
    y_delta = train["target"][:, 0, :] - train["history"][:, -1, :]
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y_delta)
    xs = x_scaler.transform(x)
    ys = y_scaler.transform(y_delta)
    vx = x_scaler.transform(validation["history"].reshape(-1, LOOKBACK * 3))
    vy = validation["target"][:, 0, :]
    vlast = validation["history"][:, -1, :]
    validation_scores = {"persistence": metrics(vy, vlast)}
    documents = {"persistence": model_document("persistence", dataset_hash)}
    print(f"Verified measured data: {len(frame):,} rows; {len(x):,} train windows; {len(vx):,} validation windows", flush=True)
    with threadpool_limits(limits=2):
        ridge = Ridge(alpha=10.0).fit(xs, ys)
        ridge_val = vlast + y_scaler.inverse_transform(ridge.predict(vx))
        validation_scores["ridge_residual"] = metrics(vy, ridge_val)
        documents["ridge_residual"] = model_document("ridge_residual", dataset_hash, x_scaler, y_scaler, ridge)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                           alpha=1e-4, batch_size=512, learning_rate_init=0.001,
                           max_iter=1, shuffle=True, early_stopping=False,
                           random_state=SEED, tol=0.0)
        history = []
        best_rmse = float("inf")
        best_epoch = 0
        no_improvement = 0
        best_mlp = None
        for epoch in range(1, args.max_epochs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                mlp.partial_fit(xs, ys)
            prediction = vlast + y_scaler.inverse_transform(mlp.predict(vx))
            score = metrics(vy, prediction)["overall_rmse_g"]
            history.append({"epoch": epoch, "training_standardized_residual_loss": float(mlp.loss_),
                            "validation_rmse_g": score, "validation_rmse_micro_g": score * 1e6})
            if score < best_rmse:
                # Keep the actual best epoch; patience resets only for material gains.
                material = score < best_rmse * (1 - 1e-4)
                best_rmse = score
                best_epoch = epoch
                best_mlp = copy.deepcopy(mlp)
                no_improvement = 0 if material else no_improvement + 1
            else:
                no_improvement += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f"MLP epoch {epoch}: validation RMSE {score * 1e6:.6f} micro-g", flush=True)
            if no_improvement >= args.patience:
                break
        mlp = best_mlp
        validation_scores["mlp_residual"] = metrics(vy, vlast + y_scaler.inverse_transform(mlp.predict(vx)))
        documents["mlp_residual"] = model_document("mlp_residual", dataset_hash, x_scaler, y_scaler, mlp)

        winner = min(validation_scores, key=lambda name: validation_scores[name]["overall_rmse_g"])
        frozen_at = datetime.now(timezone.utc).isoformat()
        selection = {
            "criterion": "Lowest overall validation one-step RMSE, including persistence",
            "winner": winner, "frozen_before_test_at_utc": frozen_at,
            "validation": validation_scores,
            "baseline_beats_learned_models": winner == "persistence",
            "hyperparameter_candidates": {"ridge_residual": {"alpha": 10.0}, "mlp_residual": {"hidden": [32, 16], "activation": "tanh", "learning_rate_init": 0.001, "alpha": 1e-4}},
        }
        write_json(OUT / "selection_frozen_before_test.json", selection)
        for name, document in documents.items():
            document["selected_on_validation"] = name == winner
            write_json(OUT / f"{name}.json", document)
        write_json(OUT / "selected_model.json", documents[winner])
        joblib.dump({"ridge": ridge, "mlp": mlp, "feature_scaler": x_scaler, "residual_scaler": y_scaler,
                     "winner": winner, "processed_data_sha256": dataset_hash}, OUT / "fitted_models.joblib")

        # The test is evaluated once after all models/settings/weights are frozen.
        test = windows_for_split(frame, "test", 1)
        long_test = windows_for_split(frame, "test", max(HORIZONS))
        tx = x_scaler.transform(test["history"].reshape(-1, LOOKBACK * 3))
        truth = test["target"][:, 0, :]
        test_predictions = {
            "persistence": test["history"][:, -1, :],
            "ridge_residual": test["history"][:, -1, :] + y_scaler.inverse_transform(ridge.predict(tx)),
            "mlp_residual": test["history"][:, -1, :] + y_scaler.inverse_transform(mlp.predict(tx)),
        }
        test_metrics = {name: metrics(truth, prediction) for name, prediction in test_predictions.items()}
        reload_checks = {}
        for name in documents:
            reloaded = portable.load_model(OUT / f"{name}.json")
            pred = portable.predict_batch(reloaded, test["history"][:256])
            max_error = float(np.max(np.abs(pred - test_predictions[name][:256])))
            if not np.allclose(pred, test_predictions[name][:256], rtol=1e-11, atol=1e-16):
                raise AssertionError(f"Portable JSON inference differs from sklearn for {name}")
            reload_checks[name] = {"compared_held_out_windows": len(pred), "max_absolute_difference_g": max_error, "passed": True}
        selected = portable.load_model(OUT / "selected_model.json")
        selected_long = portable.rollout(selected, long_test["history"], max(HORIZONS))
        baseline_long = np.repeat(long_test["history"][:, -1:, :], max(HORIZONS), axis=1)
        rollout_metrics = {"persistence": {str(h): metrics(long_test["target"][:, h - 1], baseline_long[:, h - 1]) for h in HORIZONS}}
        rollout_metrics[winner] = {str(h): metrics(long_test["target"][:, h - 1], selected_long[:, h - 1]) for h in HORIZONS}

    split_counts = {}
    for name, data in [("train", train), ("validation", validation), ("test", test)]:
        split_counts[name] = {key: data[key] for key in ["input_rows", "chunks", "segments", "first_utc", "last_utc"]}
        split_counts[name]["one_step_windows"] = len(data["history"])
        split_counts[name]["bounds"] = {"start_inclusive": SPLITS[name][0], "end_exclusive": SPLITS[name][1]}
    report = {
        "purpose": "Supplemental model of measured ISS filtered disturbance acceleration, not ring, motor-command, artificial-gravity or biological response.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"rows": len(frame), "sha256": dataset_hash, "source_urls": source["sources"],
                    "raw_binary_files": len(source["files"]), "raw_binary_records": sum(p["records"] for p in source["files"]),
                    "raw_binary_and_header_bytes": sum(f["bytes"] for pair in source["files"] for f in pair["files"]),
                    "quality_values": sorted(frame.quality.unique().tolist()), "segments": int(frame.segment_id.nunique()),
                    "raw_manifest": "data/raw/pims/source_manifest.json", "processed_csv": "data/processed/iss_mams_acceleration.csv",
                    "acceleration_units": "standard gravity g", "sample_interval_s": DT,
                    "measurement_kind": "OSS bias-compensated trimmed-mean-filtered; no second bias correction or interpolation performed",
                    "rows_by_utc_day": source["processed"]["rows_by_utc_day"], "parsing_assumptions": source["parsing_assumptions"]},
        "features": {"lookback_samples": LOOKBACK, "input_dimensions": LOOKBACK * 3, "axes": AXES,
                     "history_first_to_last_span_s": (LOOKBACK - 1) * DT, "next_sample_horizon_s": DT,
                     "residual_target": "next measured acceleration minus most recent measured acceleration",
                     "scaling_fit_on": "train only"},
        "splits": split_counts,
        "training": {"candidate_budget": "One fixed Ridge setting and one fixed MLP setting plus persistence; no test-driven tuning",
                     "random_seed": SEED, "mlp_epochs_run": len(history), "mlp_best_epoch": best_epoch,
                     "mlp_max_epochs": args.max_epochs, "mlp_patience": args.patience,
                     "mlp_stopping": "Chronological validation RMSE; no sklearn internal random validation split",
                     "mlp_refit_after_selection": False, "training_seconds": time.monotonic() - started},
        "selection": selection,
        "one_step_test": test_metrics,
        "recursive_test": {"origins": len(long_test["history"]), "horizons_steps": HORIZONS,
                           "horizons_seconds": [h * DT for h in HORIZONS],
                           "same_origins_all_horizons": True, "no_future_truth_inputs": True,
                           "metrics": rollout_metrics},
        "portable_json_reload_checks": reload_checks,
        "validation_checks": {
            "processed_sha256_matches_source_and_split_manifests": True,
            "chronologically_sorted_unique_timestamps": True,
            "finite_measurements_and_Valid_source_quality_only": True,
            "windows_never_cross_segment_gap_or_split": True,
            "window_requires_exact_16_second_cadence": True,
            "models_and_selection_saved_before_test": True,
            "test_labels_used_only_for_final_metrics_and_illustration": True,
        },
        "limits": [
            "A short November 2026 interval is not representative of every ISS mission state; there is no external mission validation.",
            "The target is an already filtered accelerometer series at its documented coordinate system and location, not whole-ISS rigid-body acceleration.",
            "No actuator command, force, torque, RPM, inertia or rotating-habitat label is present; causal command-response dynamics cannot be trained here.",
            "A recursive forecast is not a stable closed-loop controller or a validation of a crew artificial-gravity system.",
            "The 1/5/15-step recursive metrics use common 15-step-capable origins and therefore have slightly fewer samples than one-step metrics.",
            "No confidence interval or significance claim is made for differences from persistence; serial dependence is substantial.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    baseline_rmse = test_metrics["persistence"]["overall_rmse_g"]
    selected_rmse = test_metrics[winner]["overall_rmse_g"]
    report["selection"]["held_out_rmse_reduction_vs_persistence_percent"] = 100 * (1 - selected_rmse / baseline_rmse)
    report["selection"]["held_out_better_than_persistence"] = selected_rmse < baseline_rmse
    write_json(OUT / "training_report.json", report)
    pd.DataFrame(history).to_csv(OUT / "mlp_training_history.csv", index=False)
    pred_frame = pd.DataFrame({"target_timestamp_utc": pd.to_datetime(test["time"], utc=True), "chunk": test["chunk"]})
    for j, axis in enumerate(AXES):
        pred_frame[f"measured_{axis}"] = truth[:, j]
        for name, predicted in test_predictions.items():
            pred_frame[f"{name}_{axis}"] = predicted[:, j]
    pred_frame.to_csv(OUT / "held_out_predictions.csv", index=False)
    write_json(OUT / "example_history_g.json", test["history"][0].tolist())
    write_json(OUT / "inference_test_vectors.json", {
        "selected_model": winner, "units": "g", "history_order": "oldest to newest",
        "examples": [{"history_g": test["history"][i].tolist(), "expected_next_g": test_predictions[winner][i].tolist()}
                     for i in range(min(8, len(truth)))],
    })
    plot_results(report, history, validation_scores, test, test_predictions[winner], test_predictions["persistence"], rollout_metrics, OUT)
    # Lightweight plain text usage instructions live beside the portable model.
    (OUT / "README.txt").write_text(
        "NASA ISS MAMS measured acceleration forecasting\n"
        "================================================\n"
        f"Validation-selected model: {winner}\n"
        "All inputs and outputs use g (multiply by 1e6 for micro-g).\n"
        "Input exactly eight 3-axis observations, oldest first, 16 seconds apart.\n"
        "The first and last input observations are 112 seconds apart.\n"
        "Output forecasts the sample 16 seconds after the last input.\n\n"
        "Run from this directory:\n"
        "  python inference.py --history example_history_g.json --steps 15\n\n"
        "Python use:\n"
        "  from inference import load_model, predict_one, rollout\n"
        "  model = load_model('selected_model.json')\n"
        "  forecast_g = predict_one(model, history_8_by_3_g)\n\n"
        "Use training_report.json for actual metrics, source URLs, limitations and checks.\n"
        "selection_frozen_before_test.json records the decision before test evaluation.\n"
        "held_out_predictions.csv contains every evaluated one-step test prediction.\n"
        "fitted_models.joblib requires the recorded sklearn version; JSON inference needs only NumPy.\n"
        "This forecasts filtered ISS disturbances; it is not a rotating-ring or actuator-response model.\n",
        encoding="utf-8")
    print(json.dumps({"selected": winner, "validation_rmse_micro_g": validation_scores[winner]["overall_rmse_micro_g"],
                      "test_rmse_micro_g": selected_rmse * 1e6, "persistence_test_rmse_micro_g": baseline_rmse * 1e6,
                      "recursive": rollout_metrics, "output": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
