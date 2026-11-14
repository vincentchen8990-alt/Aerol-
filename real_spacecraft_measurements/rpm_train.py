"""Train and independently test an RPM forecaster on actual ISS MARS-S logs.

MHU-1 chronological train/validation; MHU-2 is the independent mission test.
No simulator labels, motor commands, or synthetic interpolated data are used.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from rpm_predict import make_features, predict, rollout, nominal_radial_g

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "mhu"
OUT = ROOT / "outputs" / "rpm"
CONTEXT = 8
SEED = 20261114
CHUNK = 150_000


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mission(mission):
    number, member = {"MHU-1": (15, "SupplementaryData12.txt"),
                      "MHU-2": (16, "SupplementaryData13.txt")}[mission]
    path = RAW / f"41467_2026_76349_MOESM{number}_ESM.zip"
    manifest = json.loads((RAW / "download_manifest.json").read_text())
    source = next(row for row in manifest["files"] if row["filename"] == path.name)
    actual_hash = sha256(path)
    if actual_hash != source["sha256"]:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        frame = pd.read_csv(handle, sep="\t", usecols=["Date", "RPM"])
    frame["timestamp"] = pd.to_datetime(frame["Date"], format="mixed", errors="raise")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Raw source timestamps are out of order.")
    original_rows = len(frame)
    duplicate_rows = int(frame.duplicated("timestamp", keep="last").sum())
    duplicates = frame[frame.duplicated("timestamp", keep=False)]
    duplicate_conflicts = int((duplicates.groupby("timestamp")["RPM"].nunique() > 1).sum())
    frame = frame.drop_duplicates("timestamp", keep="last")
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    rpm = frame["RPM"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Deduplicated timestamps must strictly increase.")
    gaps = np.diff(timestamps) != 1
    metadata = {"mission": mission, "source_archive": str(path.relative_to(ROOT)),
        "source_url": source["source_url"], "sha256": actual_hash,
        "archive_member": member, "raw_rows": original_rows,
        "unique_timestamp_rows": len(rpm), "duplicate_rows_removed": duplicate_rows,
        "conflicting_duplicate_timestamps": duplicate_conflicts,
        "duplicate_policy": "Keep the last row in the source for each recorded timestamp; no averaging.",
        "timestamp_timezone": "Unspecified in the released export; timestamps are preserved as recorded, not relabeled UTC.",
        "first_timestamp_as_recorded": str(frame["timestamp"].iloc[0]),
        "last_timestamp_as_recorded": str(frame["timestamp"].iloc[-1]),
        "nonfinite_rpm_rows": int((~np.isfinite(rpm)).sum()),
        "zero_rpm_rows": int((rpm == 0).sum()),
        "one_second_segments": int(gaps.sum()) + 1,
        "largest_recorded_interval_seconds": int(np.diff(timestamps).max()),
        "rpm_min": float(np.nanmin(rpm)), "rpm_max": float(np.nanmax(rpm))}
    print(f"Loaded {mission}: {original_rows:,} raw rows, {len(rpm):,} unique timestamps", flush=True)
    return timestamps, rpm, metadata


def valid_targets(timestamps, rpm, horizon=1):
    # A target t has history [t-8, ..., t-1] and futures [t, ..., t+horizon-1].
    indices = np.arange(CONTEXT, len(rpm) - horizon + 1, dtype=np.int64)
    starts, ends = indices - CONTEXT, indices + horizon - 1
    good_cadence = timestamps[ends] - timestamps[starts] == CONTEXT + horizon - 1
    missing_prefix = np.r_[0, np.cumsum(~np.isfinite(rpm))]
    good_values = missing_prefix[ends + 1] - missing_prefix[starts] == 0
    return indices[good_cadence & good_values]


def histories(rpm, indices):
    return rpm[indices[:, None] - np.arange(CONTEXT, 0, -1)]


def dynamic_targets(rpm, indices):
    result = []
    for start in range(0, len(indices), CHUNK):
        batch = indices[start:start + CHUNK]
        result.append(batch[np.ptp(histories(rpm, batch), axis=1) > 2.0])
    return np.concatenate(result)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {"count": int(len(truth)), "finite_count": 0, "rmse_rpm": None, "mae_rpm": None}
    error = prediction[valid] - truth[valid]
    return {"count": int(len(truth)), "finite_count": int(valid.sum()),
        "rmse_rpm": float(np.sqrt(np.mean(error ** 2))),
        "mae_rpm": float(np.mean(np.abs(error))),
        "rmse_rad_per_s": float(np.sqrt(np.mean(error ** 2)) * 2 * np.pi / 60),
        "negative_predictions": int((prediction[valid] < 0).sum()),
        "maximum_absolute_error_rpm": float(np.max(np.abs(error)))}


def document(name, kind, scaler_x=None, scaler_y=None, fitted=None):
    result = {"schema_version": 1, "name": name, "kind": kind,
        "hardware": "JAXA ISS Kibo MARS-S centrifuge, MHU-1/MHU-2 configuration",
        "input": "8 consecutive measured RPM samples, oldest to newest",
        "context_samples": 8, "history_span_seconds": 7, "sample_interval_seconds": 1,
        "horizon_seconds": 1, "output_unit": "RPM", "nominal_radius_m": 0.15,
        "nominal_rotation_setting_rpm": 77,
        "feature_order": ["latest_rpm"] + [f"delta_rpm_{i}_to_{i+1}" for i in range(7)],
        "target": "next recorded RPM minus latest recorded RPM",
        "training_mission": "MHU-1", "independent_test_mission": "MHU-2",
        "scope": "Observed closed-loop RPM forecaster; not torque-response identification or a controller",
        "gravity_status": "r*omega^2/9.80665 is a derived nominal radial component, not a measured acceleration label",
        "unknown_inputs": ["motor command", "motor torque", "current", "inertia", "hub motion", "maintenance schedule"]}
    if scaler_x is not None:
        result.update(x_mean=scaler_x.mean_.tolist(), x_scale=scaler_x.scale_.tolist(),
                      y_mean=float(scaler_y.mean_[0]), y_scale=float(scaler_y.scale_[0]))
    if kind == "ridge_residual":
        result.update(coef=fitted.coef_.tolist(), intercept=float(fitted.intercept_))
    elif kind == "mlp_residual":
        result.update(weights=[x.tolist() for x in fitted.coefs_],
                      biases=[x.tolist() for x in fitted.intercepts_], activation="relu")
    return result


def predict_all(model, rpm, indices):
    result = np.empty(len(indices))
    for start in range(0, len(indices), CHUNK):
        result[start:start + CHUNK] = predict(model, histories(rpm, indices[start:start + CHUNK]))
    return result


def spaced_indices(indices, timestamps, interval):
    # Deterministic spacing uses recorded times, never future labels.
    chosen, last = [], None
    for index in indices:
        if last is None or timestamps[index] - last >= interval:
            chosen.append(index)
            last = timestamps[index]
    return np.asarray(chosen, dtype=np.int64)


def evaluate_rollouts(models, timestamps, rpm):
    eligible = valid_targets(timestamps, rpm, horizon=60)
    uniform = spaced_indices(eligible, timestamps, 600)
    active = spaced_indices(dynamic_targets(rpm, eligible), timestamps, 60)
    result = {"selection": "Origins are chosen from timestamps or past RPM only; every history and all 60 future timestamps must be contiguous.",
        "forecasting": "Recursive predictions use initial history only; no future measurements or commands are supplied.",
        "groups": {}}
    for group, origins in [("uniform_10_minute_origins", uniform), ("active_past_history_origins", active)]:
        initial = histories(rpm, origins)
        future = rpm[origins[:, None] + np.arange(60)]
        group_metrics = {"origin_count": len(origins), "models": {}}
        for name, model in models.items():
            forecasts = rollout(model, initial, 60)
            group_metrics["models"][name] = {str(h): metrics(future[:, h - 1], forecasts[:, h - 1]) for h in [1, 10, 30, 60]}
        result["groups"][group] = group_metrics
    return result


def make_plots(report, training_history, timestamps, rpm, indices, predictions, selected_name):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    history = pd.DataFrame(training_history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history.epoch, history.validation_rmse_rpm, color="#167a9b", label="MLP, MHU-1 validation")
    base = report["validation_metrics"]["persistence"]["rmse_rpm"]
    axes[0].axhline(base, color="#687787", linestyle="--", label="Persistence")
    axes[0].set(xlabel="Training epoch", ylabel="RMSE (RPM)", title="Checkpoint selected using validation only")
    axes[0].legend(fontsize=8)
    names = list(report["test_metrics"])
    errors = [report["test_metrics"][name]["all"]["rmse_rpm"] for name in names]
    labels = [name.replace("_residual", "").replace("_", " ") for name in names]
    axes[1].barh(labels, errors, color=["#f09238" if name == selected_name else "#689eb3" for name in names])
    axes[1].set(xlabel="Held-out MHU-2 RMSE (RPM)", title="Independent flight test, one second ahead")
    fig.suptitle("Measured MARS-S centrifuge speed forecasting", fontsize=14)
    fig.savefig(OUT / "validation_and_test.png", dpi=160)
    plt.close(fig)

    # Show the first eligible active history in source order, not the best-fit event.
    active = dynamic_targets(rpm, valid_targets(timestamps, rpm, 60))
    anchor = int(active[0])
    visible = np.arange(anchor - 8, anchor + 60)
    horizon_times = timestamps[anchor:anchor + 60] - timestamps[anchor - 1]
    forecast = rollout(report["selected_model_document"], histories(rpm, np.array([anchor])), 60)[0]
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(timestamps[visible] - timestamps[anchor - 1], rpm[visible], color="#202c3c", marker=".", label="Measured RPM")
    ax.plot(horizon_times, forecast, color="#d67a24", label=f"Recursive {selected_name}")
    ax.plot(horizon_times, np.repeat(rpm[anchor - 1], 60), linestyle="--", color="#687787", label="Persistence")
    ax.axvline(0, color="#9aa6b3", linewidth=1)
    ax.set(xlabel="Seconds after last input sample", ylabel="Rotor speed (RPM)",
           title="First eligible active-history test origin: recursive forecasts drift")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(OUT / "held_out_active_forecast.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (group, data) in zip(axes, report["recursive_metrics"]["groups"].items()):
        for name in ["persistence", selected_name]:
            scores = data["models"][name]
            ax.plot([1, 10, 30, 60], [scores[str(h)]["rmse_rpm"] for h in [1, 10, 30, 60]], marker="o", label=name)
        ax.set(xlabel="Forecast horizon (seconds)", ylabel="RMSE (RPM)",
               title=("Regularly spaced origins" if group.startswith("uniform") else "Active past histories") + f" (n={data['origin_count']:,})")
        if group == "active_past_history_origins":
            ax.set_yscale("log")
        ax.legend(fontsize=9)
    fig.suptitle("Recursive forecast: no future measurements", fontsize=14)
    fig.savefig(OUT / "recursive_forecast.png", dpi=160)
    plt.close(fig)


def main():
    start_time = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"fixed_before_training": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "history_samples": 8, "horizon_seconds": 1,
        "train": "MHU-1 first 70% of elapsed source time",
        "validation": "MHU-1 last 30%; entire input history must be inside validation",
        "test": "MHU-2 independent mission; first scored after model selection and saved checkpoints",
        "preprocessing": "Keep last duplicate; no interpolation; retain actual zero RPM; windows require 1-second cadence and finite data",
        "training_sample": "Up to 150000 uniformly spaced training windows, union every training window with past-history peak-to-peak RPM >2.0; no future-target selection",
        "candidates": ["persistence", "moving_mean_8", "ridge_alpha_1", "ridge_alpha_100", "mlp_residual_32_32_relu"],
        "selection_metric": "Full chronological validation RMSE in RPM",
        "mlp": {"hidden_layer_sizes": [32, 32], "activation": "relu", "solver": "adam", "alpha": 0.0001,
                "learning_rate_init": 0.001, "batch_size": 256, "max_epochs": 40, "patience": 8},
        "rollout": "1,10,30,60 s endpoints; fixed 600-second-spaced and past-history-active origins with a full contiguous future"}
    write_json(OUT / "training_plan.json", plan)
    train_time, train_rpm, train_meta = load_mission("MHU-1")
    targets = valid_targets(train_time, train_rpm)
    cutoff = int(train_time[0] + (train_time[-1] - train_time[0]) * 0.7)
    train_ids = targets[train_time[targets] < cutoff]
    val_ids = targets[train_time[targets - CONTEXT] >= cutoff]
    assert train_time[train_ids[-1]] < train_time[val_ids[0] - CONTEXT]
    active_train = dynamic_targets(train_rpm, train_ids)
    uniform_train = train_ids[np.linspace(0, len(train_ids) - 1, min(150_000, len(train_ids)), dtype=int)]
    fit_ids = np.union1d(uniform_train, active_train)
    fit_history = histories(train_rpm, fit_ids)
    x_scaler = StandardScaler().fit(make_features(fit_history))
    residual = train_rpm[fit_ids] - fit_history[:, -1]
    y_scaler = StandardScaler().fit(residual[:, None])
    x_train = x_scaler.transform(make_features(fit_history))
    y_train = y_scaler.transform(residual[:, None])[:, 0]
    val_truth = train_rpm[val_ids]
    models = {"persistence": document("persistence", "persistence"),
              "moving_mean_8": document("moving_mean_8", "moving_mean")}
    validation = {name: metrics(val_truth, predict_all(model, train_rpm, val_ids)) for name, model in models.items()}
    print(f"Eligible training: {len(train_ids):,}; sampled fit: {len(fit_ids):,}; validation: {len(val_ids):,}", flush=True)
    print(f"Persistence validation RMSE: {validation['persistence']['rmse_rpm']:.6f} RPM", flush=True)
    with threadpool_limits(limits=2):
        for alpha in [1.0, 100.0]:
            fitted = Ridge(alpha=alpha).fit(x_train, y_train)
            name = f"ridge_alpha_{int(alpha)}"
            models[name] = document(name, "ridge_residual", x_scaler, y_scaler, fitted)
            validation[name] = metrics(val_truth, predict_all(models[name], train_rpm, val_ids))
        fitted = MLPRegressor(hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
            alpha=0.0001, learning_rate_init=0.001, batch_size=256, random_state=SEED,
            shuffle=True, early_stopping=False, max_iter=1, tol=0.0)
        best_score, best_epoch, stale = float("inf"), 0, 0
        training_history = []
        for epoch in range(1, 41):
            fitted.partial_fit(x_train, y_train)
            candidate = document("mlp_residual", "mlp_residual", x_scaler, y_scaler, fitted)
            score = metrics(val_truth, predict_all(candidate, train_rpm, val_ids))["rmse_rpm"]
            training_history.append({"epoch": epoch, "train_standardized_residual_loss": float(fitted.loss_), "validation_rmse_rpm": score})
            if score < best_score - 1e-8:
                best_score, best_epoch, stale = score, epoch, 0
                best_model = copy.deepcopy(candidate)
                best_fitted = copy.deepcopy(fitted)
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0:
                print(f"MLP epoch {epoch:2d}: validation RMSE {score:.6f} RPM; best epoch {best_epoch}", flush=True)
            if stale >= 8:
                break
        models["mlp_residual"] = best_model
        validation["mlp_residual"] = metrics(val_truth, predict_all(best_model, train_rpm, val_ids))
    selected_name = min(validation, key=lambda name: validation[name]["rmse_rpm"])
    best_learned = min([name for name in models if name.startswith(("ridge", "mlp"))], key=lambda name: validation[name]["rmse_rpm"])
    # Save every candidate and the frozen decision before opening/scoring the test mission here.
    for name, model in models.items():
        write_json(OUT / f"{name}.json", model)
    write_json(OUT / "selected_model.json", models[selected_name])
    selection = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selected_name": selected_name,
        "best_learned_name": best_learned, "selection_metric": "Full MHU-1 validation RMSE",
        "validation_metrics": validation, "selected_checkpoint_sha256": sha256(OUT / "selected_model.json"),
        "mlp_best_epoch": best_epoch, "epochs_run": len(training_history)}
    write_json(OUT / "selection_frozen_before_test.json", selection)
    pd.DataFrame(training_history).to_csv(OUT / "mlp_training_history.csv", index=False)
    print(f"Frozen model selected using validation only: {selected_name}. Loading independent MHU-2 test.", flush=True)
    test_time, test_rpm, test_meta = load_mission("MHU-2")
    test_ids = valid_targets(test_time, test_rpm)
    test_truth = test_rpm[test_ids]
    active_ids = dynamic_targets(test_rpm, test_ids)
    active_mask = np.isin(test_ids, active_ids)
    new_change_mask = np.abs(test_truth - test_rpm[test_ids - 1]) > 2.0
    scores, predictions = {}, {}
    with threadpool_limits(limits=2):
        for name, model in models.items():
            predictions[name] = predict_all(model, test_rpm, test_ids)
            scores[name] = {"all": metrics(test_truth, predictions[name]),
                "active_past_history": metrics(test_truth[active_mask], predictions[name][active_mask]),
                "stable_past_history": metrics(test_truth[~active_mask], predictions[name][~active_mask]),
                "actual_next_change_over_2_rpm_reporting_only": metrics(test_truth[new_change_mask], predictions[name][new_change_mask])}
            print(f"{name}: MHU-2 RMSE {scores[name]['all']['rmse_rpm']:.6f} RPM, MAE {scores[name]['all']['mae_rpm']:.6f} RPM", flush=True)
        recursive = evaluate_rollouts(models, test_time, test_rpm)
    # Verify exported neural weights against the fitted sklearn checkpoint on held-out histories.
    verify_ids = test_ids[np.linspace(0, len(test_ids) - 1, 256, dtype=int)]
    verify_history = histories(test_rpm, verify_ids)
    sklearn_prediction = verify_history[:, -1] + y_scaler.inverse_transform(best_fitted.predict(x_scaler.transform(make_features(verify_history)))[:, None])[:, 0]
    portable_prediction = predict(json.loads((OUT / "mlp_residual.json").read_text()), verify_history)
    max_difference = float(np.max(np.abs(sklearn_prediction - portable_prediction)))
    if max_difference > 1e-10:
        raise ValueError(f"Portable model verification failed: {max_difference}")
    example_id = int(valid_targets(test_time, test_rpm, 60)[0])
    write_json(OUT / "example_history_rpm.json", {"mission": "MHU-2", "history_rpm": histories(test_rpm, np.array([example_id]))[0].tolist(), "sample_interval_seconds": 1})
    write_json(OUT / "inference_test_vectors.json", {"histories_rpm": verify_history[:5].tolist(), "mlp_next_rpm": portable_prediction[:5].tolist()})
    # Small deterministic diagnostic extract; full test scores above use every eligible window.
    display_positions = np.union1d(np.arange(0, len(test_ids), 600), np.flatnonzero(active_mask))
    extract = pd.DataFrame({"timestamp_as_recorded": test_time[test_ids[display_positions]].astype("datetime64[s]").astype(str),
        "measured_next_rpm": test_truth[display_positions], "active_past_history": active_mask[display_positions]})
    for name, prediction in predictions.items():
        extract[name] = prediction[display_positions]
    extract.to_csv(OUT / "held_out_predictions_sample.csv", index=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": "Actual ISS MARS-S measured RPM forecasting",
        "selected_model": selected_name, "best_learned_model": best_learned,
        "hardware": {"name": "MARS-S", "host": "ISS Kibo CBEF", "nominal_radius_m": 0.15, "nominal_setting_rpm": 77,
                     "configuration_source": "https://ibsls.megabank.tohoku.ac.jp/overview"},
        "source_publication": "https://doi.org/10.1038/s41467-026-76349-y",
        "source_missions": [train_meta, test_meta], "fixed_plan": plan,
        "split": {"cutoff_timestamp_as_recorded": str(np.datetime64(cutoff, "s")),
            "train_eligible_windows": len(train_ids), "train_fitted_windows": len(fit_ids),
            "uniform_training_windows": len(uniform_train), "active_history_training_windows": len(active_train),
            "validation_windows": len(val_ids), "test_windows": len(test_ids),
            "test_active_past_history_windows": int(active_mask.sum()),
            "training_target_last": str(np.datetime64(int(train_time[train_ids[-1]]), "s")),
            "validation_input_first": str(np.datetime64(int(train_time[val_ids[0] - CONTEXT]), "s"))},
        "validation_metrics": validation, "mlp_best_epoch": best_epoch, "epochs_run": len(training_history),
        "test_metrics": scores, "recursive_metrics": recursive,
        "verification": {"no_train_validation_window_overlap": True, "no_cross_gap_windows": True,
            "no_synthetic_or_interpolated_labels": True, "train_only_normalization": True,
            "test_mission_scored_only_after_frozen_selection": True,
            "portable_mlp_max_abs_difference_rpm_on_256_test_windows": max_difference},
        "limitations": [
            "Two mission histories, not millions of independent experiments; stable quantized speeds dominate.",
            "Training oversamples windows with changing PAST speed; full validation/test metrics are unweighted and unsampled.",
            "The export contains measured RPM and time only; start/stop commands and maintenance schedule are unobserved.",
            "Predicts already controlled rotor behavior. Does not identify motor torque, inertia, hub dynamics, attitude, thrust, or a 100 m crew habitat.",
            "A derived r*omega^2 radial component at nominal 0.15 m is not an independent acceleration measurement or whole-body exposure measurement.",
            "No evidence establishes transfer to another radius, controller, artificial-gravity setting, or human spacecraft.",
            "Primary metrics are unclipped. Negative and unstable recursive predictions must not be used to command hardware.",
            "No statistical confidence claim follows from the overlapping, autocorrelated windows."],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "cpu_threads": 2,
            "elapsed_seconds_before_plots": time.perf_counter() - start_time}}
    report["use_decision_after_test"] = {
        "supported_experiment": "One-second forecast from eight fresh measured RPM samples",
        "recursive_forecasting_status": "FAILED: selected MLP can diverge dramatically during active histories",
        "integration": "Display one-step forecasts only; require fresh measured inputs; keep recursive results as a diagnostic",
        "does_not_change_frozen_model_selection": True,
        "physical_limits": "Raw regression output can be negative near stopped states; retain it in metrics, mark it invalid in a physical display, and suppress derived gravity for invalid negative-speed predictions; never use it as a command"}
    # Attach the observed usage limitation without changing any fitted weights or selection.
    selected_document = dict(models[selected_name], use_decision_after_test=report["use_decision_after_test"])
    write_json(OUT / "selected_model_with_usage_notes.json", selected_document)
    write_json(OUT / "training_report.json", report)
    plot_report = dict(report, selected_model_document=models[selected_name])
    make_plots(plot_report, training_history, test_time, test_rpm, test_ids, predictions, selected_name)
    print(f"Completed actual measured-data RPM training in {time.perf_counter() - start_time:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
