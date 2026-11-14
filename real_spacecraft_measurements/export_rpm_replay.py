"""Export small, real MHU-2 clips for an archived-measurement viewer.

Forecasts are always generated from fresh measured history. Gaps remain gaps.
Clip selection is illustrative and does not change the already-frozen evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from rpm_predict import predict, nominal_radial_g

ROOT = Path(__file__).resolve().parent


def main():
    raw = ROOT / "data/raw/mhu/41467_2026_76349_MOESM16_ESM.zip"
    with zipfile.ZipFile(raw) as archive, archive.open("SupplementaryData13.txt") as handle:
        frame = pd.read_csv(handle, sep="\t", usecols=["Date", "RPM"])
    frame["timestamp"] = pd.to_datetime(frame["Date"], format="mixed")
    frame = frame.drop_duplicates("timestamp", keep="last")
    timestamps = frame.timestamp.to_numpy(dtype="datetime64[s]").astype(np.int64)
    rpm = frame.RPM.to_numpy(dtype=float)
    model = json.loads((ROOT / "outputs/rpm/selected_model_with_usage_notes.json").read_text())
    first_drop = int(np.flatnonzero(np.diff(rpm) < -3)[0] + 1)
    ranges = [("initial_recorded_spin_up", timestamps[0], timestamps[0] + 180),
              ("early_steady_operation", timestamps[0] + 600, timestamps[0] + 780),
              ("first_recorded_slowdown", timestamps[first_drop] - 30, timestamps[first_drop] + 150)]
    clips = []
    for name, start, end in ranges:
        ids = np.flatnonzero((timestamps >= start) & (timestamps < end))
        points = []
        for idx in ids:
            row = {"timestamp_as_recorded": str(np.datetime64(int(timestamps[idx]), "s")),
                   "elapsed_seconds": int(timestamps[idx] - start),
                   "measured_rpm": float(rpm[idx]),
                   "derived_nominal_radial_g_from_measured_rpm": float(nominal_radial_g(rpm[idx])),
                   "forecast_for_this_timestamp_rpm": None,
                   "derived_nominal_radial_g_from_forecast": None,
                   "forecast_status": "insufficient_contiguous_measured_history"}
            if idx >= 8 and timestamps[idx] - timestamps[idx - 8] == 8 and np.isfinite(rpm[idx - 8:idx + 1]).all():
                forecast = float(predict(model, rpm[idx - 8:idx])[0])
                row["forecast_for_this_timestamp_rpm"] = forecast
                row["forecast_issued_at_recorded_timestamp"] = str(np.datetime64(int(timestamps[idx - 1]), "s"))
                row["forecast_status"] = "invalid_negative_rpm" if forecast < 0 else "one_step_from_fresh_measured_history"
                if forecast >= 0:
                    row["derived_nominal_radial_g_from_forecast"] = float(nominal_radial_g(forecast))
            points.append(row)
        clips.append({"name": name, "points": points})
    output = {"data_kind": "Archived real flight measurements with independent-model one-step predictions",
              "mission": "MHU-2", "flight_year": 2017,
              "timestamp_timezone": "Unspecified in raw source; preserve recorded labels",
              "nominal_radius_m": 0.15, "sample_interval_seconds": 1,
              "display_rule": "A missing second is a gap. Do not interpolate new observations. Forecasts at each timestamp use actual preceding measurements, not prior predictions.",
              "hardware_scope": "Small MARS-S centrifuge aboard ISS, not a crew ring",
              "source_archive": str(raw.relative_to(ROOT)), "clips": clips}
    target = ROOT / "outputs/rpm/replay_mhu2.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(x['points']) for x in clips)} measured records in {len(clips)} clips to {target}")


if __name__ == "__main__":
    main()
