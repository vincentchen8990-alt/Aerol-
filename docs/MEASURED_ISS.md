# Archived ISS measurements: MARS-S RPM and NASA PIMS acceleration

This mode adds **recorded International Space Station data** and two **already-trained, frozen
measured-data models** to Orbital Lab. It is separate from the hypothetical 100 m ring simulator,
which is unchanged. Everything here is archived (2016/2017 MARS-S flight logs; 1–11 Nov 2026 NASA
PIMS records) — **not live telemetry**.

| | MARS-S centrifuge RPM | NASA PIMS / MAMS acceleration |
|---|---|---|
| Hardware | JAXA MARS-S small-animal centrifuge, ISS Kibo CBEF; nominal radius 0.15 m, nominal setting 77 RPM | MAMS OSS bias-compensated trimmed-mean-filtered accelerometer (sensor `ossbtmf`) |
| Recorded | MHU-1 2016 (training/validation), MHU-2 2017-08-17 → 2017-09-16 (independent test); timestamps as recorded, timezone unspecified | 2026-11-01 → 2026-11-11 UTC; frame `SSAnalysis` |
| Model input | 8 measured RPM samples, oldest → newest, shape `(8,)` (or batched `(N, 8)` in Python), exactly 1 s apart | 8 three-axis samples in standard gravity **g**, shape `(8, 3)` (or `(N, 8, 3)`), exactly 16 s apart (112 s span) |
| Output | next recorded RPM, 1 s ahead | next three-axis sample in g, 16 s ahead (displayed in µg = g × 10⁶) |
| Held-out result | MHU-2 one-step RMSE **0.245751 RPM** vs persistence 0.474178 | 10–11 Nov one-step RMSE **0.170533 µg** vs persistence 0.225622; Ridge 0.170794 (almost tied; Ridge has lower MAE — no claim of neural superiority) |
| Known failure | recursive forecasts unstable/divergent (60 s RMSE 5.88 vs persistence 1.39 RPM); 66,222 of 2,428,381 one-step outputs (2.73%) negative | — |
| Not | a torque-response model, inertia model, motor controller, simulator or health model | whole-station rigid-body motion, all vibration frequencies, or simultaneous/co-located with MARS-S |

## Where things are

| Path | Role |
|---|---|
| `real_spacecraft_measurements/` | Handoff package, used in place. Every served file is checked against `CORE_TRANSFER_MANIFEST.json` (SHA256 + bytes) before use. Override with `GRAVITY_MEASURED_ROOT`. |
| `real_spacecraft_measurements/data/raw/mhu/41467_2026_76349_MOESM16_ESM.zip` | MHU-2 source archive restored from the publisher URL; SHA256 matches `download_manifest.json`. Used only for verification. |
| `real_spacecraft_measurements/reacquired/pims/` | NASA PIMS 1–11 Nov 2026 reacquired with the unmodified `pims_acquire.py`, output root redirected so the **original recorded manifests were not overwritten**. |
| `outputs/measured_iss/pims_replay_nov2026.json` | Two short clips of real 16-s samples, built by `scripts/build_pims_replay.py` only after the reacquired table proved byte-identical to the training dataset. Override with `GRAVITY_PIMS_REPLAY`. |
| `gravity_ai/measured/` | `assets.py` (integrity checks), `inference.py` (NumPy inference, shape/unit/timestamp validation), `service.py` (one-step results, replays, status). |
| `gravity_ai/api/measured_iss.py` | API router under `/api/measured`. |
| `frontend/offline/iss_measured.js` | Archived-measurement mode of the built-in viewer (tab “ISS archived measurements”, or `/#iss`). |
| `scripts/verify_measured_iss.py` | Re-runs the integration verification and prints a JSON report. |
| `tests/test_measured_iss.py` | Integration tests. |

The simulated-dynamics checkpoint (`outputs/runs/selected_dynamics.json` → `dyn-v2-residual-e200-p20`,
13 inputs / 5 outputs) is not modified, reloaded or reused; the measured models have their own
identifiers (`mars_s_rpm_next_second_mlp`, `iss_mams_ossbtmf_next_16s_mlp`) and files.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/measured/status` | Mode (`ARCHIVED_MEASUREMENTS`, `live_telemetry: false`), model availability + SHA256, metrics read from the training reports, replay availability, evaluation note |
| `GET /api/measured/hardware` | `hardware_profiles.json` with provenance, plus the derived nominal radial g at 77 RPM (0.9945 g) |
| `POST /api/measured/rpm/predict` | `{"history_rpm": [8 values], "timestamps": [8 ISO strings 1 s apart]}` → one-step forecast |
| `POST /api/measured/pims/predict` | `{"history_g": [[ax,ay,az] × 8], "timestamps_utc": [8 explicit-UTC strings 16 s apart]}` → one-step forecast in g |
| `GET /api/measured/rpm/replay` | MHU-2 clips with server-computed one-step forecasts |
| `GET /api/measured/pims/replay` | PIMS clips with server-computed one-step forecasts |
| `GET /api/models` → `measured_iss` | Compact availability entry, separate from `learned_dynamics_predictor` |

Responses carry the model id/file/SHA256, units, cadence, horizon, history timestamps, forecast
timestamp, provenance, `valid`/`status`, and limitations.

Rules enforced by the backend:

* **One step only.** `steps` other than 1 → HTTP 422 `one_step_only`; no rollout endpoint exists.
* **Timestamps prove the cadence.** Missing, mis-counted, unordered, duplicated, mixed-timezone or
  gapped timestamps → HTTP 422 (`invalid_cadence`, `invalid_timestamps`, `timestamps_required`,
  `timestamp_timezone`) with `forecast: null`. PIMS timestamps must be explicit UTC.
* **Shapes.** RPM `(8,)`; PIMS `(8, 3)`. Anything else (`(7,)`, `(8,1)`, `(3,8)`, `(24,)`, …) → 422 `invalid_shape`.
* **Units.** PIMS inputs above 1e-3 g are refused as `units_suspect` (micro-g passed as g); recorded
  values are about ±5e-6 g. Values are returned in g; the viewer multiplies by 1e6 for µg and applies no other conversion.
* **Negative RPM forecasts** → `valid: false`, `status: invalid_negative_rpm`, `forecast_rpm: null`,
  derived g suppressed, raw unclipped value in `diagnostics.raw_model_output_rpm`. Negative *measured* RPM input is refused.
* **Unavailable, never substituted.** A missing or altered model (SHA256 mismatch) → HTTP 503
  `model_unavailable`; replays then show measured values with `forecast_status: model_unavailable`. A missing replay → 503 `replay_unavailable`.
* **Gaps.** In replays, a forecast exists only where the 8 preceding samples and the target are
  contiguous at the model cadence; after a gap the history restarts (`history_reset_reason: after_gap`).
  Nothing is interpolated and forecasts are never fed back as inputs.

**Derived nominal radial g** = r·(2π·RPM/60)²/9.80665 with r = 0.15 m. It is a physics calculation
at the published nominal radius — not directly measured gravity, crew exposure or a learned health prediction.

## Viewer

Open `http://127.0.0.1:8001/#iss` (or click **ISS archived measurements**). The header tag reads
*ARCHIVED MEASUREMENTS · NOT LIVE*. Panels:

1. **ISS host · schematic plan view** — one scale for the four NASA reference extents (109 m
   solar-array span, 73 m solar-array longitudinal extent, 94 m truss, 67 m pressurized-module major
   axis), labelled *schematic, not CAD*; element widths/positions are illustrative and the 109 m span is
   marked as not a ring radius. MARS-S appears in an enlarged inset (magnification shown); its rotor
   animates at a disclosed display factor (default 1/10 of the measured RPM) only while the replay plays,
   and every number shows the recorded RPM.
2. **MARS-S measured RPM replay** — three MHU-2 clips; measured RPM, one-step forecast for each second,
   error, status, derived nominal radial g from measured and forecast RPM, recorded gaps shaded, invalid
   negative outputs drawn as red crosses at their raw value, raw values in Diagnostics.
3. **NASA PIMS measured acceleration** — separate panel and dataset; µg per axis in the SSAnalysis frame,
   16-s samples only, one-step forecasts, recorded gap clip.
4. **Measured models / Hardware reference / What this mode is not.**

## Verification performed (2026-11-14)

Re-run with `python -m scripts.verify_measured_iss` (JSON report, exit code 1 on failure).

* **Handoff integrity:** all 41 files in `CORE_TRANSFER_MANIFEST.json` match SHA256 and byte count
  (re-checked after all work; no handoff file was modified).
* **Test-vector parity:** RPM 5/5 vectors, max |Δ| 0 (batched) / 7.1e-15 RPM (single rows); PIMS 8/8
  vectors, max |Δ| 0 (batched) / 2.1e-22 g (single rows). Against the handoff's own `rpm_predict.py` on
  513 replay histories and `outputs/pims/inference.py`: max |Δ| 0. Both handoff CLIs were run and agree.
* **MHU-2 source archive** (restored, 7,154,515 bytes, SHA256 `c416c8dc…0c7790` = recorded): 2,545,141
  unique timestamps (= training report); all 537 replay measurements equal the archive; each clip's gap
  second is genuinely absent from the archive; the 16 exported clip-start forecasts reproduce exactly
  from the archive's preceding measured seconds. Server-side replay forecasts equal the exported ones
  to ≤ 1.4e-14 RPM, including all 92 invalid negative outputs in the slowdown clip.
* **NASA PIMS reacquisition** (1–11 Nov 2026, run 2026-11-14 through a wrapper that redirects
  `pims_acquire.py`'s output root): 268/268 binary/header files match the SHA256s in the original,
  untouched `source_manifest.json`; the reconstructed table (59,062 rows, 5 segments) has SHA256
  `50747a1e…5ad74b`, identical to the model's training dataset. Replay clips: 150 samples from the
  held-out test split (2026-11-10 00:00:07 → 00:39:51 UTC) and 100 samples across a real 45-sample gap in
  the validation split (2026-11-09 01:12:39 → 01:51:03 UTC); all 250 rows are exact table rows.
* **Training scripts** (`rpm_train.py`, `pims_train.py`, `pims_acquire.py`, `export_rpm_replay.py`)
  compile under Python 3.11 and 3.14 (in-memory, not executed). No retraining or retuning was done.
* **Tests:** `python -m pytest -q` → 95 passed (58 pre-existing + 37 in `tests/test_measured_iss.py`).
* **Browser (headless Chrome, served app):** 22/22 checks — ring mode default and unchanged; archived
  labels, mission/dates, NASA dimensions, metrics and display-speed disclosure; valid forecast readouts
  match the API; clip-start and post-gap history resets; gap seconds show no measurement; invalid
  negative forecast with suppressed g and raw value in diagnostics; replay advances and the rotor animates
  while numeric RPM stays the recorded value; PIMS µg = g × 10⁶, UTC forecast, gap reset; switching back to
  the ring; no unexpected console/network errors.

## Retraining and new claims

Do not retrain or retune by default. If the architecture changes or the models are retrained later,
record that **MHU-2 (and the 10–11 Nov 2026 PIMS test days) have already been examined**, and reserve a
new, defensible evaluation dataset before making new performance claims. `rpm_train.py` has no argument
parser (running it with `--help` starts training); full reproduction used Python 3.12 with the pinned
`requirements-training.txt` in an isolated environment, which is not installed here.
