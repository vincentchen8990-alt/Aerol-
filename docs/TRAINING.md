# Training: from the user's files to the models

## 1. Check environment

```bash
python -c "import torch; print('CUDA:',torch.cuda.is_available())"
python -m gravity_ai.data.catalog_loader --validate
```

You need Python >=3.10, packages in `requirements.txt`, and network access for raw human data. The original four CSV files are **catalogs**, never labels or raw physiological observations.

## 2. Download and prepare actual human data

```bash
python -m gravity_ai.data.download --from-catalog --public-only --dataset pilot
python -m gravity_ai.data.prepare --dataset all
cat data/manifests/preparation_report.json
```

A successful preparation should show participant counts and output paths. If it reports missing data, check the download report and the host's current access/license. Do not train on a fake replacement.

## 3. Train the wearable physiological model

After at least five different participants' genuine E4 `HR.csv` and companion data are available:

```bash
python -m gravity_ai.train.physiology --config configs/physiology.yaml
```

This learns to predict the next short-window PPG-derived heart-rate measurement from earlier E4 features, using a **participant-held-out evaluation split**. It compares a regularized ridge model and a histogram gradient boosting model. It **does not** learn a gravity-dose effect.

Optional time-series encoder comparison:

```bash
python -m gravity_ai.train.sequence --model gru --epochs 12
python -m gravity_ai.train.sequence --model lstm --epochs 12
python -m gravity_ai.train.sequence --model transformer --epochs 12
```

The code preserves complete recorded sessions and participant-holdouts, and uses the training subjects only to fit the scaler. Models and results are written to `outputs/checkpoints/`.

## 4. Train the NHANES skeletal model

```bash
python -m gravity_ai.data.download --from-catalog --public-only --dataset NHANES2013
python -m gravity_ai.data.prepare --dataset nhanes
python -m gravity_ai.train.bone --config configs/bone.yaml
```

The BMD model estimates **observed cross-sectional DXA BMD** using available NHANES features. Do not use the result as longitudinal bone treatment response. For population inference use the CDC survey weights and design guidelines; this prototype's standard train/test evaluation is predictive, not weighted national inference.

## 5. Train the spacecraft dynamics predictor (dynamics-v2, simulated data only)

The predictor learns how the simulated ring/hub spacecraft responds to motor commands: given the
current state, the command, thrust, spacecraft parameters and time step, it predicts the next ring
and hub speeds and the crew's apparent-gravity vector. Labels come from the analytical simulator
(`gravity_ai.simulation.physics.step`), which remains the ground truth. This is **not** a human,
medical or flight-qualified model, and it does not decide which gravity level is appropriate for anyone.

All commands below were executed successfully on 14 Nov 2026 (Windows, PowerShell, `.venv` activated,
CPU-only PyTorch 2.15.0+cpu). Settings live in `configs/dynamics.yaml`; precedence is built-in
defaults < YAML < command line, and unknown or invalid keys are rejected. (YAML 1.1 note: write
`1.0e+9`, not `1.0e9`, which PyYAML reads as a string; the loader rejects it.)

```powershell
# 1. Generate the frozen benchmark: 500 episodes x 120 steps = 60,000 transitions (~2 s)
python -m gravity_ai.simulation.generate

# 2. Train (from scratch). The selected run used a longer bounded budget than the YAML default:
python -m gravity_ai.train.dynamics --run-id dyn-v2-residual-e200-p20 --epochs 200 --patience 20

# 3. Fresh evaluation: reloads checkpoint_best.pt from disk, recomputes on held-out test episodes
python -m gravity_ai.evaluate_dynamics --run outputs/runs/dyn-v2-residual-e200-p20

# 4. Inference example (latest completed run), compared with the simulator for the same input
python -m gravity_ai.predict_dynamics

# 5. Collect results (does not retrain or re-evaluate)
python -m gravity_ai.evaluate --all
```

Also executed: the YAML-default run `python -m gravity_ai.train.dynamics --run-id dyn-v2-residual-e50`
(50-epoch limit, patience 8), and the ablation with the archived formulation
`python -m gravity_ai.train.dynamics --run-id dyn-v2-absolute-e50 --target-mode absolute`.

Smoke test and resume (executed on a separate 80-episode dataset):

```powershell
python -m gravity_ai.simulation.generate --episodes 80 --output-dir data/synthetic/dynamics_smoke --overwrite
python -m gravity_ai.train.dynamics --dataset-dir data/synthetic/dynamics_smoke --output-root outputs/smoke_runs --run-id smoke-1 --epochs 3
python -m gravity_ai.train.dynamics --output-root outputs/smoke_runs --resume outputs/smoke_runs/smoke-1 --epochs 5
python -m gravity_ai.evaluate_dynamics --run outputs/smoke_runs/smoke-1
```

`--resume` restores model weights, optimizer state, shuffle-generator state, normalisation,
history and early-stopping counters from `training_state_last.pt`, refuses a changed dataset or
changed seed/batch size/learning rate/weight decay/target mode/architecture, and records the
resume in `run_manifest.json`. A run started without `--resume` always starts from scratch.

### Data (`data/synthetic/dynamics_v2/`)

| File | Content |
|---|---|
| `transitions.npz` | every transition in physical SI units: inputs, simulator truth, episode/step ids, scenario, split |
| `split_manifest.json` | the exact episode -> train/val/test assignment |
| `generation_report.json` | config, content fingerprint, counts per split/time step/scenario, value ranges, rejected attempts |

* Scenarios: `mission_pid` (PID tracks a 0.3-1.5 g target through accelerate/coast/brake thrust),
  `setpoint_steps` (PID spin-up/spin-down between 0.1-1.6 g targets under changing thrust) and
  `random_torque` (open-loop piecewise commands incl. saturation, kept within 0.15-1.8 g).
* Spacecraft parameters are sampled log-uniformly; app defaults lie inside the ranges. Motor torque,
  power and slew limits are now model inputs (they were hidden, and fixed at non-app values, in v1).
* Time steps 0.5, 1, 2 (API default) and 5 s; episode *i* uses `dt_values_s[i % 4]`.
* Episodes failing physical checks are resampled: non-finite values, ring stopped/reversed, crew
  gravity > 3 g, angular momentum not conserved, misaligned input/target. (3 attempts rejected.)
* Splits are stratified by time step; within each time step every block of 20 episodes is 14/3/3.
  Episode content and split depend only on the seed and episode id, so generating more episodes
  never moves an existing validation/test episode into training. The generator refuses to overwrite
  an existing dataset unless `--overwrite` is given.
* Observation model: clean simulator truth. No sensor noise, delay or faults were added or evaluated.

Result: 60,000 transitions; 349/76/75 train/val/test episodes (41,880/9,120/9,000 rows);
86-88 train and 18-20 test episodes per time step.

### Features, targets and preprocessing (`gravity_ai/models/dynamics_schema.py`)

Inputs, in order (raw unit -> divided by): ring speed (rad/s, 1), hub speed (rad/s, 1), applied
torque (N*m, 1e7), requested torque (N*m, 1e7), axial acceleration (m/s^2, g0), ring inertia
(kg*m^2, 1e10), hub inertia (kg*m^2, 1e10), bearing drag (N*m*s/rad, 1e6), radius (m, 100), dt
(s, 1), torque limit (N*m, 1e7), power limit (W, 1e7), slew limit (N*m/s, 1e6). Then
standardisation with training-split mean/std.

Outputs: next ring speed, next hub speed (rad/s), crew radial, tangential, axial acceleration
(m/s^2). With `target_mode: residual` (default) the network predicts `output - reference`, where the
references are the current ring/hub speeds, `radius * ring_speed^2`, 0, and `-axial_accel`; targets are
standardised with training-split statistics. The axial component equals the known thrust exactly,
so its residual is identically zero.

Why residual: on its own archived test set, the archived absolute-target model was worse than "no
change" persistence (ring speed MAE 0.00288 vs 0.00111 rad/s; radial 0.59 vs 0.11 m/s^2). On the
v2 benchmark the same formulation (`dyn-v2-absolute-e50`) was again worse than persistence for speeds
and drifted to 0.73 g after 60 s at dt = 0.5 s, versus 0.03 g for the residual model.

Architecture (unchanged): MLP 13 -> 64 -> 64 -> 64 -> 5, SiLU, 9,541 parameters; AdamW
(lr 0.0015, weight decay 1e-4), batch 256, MSE on standardised targets, seed 17; device `auto`.

### Run artifacts (`outputs/runs/<run_id>/`)

`checkpoint_best.pt` (weights, normalisation, schema with units/scales, envelope, dataset
fingerprint), `training_state_last.pt` (for `--resume`), `run_manifest.json` (effective config and
its sources, dataset and split hashes, seeds, versions, device, history, elapsed time, stop reason),
`evaluation/evaluation.json` and plots (`training_loss.png`, `pred_vs_actual_speeds.png`,
`pred_vs_actual_accel.png`, `rollout_error_growth.png`, `rollout_example.png`).
`outputs/runs/latest_dynamics.json` points to the most recently completed run (used by the API and
`predict_dynamics`). The archived v1 model in `outputs/checkpoints/` is left unchanged.

### Selected run and results (`dyn-v2-residual-e200-p20`)

Selected on validation one-step error among the three runs above. Early stopping after 94 epochs
(best epoch 74, validation MSE 0.0205), 21.7 s on CPU (i7-11800H, 8 threads). Held-out **test**
episodes (75 episodes, 9,000 rows; 417 rows lie slightly outside the per-feature training min/max):

| Output | Learned MAE | Learned RMSE | Persistence MAE | Ridge MAE | Unit |
|---|---|---|---|---|---|
| next ring speed | 0.000238 (0.0023 rpm) | 0.000575 | 0.00126 | 0.000954 | rad/s |
| next hub speed | 0.000117 (0.0011 rpm) | 0.000226 | 0.000535 | 0.000551 | rad/s |
| radial acceleration | 0.0172 (0.0018 g) | 0.0457 | 0.0855 | 0.0720 | m/s^2 |
| tangential acceleration | 0.0140 (0.0014 g) | 0.0395 | 0.0912 | 0.0671 | m/s^2 |
| axial acceleration | 0.0078 (0.0008 g) | 0.0096 | 0 (exact) | 0 (exact) | m/s^2 |
| total gravity magnitude | 0.0162 (0.0017 g) | 0.0425 | 0.0794 | 0.0650 | m/s^2 |

Rollouts (recursive, from start points every 10 steps in test episodes; known commands and thrust
replayed open-loop; speeds fed back from predictions; motor torque advanced with the explicit motor
model; no future simulator state used). End-of-horizon MAE, learned / persistence / ridge:

| dt | horizon | rollouts | ring speed (rpm) | gravity magnitude (g) |
|---|---|---|---|---|
| 0.5 s | 10 / 30 / 60 s | 198 / 126 / 18 | 0.010 / 0.026 / 0.053 (persistence 0.047 / 0.119 / 0.211) | 0.006 / 0.016 / 0.033 (persistence 0.024 / 0.057 / 0.107) |
| 1 s | 10 / 30 / 60 s | 228 / 190 / 133 | 0.027 / 0.064 / 0.088 (persistence 0.085 / 0.206 / 0.296) | 0.017 / 0.041 / 0.055 (persistence 0.057 / 0.137 / 0.190) |
| 2 s | 10 / 30 / 60 s | 240 / 220 / 200 | 0.009 / 0.023 / 0.041 (persistence 0.060 / 0.163 / 0.247) | 0.005 / 0.014 / 0.025 (persistence 0.038 / 0.102 / 0.154) |
| 5 s | 10 / 30 / 60 s | 216 / 216 / 198 | 0.006 / 0.017 / 0.030 (persistence 0.044 / 0.127 / 0.232) | 0.004 / 0.012 / 0.020 (persistence 0.032 / 0.089 / 0.164) |

The ridge baseline was worse than the learned model in every rollout setting (full numbers in
`evaluation.json`). Error grows with horizon; an open-loop rollout of the complete default
1,200 s mission (600 steps) reached 0.33 g error, so long open-loop predictions are not reliable.
The dt = 1 s weakness is concentrated in 3 aggressive `random_torque` test episodes; with 3 test
episodes per time-step/scenario cell these differences are not statistically robust. Errors are
highest when the motor power limit is active.

### Application connection

The Orbital Lab viewer **displays** the learned model's predictions next to the simulated physics.
The PID (default) or MPC controller still selects every motor command; the learned model never does.

* **Which checkpoint is served:** `outputs/runs/selected_dynamics.json` (a deliberate choice,
  currently `dyn-v2-residual-e200-p20`; training never changes it). Fallbacks: the environment
  variable `GRAVITY_DYNAMICS_RUN_DIR` overrides it, and without a selection the most recent
  completed run (`latest_dynamics.json`) is used. `python -m gravity_ai.predict_dynamics` uses the same rule.
* `GET /api/models` -> `learned_dynamics_predictor`: checkpoint exists / loads (with the load error),
  run id, epoch, selection source, supported time steps, per-feature training ranges, evaluated
  forecast horizons, `used_by_viewer`, `used_by_api_simulate: false`, `used_for_control: false`.
* `POST /api/simulate` is unchanged.
* `POST /api/dynamics/compare` takes the `/api/simulate` body plus `forecast_start_s` (default 0),
  `forecast_horizon_s` (default 60; `null` = rest of mission) and `allow_unsupported_time_step`
  (default false). It runs the same PID/MPC mission (same initial state, time step, parameters and
  commands), then returns:
  * `series`: per-step simulator reference, **one-step** prediction (from the simulated state one
    step earlier) and a whole-mission open-loop rollout, at the same timestamps as `/api/simulate`
    with the same `max_points`;
  * `forecast`: an **open-loop forecast** window starting from the simulated state at
    `forecast_start_s`, fed only its own predictions, with errors at the end and the maximum in the window;
  * `training_envelope`: which inputs fall outside the recorded training ranges (trained min/max, units);
  * `mission` (echo of the request) and `active_controller`.
  Errors: 422 `unsupported_time_step` (dt not in 0.5/1/2/5 s), 422 `invalid_forecast` (start or
  horizon not a whole number of steps, start at or after mission end), 503 `model_unavailable`.
  Windows running past the mission end are shortened and flagged `truncated_at_mission_end`.

**Viewer panel "Learned Model vs Simulated Physics"** (http://127.0.0.1:8001/ when the server runs on 8001):

* Controls added to *Spacecraft Configuration*: controller (PID/MPC) and simulation time step
  (0.5, 1, 2, 3, 5, 10 s; options are labelled from the model metadata as model-trained or not).
* Badges: model status and run id, the controller selecting the commands, and "Reference: simulated
  physics — not measured spacecraft data". Compact *Model diagnostics* (checkpoint, selection source,
  trained time steps and ranges) are in a collapsible section.
* **1 · One-step prediction**: simulated vs predicted ring RPM and crew acceleration magnitude (G)
  and the signed error (learned − simulated) at the playback time; chart over the mission (RPM or G)
  with an error band; mean error for the mission.
* **2 · Open-loop forecast**: horizon 10 / 30 / 60 s or rest of mission, starting at 0 s or at the
  playback time; values and error at the end of the window; chart with error growth. The text states
  that the recorded controller commands and thrust schedule are replayed unchanged, that the forecast
  is never reset to the simulator, and, for horizons beyond 60 s, that drift grows with horizon
  (default mission, rest-of-mission forecast: about 0.33 G error at 1,200 s).
* Unsupported time steps show a notice and no prediction values; settings outside the recorded
  training ranges show an extrapolation warning naming the inputs and ranges; inside the ranges the
  notice says this is not a guarantee of accuracy. A missing or unloadable checkpoint, a failed
  request, or a server still running older code (the page is served from disk, the API is loaded at
  start-up, so restart the server after updating) shows the reason while the simulation keeps working. Changing any setting clears the old
  predictions immediately, and responses for superseded settings are discarded.

`gravity_ai/controllers/learned.py` remains a placeholder that does not use the model.

## 6. Validate the integrated simulation

```bash
python -m scripts.run_demo
python -m pytest -q
python -m gravity_ai.api
```

Open http://127.0.0.1:8000/. Inspect commanded RPM, physical ring rotation, actual gravity, thrust, torque and head-to-foot gradient. Adjust the demonstration mission settings. Biological screen values remain explicitly synthetic.

## 7. Basic hyperparameters

- `batch_size`: number of training windows in each PyTorch optimizer step; 256 for the dynamics model (`configs/dynamics.yaml`), 128 for sequence models.
- `learning_rate`: gradient step magnitude; 0.0015 for the dynamics model and 0.001 for the sequence encoder.
- `epochs`: complete passes through the train subset; dynamics uses an upper limit (50 in the YAML) with early stopping on validation loss (`early_stopping_patience`, 8); 12 for optional sequence models.
- `optimizer`: AdamW for dynamics, Adam for sequence models.
- `loss`: mean squared error for standardized numerical output features; model evaluation is reported in physical units.
- `sequence_length`: 8 previous short windows for the optional RNN/Transformer.
- `split`: by entire participant or spacecraft episode rather than random neighboring rows.
- `device`: dynamics training accepts `auto` (CUDA if `torch.cuda.is_available()`, else CPU), `cpu` or `cuda`; requesting `cuda` without a CUDA-enabled PyTorch build is an error. The executed runs used CPU (CPU-only PyTorch build).
- `random_seed`: reproducible pseudo-random initialization and dataset splits.

Starter values are examples, not optimally tuned settings. Compare simpler baselines and validate on unseen participants or configurations.

## Troubleshooting

| Symptom | Response |
|---|---|
| DNS/name resolution failure | Use a networked machine to obtain raw public records, then transfer files respecting licenses |
| No `data/processed/wearable/e4_windows.csv` | Run downloader and prepare; check original E4 filenames/session folders |
| `No DXX*BMD` | Inspect NHANES XPT fields and download the correct DXA files; never invent a BMD label |
| Only 1–4 E4 subjects | Add genuine independent subjects; do not split one person's adjacent windows into train/test |
| Wrong E4 time or rate | File row 1 is UTC start, row 2 is Hz; do not treat as measurements |
| Package/import error | Activate virtual environment; run from repo root; `python -m pytest -q` |
| `npm install` fails | Use dependency-free viewer via Python API; React build needs npm registry |
| Model diverges | Check physical units, scalers fitted on train only, sample alignment, loss and learning rate |
| Unsupported clinical gravity adjustment | Expected; medically actionable treatment is intentionally disabled |
