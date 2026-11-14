# Personalized Artificial Gravity · Research Prototype

A **working computational research project** combining a torque-limited rotating-ring spacecraft simulator, NASA/EmbracePlus data-source catalogs, physiology data import/training code, a synthetic-data dynamics model trained with PyTorch, an API, and two interactive 3D viewers.

> **Not** a clinical recommendation tool or flight-ready spacecraft controller. Human gravity-dose response is **not validated** from the supplied catalogs. Physiological values in the browser demo are explicitly synthetic examples.

## What is included and what is not

| Capability | Actual status |
|---|---|
| Four user CSV catalogs | Present, parsed and validated (9 signals, 26 biomarkers, 11 source records, 16 mappings) |
| Companion Excel catalog | Preserved as an original reference file; 8 sheets |
| Previous research manifest/dictionary | Present: 40 source/file records and 102 variable mappings, plus research plan |
| Torque-driven ring/hub physics | Implemented and tested, includes finite torque, slew, power and reaction on hub |
| Full mission timeline and actual simulated ring angle | Implemented |
| Numerical PID and example MPC | Implemented; PID used by default |
| Synthetic dynamics dataset | v2: **60,000** transitions over 500 episodes, 4 time steps, frozen 70/15/15 episode splits (`data/synthetic/dynamics_v2/`); archived v1: 8,400 transitions over 70 episodes |
| PyTorch learned dynamics checkpoint | **Trained** on synthetic numerical physics (`outputs/runs/`); evaluated from disk against persistence/ridge baselines and multi-step rollouts. Shown in the viewer's "Learned Model vs Simulated Physics" panel (one-step and open-loop forecast). Prediction only: PID/MPC still select commands |
| Browser 3D viewer | **Usable without Node/npm** at `http://localhost:8000/` |
| React + Three.js source | Implemented; needs npm install in an internet-connected environment |
| Human raw wearables / NHANES DXA | **Not included**; requires source download; network blocked in this execution environment |
| Trained human physiological/BMD model | **Not trained**; training scripts only, intentionally no fabricated patients |
| Validated health-dependent gravity prescription | **Unavailable**, on purpose; research uncertainty displayed |
| Real ISS measurements (archived) | **Integrated** as a separate viewer mode and `/api/measured/*`: MARS-S centrifuge RPM replay (MHU-2, 2017) with a frozen one-step next-second RPM model, NASA PIMS ISS acceleration (1–11 Nov 2026) with a frozen one-step 16-s model, NASA ISS reference dimensions as a schematic. Not live, not a torque/controller/health model. See [MEASURED_ISS](docs/MEASURED_ISS.md) |

## Try it locally (minimal)

```bash
cd personalized-gravity-ai
python -m pip install -r requirements.txt
python -m gravity_ai.api
```

Open **http://127.0.0.1:8000/**. If port 8000 is already used by another application, run `python -m uvicorn gravity_ai.api.app:app --host 127.0.0.1 --port 8001` and open http://127.0.0.1:8001/ instead. The built-in dependency-free 3D perspective viewer uses the real numerical simulation API and includes a "Learned Model vs Simulated Physics" panel (predictions only; PID/MPC select the commands). Drag to orbit and scroll to zoom. It has mission playback, health labels, and physics readouts. Run `python -m scripts.run_demo` to regenerate the mission CSV/plot.

For the richer React/Three.js viewer, start the API as above, then in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173/**. `frontend/node_modules` was not bundled or tested here: the build host had no outbound DNS access. The self-contained fallback viewer *was* served and tested.

## Archived ISS measurements mode

Click **ISS archived measurements** in the viewer (or open `/#iss`). It replays recorded MARS-S
centrifuge RPM (mission MHU-2, 2017) with one-step next-second forecasts from fresh measured
history, shows a separate NASA PIMS acceleration panel (1–11 Nov 2026 UTC, SSAnalysis frame, µg),
and draws a schematic plan view proportional to NASA's published ISS dimensions with an enlarged
MARS-S inset. The hypothetical 100 m ring above is unchanged and remains the default mode.

```bash
python -m pytest -q tests/test_measured_iss.py   # integration tests
python -m scripts.verify_measured_iss            # hash, parity and raw-archive checks (JSON report)
python -m scripts.build_pims_replay              # rebuild the PIMS replay from reacquired NASA data
```

Details, endpoint contracts and limitations: [MEASURED_ISS](docs/MEASURED_ISS.md).

## Catalogs and research evidence

The actual **five user inputs** are stored unchanged in `data/catalogs/`:

```
embraceplus_watch_signals.csv
embraceplus_watch_biomarkers.csv
earth_human_dataset_sources.csv
earth_human_metric_mapping.csv
embraceplus_earth_human_data_catalog.xlsx
```

The **earlier conversation's** research artifacts are also preserved under `reference/`: `dataset_manifest.csv`, `data_dictionary.csv`, and `data_gaps_and_simulation_plan.md`.

Each original research dataset remains a separate population and study. The CSV catalogs are not raw human data and **must never** be passed directly as training rows. See [DATASETS](docs/DATASETS.md).

## Commands

```bash
python -m gravity_ai.data.catalog_loader --validate
python -m gravity_ai.simulation.generate                     # dynamics-v2 dataset (configs/dynamics.yaml)
python -m gravity_ai.train.dynamics --run-id <run_id>        # writes outputs/runs/<run_id>/
python -m gravity_ai.evaluate_dynamics --run outputs/runs/<run_id>
python -m gravity_ai.predict_dynamics                        # inference example, latest run
python -m scripts.run_demo
python -m gravity_ai.evaluate --all
python -m pytest -q
```

See [TRAINING](docs/TRAINING.md) section 5 for the executed dynamics commands, settings and results.

Human data ingestion/training after an Internet-enabled download:

```bash
python -m gravity_ai.data.download --from-catalog --public-only --dataset pilot
python -m gravity_ai.data.prepare --dataset all
python -m gravity_ai.train.physiology --config configs/physiology.yaml
python -m gravity_ai.train.sequence --model gru  # optional, requires multiple real participants
python -m gravity_ai.train.bone --config configs/bone.yaml
```

See [INSTALLATION](docs/INSTALLATION.md), [TRAINING](docs/TRAINING.md), [PHYSICS](docs/PHYSICS.md), [VALIDATION](docs/VALIDATION.md), and [LIMITATIONS](docs/LIMITATIONS.md).

## Source citations

- PhysioNet, *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (version 1.0.1, 2025): https://physionet.org/content/wearable-device-dataset/1.0.1/ (DOI 10.13026/he0v-tf17).
- CDC NHANES 2013–14: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2013-2014
- NASA OSDR: https://science.nasa.gov/biological-physical/data/osdr/
- NASA PIMS: https://gipoc.grc.nasa.gov/wp/pims/acceleration-archives/
- AGBRESA centrifuge tolerability supplements: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0239228
- Uppsala Coupled Electric Drives benchmark: https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives
- Basilisk spinning-body model: https://avslab.github.io/basilisk/Documentation/simulation/dynamics/spinningBodies/spinningBodiesOneDOF/spinningBodyOneDOFStateEffector.html

Respect the individual dataset's license, access terms, and required citations before importing raw data.
