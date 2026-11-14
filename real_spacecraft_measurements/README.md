# Real ISS measurements for the artificial-gravity project

**Two measured-data models have been trained and tested.** The main experiment forecasts the next second of MARS-S centrifuge RPM from real ISS flight records. A separate model forecasts the ISS's filtered three-axis acceleration measurements. This package also supplies NASA's published ISS dimensions for a station visualization.

The current 100 m crew-ring simulator has a different physical system and input/output contract. These results support specific measured-data components; they do not establish a validated full-spacecraft motor/controller model. This package has not been installed into the Windows project.

## Hardware and evidence

The real host spacecraft is the **International Space Station**. The rotating apparatus is JAXA's **MARS-S** centrifuge in the Kibo CBEF facility. The [official mission overview](https://ibsls.megabank.tohoku.ac.jp/overview) gives a **0.15 m nominal radius and 77 RPM nominal setting** for MHU-1 and MHU-2. The records used here were measured during **2016 and 2017 flights** and released with the [2026 ibSLS paper](https://doi.org/10.1038/s41467-026-76349-y), Supplementary Data 12 and 13.

The [JAXA catalogue updated May 27, 2026](https://humans-in-space.jaxa.jp/kibouser/subject/tag_10.html) lists later MHU missions in analysis and preparation. It supports a continuing implemented program, not a claim that the specific 2016 rotor is rotating today. The small animal centrifuge is not a human-scale rotating station.

`hardware_profiles.json` records exact source URLs, definitions and caveats. NASA's [ISS Facts and Figures](https://www.nasa.gov/international-space-station/space-station-facts-and-figures/) supplies 109 m solar-array span, 73 m longitudinal array extent, 94 m truss length, 67 m pressurized-module major-axis length and 419,725 kg reference mass. These are reference specifications with specific boundaries, not time-aligned mass/inertia observations. The [NASA reference page](https://www.nasa.gov/reference/international-space-station/) explains that visiting vehicles change the station's configuration and mass.

## 1. Completed centrifuge training

### Data and split

The two used archives contain **5,539,624 raw timestamp/RPM rows**. Duplicate recorded timestamps are resolved by keeping the last row in original source order. No timestamps are relabeled UTC because the raw files do not specify a timezone. No missing observations are interpolated; genuine stopped-hardware zero RPM is retained. A supervised window requires eight measured history values and the following label at consecutive one-second timestamps.

| Partition | Source | Eligible windows | Actually used for fitting/scoring |
|---|---|---:|---:|
| Training | First 70% of MHU-1 elapsed time | 1,970,009 | 150,933 for fitting |
| Validation | Final 30% of MHU-1, with complete history inside split | 846,611 | All 846,611 |
| Test | Independent MHU-2 mission | 2,428,381 | All 2,428,381 |

The bounded fitting set combines 150,000 uniformly spaced training windows with every training history whose RPM range exceeds 2 RPM. The union contains 150,933 windows. Selection uses historical observations only, never the next label. The training/validation boundary is `2016-08-15T00:19:29` as recorded. Overlapping observations are strongly dependent; millions of rows are not millions of independent flight experiments.

### Model and measured result

Input is **eight measured RPM samples**, seven seconds from first to last. Features are latest RPM plus seven chronological differences. Target is the residual change to the next measured RPM, **one second ahead**. The nominal radius is fixed source metadata; it is not a learned input.

The candidate budget was persistence, eight-sample moving mean, two Ridge regularization values, and one MLP with two 32-unit ReLU layers. Scalers were fitted on training data only. Full chronological validation RMSE selected the MLP checkpoint at **epoch 38 of 40**. We saved the selection and weights before loading/scoring the test mission in the training pipeline.

| Independent MHU-2 one-step result | RMSE, RPM | MAE, RPM |
|---|---:|---:|
| Persistence: repeat latest reading | 0.474178 | 0.323879 |
| Eight-sample moving mean | 0.374727 | 0.269249 |
| Ridge, alpha 1 | 0.311464 | 0.219713 |
| **Validation-selected MLP** | **0.245751** | **0.155374** |

MLP RMSE is **48.17% lower than persistence** on this mission. On the 2,130 windows with changing past histories, RMSE is **2.189355 RPM**, compared with **2.795180 RPM** for persistence. The larger transition error matters: the numerous steady-state rows otherwise dominate the aggregate result.

### Observed limitations: one-step use only

The model forecasts already controlled rotor behavior from RPM history. The data does not contain motor commands, torque, current, inertia, hub motion, thrust or maintenance scheduling. It cannot know an unannounced future start/stop command or identify response to arbitrary commanded torque.

**The MLP failed autonomous recursive forecasting.** On 4,074 regularly spaced origins, its 60-second RMSE was 5.879422 RPM versus persistence 1.387750 RPM. On 64 origins with changing past histories, some predictions diverged severely: aggregate 60-second error reached approximately 5.74e15 RPM. These are numerical model failures, not physical measurements. Full failed results and plots are retained. There was no test-driven model replacement or tuning.

**Negative predictions also occur:** 66,222 one-step predictions, about 2.73% of the test windows, were negative in this nonnegative-speed operating regime. Some exceed a trivial rounding error. The raw regression outputs remain in evaluation. A viewer must mark negative predictions invalid and suppress derived gravity for them. The CLI does this and defaults to a single step; longer diagnostic recursion requires an explicit flag.

Treat the checkpoint as an exploratory measured-history predictor. It must not drive a motor or replace the physics simulator. New architecture/stability work needs a new evaluation design because the MHU-2 results have now been examined.

### Gravity calculation

For nonnegative measured or forecast speed, the nominal radial component is calculated as:

`radial_g = 0.15 * (2*pi*RPM/60)**2 / 9.80665`

This is a physics calculation at the published nominal radius, not an independent gravity measurement, a learned physiology target or whole-body exposure. The formula alone does not validate a 100 m crew habitat.

## 2. Completed NASA ISS acceleration training

The [NASA PIMS archive](https://gipoc.grc.nasa.gov/wp/pims/acceleration-archives/) provides genuine MAMS engineering measurements. This experiment uses **59,062 valid, filtered three-axis observations from November 1–11, 2026 UTC**. The OSS BTMF product is already bias compensated and trimmed-mean filtered. We kept the declared **16-second** sampling cadence and `SSAnalysis` coordinates.

The acquisition script follows NASA's PAD timing instructions, ignores the inconsistent embedded relative-time field, and preserves four real gaps as five segments. No extra observations were created. Full source/header hashes and parsing decisions are in the source manifest.

Training used November 1–7 (37,475 windows); validation November 8–9 (10,739); test November 10–11 (10,792). Each input is **eight three-axis samples**, spanning **112 seconds**, and predicts the next sample **16 seconds** later. Windows never cross split or gap boundaries. Training-only scaling and chronological validation selected a 32/16-unit tanh residual MLP at epoch 49.

| Held-out one-step result | RMSE, micro-g | MAE, micro-g |
|---|---:|---:|
| Persistence | 0.225622 | 0.152938 |
| Ridge | 0.170794 | **0.117635** |
| Selected MLP | **0.170533** | 0.119118 |

The MLP's RMSE is **24.42% lower than persistence**. Its advantage over Ridge is tiny, and Ridge has lower MAE. This is not evidence of meaningful neural superiority over the simpler learned model.

For recursive forecasts from 10,778 common origins, selected-MLP RMSE at 16/80/240 seconds is 0.170624/0.229110/0.231565 micro-g, versus 0.225755/0.302202/0.301897 for persistence. These values describe the already filtered series at the measurement location, not all spacecraft vibration frequencies.

The PIMS records are separate in date, location and variables from the MARS records. They must not be joined as if simultaneous, co-located measurements or used to invent torque labels.

## Run the trained models

Both models are already trained; inference requires only NumPy. Extract the ZIP, open this directory in a terminal with a Python environment that has NumPy, and run:

```powershell
python rpm_predict.py
python outputs/pims/inference.py --model outputs/pims/selected_model.json --history outputs/pims/example_history_g.json --steps 1
```

If NumPy is missing, install `requirements-inference.txt` into a project environment. Neither inference path needs PyTorch, a GPU, an external API or access to a live spacecraft.

The RPM CLI takes a JSON file with eight chronological RPM values via `--history`. Its default sample is a real MHU-2 startup history. The PIMS CLI takes an 8-by-3 history in units of standard gravity. In a live or replay integration, the caller must validate the accompanying timestamps and frame; eight numbers alone cannot prove the cadence is correct.

## Reproduce training

The source archives and parsed PIMS CSV are included, so retraining does not require downloading again. The recorded environment is **Python 3.12.14**, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0, with the remaining versions in `requirements-training.txt`. Use an isolated Python 3.12 environment for reproduction rather than changing the existing app's dependencies.

```powershell
py -3.12 -m venv .venv-training
.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
.venv-training\Scripts\python.exe rpm_train.py
.venv-training\Scripts\python.exe pims_train.py
.venv-training\Scripts\python.exe export_rpm_replay.py
```

These commands assume Python 3.12 is installed and discoverable by the Windows Python launcher. Equivalent `python3.12` virtual-environment commands work on Linux/macOS. CPU training was sufficient. Exact timings and tiny floating-point differences can vary by machine.

Retraining writes the files in `outputs/`; preserve the bundled original results if comparing new work. `pims_acquire.py --help` describes reacquisition options. The raw MHU publisher URLs and checksums are in `data/raw/mhu/download_manifest.json`.

## Integrate into the existing VS Code project

Use `CLAUDE_INTEGRATION_PROMPT.txt`. It directs Claude to inspect the current repository, add an archived measured-data mode, use the hardware profile, and check inference contracts and units. `outputs/rpm/replay_mhu2.json` supplies short actual measurement clips with one-step predictions for an initial viewer integration.

The RPM model has **8 inputs / 1 output** and the PIMS model **24 inputs / 3 outputs**. Neither matches the project's existing **13-input / 5-output** ring dynamics checkpoint. Keep the existing selected dynamics checkpoint and hypothetical ring mode intact. Use separate data/model identifiers and source labels. Existing gravity-server port 8001 and the separate app on port 8000 should be handled independently.

## Package contents and verification

- `hardware_profiles.json`: source-backed ISS and MARS configuration metadata.
- `rpm_train.py`, `rpm_predict.py`: reproducible RPM training and portable inference.
- `pims_acquire.py`, `pims_train.py`: NASA data acquisition/parsing and training.
- `outputs/rpm/`: frozen selected weights, usage notes, all candidate weights, full metrics, training history, plots and diagnostic/replay data.
- `outputs/pims/`: weights, portable inference, full held-out predictions, metrics and plots.
- `data/raw/`: original MHU archives/PDFs plus 134 NASA binary/header pairs and provenance manifests. MHU-4 and MHU-5 were audited but not used for this training; their different nominal settings and a radius-source discrepancy are documented.
- `data/processed/`: measured PIMS table and dated split manifest.
- `SOURCES_AND_ATTRIBUTION.txt`: primary sources and original-material attribution.
- `FILE_MANIFEST.json`: package file sizes and SHA256 values.

Source hashes, strict cadence, split boundaries, train-only normalization and saved selection before test were checked. Portable RPM MLP inference matches scikit-learn within 1.43e-14 RPM on 256 held-out histories. Portable PIMS MLP inference matches exactly on its 256 checked histories. Both command-line inference paths were exercised. The failed RPM recursion is a retained result, not a hidden test or a success claim.
