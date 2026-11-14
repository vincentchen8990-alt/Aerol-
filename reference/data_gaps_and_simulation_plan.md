# Data audit and simulation plan for an artificial-gravity controller

**Checked:** 8 November 2026  
**Reference design:** rotating habitat ring coupled to a central hub; 100 m crew-floor radius; target apparent gravity 9.80665 m/s² at a specified reference point.  
**Research stage:** computational modelling and dataset selection.

## 1. Main finding

The search found useful measured motor experiments, real spacecraft reaction-wheel telemetry, ISS acceleration recordings, human centrifuge outcomes, and OSDR organism-response datasets. It did **not** identify a verified, complete measured dataset containing a rotating inhabited spacecraft's observations, requested motor action, delivered torque, hub reaction, thrust history, and resulting crew-floor apparent acceleration.

This is a bounded search result, not proof that no proprietary, unindexed or request-only dataset exists. The evidence is sufficient to start a numerical system-identification experiment and to design a physically meaningful simulator. It is not a ready-made training corpus for an operational 1G habitat controller.

The strongest distinction is between **a torque command** and **a measured or estimated consequence**. Rosetta's friction-torque product is an estimate of friction, not a motor command. The OSDR fly experiment's 68 rpm is an exposure setting, not a sampled command-to-response history. Neither can be relabelled as supervised action/outcome data.

### Deliverables

| File | Contents |
|---|---|
| `dataset_manifest.csv` | 40 file/source records, exact identifiers and source links, verification scope, access/reuse status, file sizes and SHA-256 where available, counts, preprocessing and limitations. Rows include metadata and documentation; they are not 40 independent experiments. |
| `data_dictionary.csv` | 102 mappings covering required variables, exact observed fields, surrogate measurements, protocol-only parameters, and explicit gaps. |
| `data_gaps_and_simulation_plan.md` | This report: interpretation, preprocessing, physical model, missing-data generation and experiment design. |
| `research_samples.zip` | Selected original numerical files, headers, OSDR metadata and audit notes. Data have not been silently corrected, rescaled or joined. |

Source timestamps, file versions and the check date are retained because an archive can change. A file checksum identifies the inspected bytes even when a repository does not assign an individual file version.

## 2. Ranked sources

Priority is based on usefulness to the proposed computation, with separate roles for engineering and human evidence.

| Priority | Source | Verified scope | Contribution | Principal limitation |
|---|---|---|---|---|
| 1 | [Coupled Electric Drives Data Set and Reference Models](https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives), `diva2:1165531` | Complete source ZIP downloaded; two CSVs and technical report inspected | Five recorded input/output experiments for testing dynamics learning | Ground apparatus; no torque/current; direction-insensitive output and unverified physical speed scale |
| 1 | [Basilisk spinningBodyOneDOFStateEffector](https://avslab.github.io/basilisk/Documentation/simulation/dynamics/spinningBodies/spinningBodiesOneDOF/spinningBodyOneDOFStateEffector.html), documentation 2.12.2 | Official documentation and ISC license inspected online | Coupled hub/spinner dynamics with motor torque input | Software component; no measured habitat trajectories or generated dataset supplied |
| 2 | [OpenMCT DC motor workflow](https://data.mendeley.com/datasets/5xvg43r9r8/2), DOI `10.17632/5xvg43r9r8.2` | Repository/article metadata only | Promising PWM/current/RPM and controller-validation recordings | Raw files could not be inspected through the available public extraction; counts remain repository claims |
| 3 | [Rosetta spacecraft reaction-wheel engineering data](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=RO-X-HK-3-RWL-V1.0), `RO-X-HK-3-RWL-V1.0` | Entire index plus two complete representative products and labels downloaded | Real flight actuator telemetry, gaps and friction-estimate behaviour | No commanded actuation in inspected pair; slow, irregular sampling |
| 4 | [AGBRESA tolerability supplements](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0239228), `10.1371/journal.pone.0239228.s001`–`.s004` | All four files downloaded and inspected | Human response and tolerance context | Sparse outcomes from supine ground centrifugation, without control trajectories |
| 5 | [OSD-595](https://osdr.nasa.gov/bio/repo/data/studies/OSD-595), version 1; [OSD-596](https://osdr.nasa.gov/bio/repo/data/studies/OSD-596), version 2 | Two transformed CSVs, two workbooks, two ISA ZIPs; protocols/metadata inspected | Actual biological responses to centrifugal gravity exposure | Fruit-fly behaviour, not crew tolerance or motor control |
| 6 | [NASA PIMS acceleration archive](https://gipoc.grc.nasa.gov/wp/pims/acceleration-archives/) | One MAMS and one SAMS binary segment, both headers and timing primer downloaded | ISS disturbance ingestion and robustness-test examples | ISS environment is not the future ring's disturbance or sensor-noise calibration |
| 7–9 | NASA NTRS, NLSP, DLR, ESA, OSDR EDA/SPLASH, OSD-686 | References, catalog records or metadata as individually labelled | Physics, human-data discovery, experiment/hardware context | Not additional verified actuator-to-gravity datasets |

The six verification labels are used as requested. “Downloaded and inspected” always has a scope: it can mean a complete small table, one representative product, or an archive inventory. It does not mean an entire mission archive was inspected. OSD-686 remains “Metadata only” even though its API metadata was downloaded; its biological assay values were not inspected.

## 3. What is actually in the strongest files

### 3.1 Measured numerical input/output: Coupled Electric Drives

The [original university ZIP](https://uu.diva-portal.org/smash/get/diva2%3A1165531/FULLTEXT01.zip) is 277,015 bytes. Its two CSV members are included as numerical samples; the source report and original archive remain linked in the manifest.

| Member | Actual column names | Shape | Experimental meaning |
|---|---|---|---|
| `DATAPRBS.csv` | `z1,u1,z2,u2,z3,u3,` | 500 numeric rows; final empty header field | Three experiments: `u1→z1`, `u2→z2`, `u3→z3` |
| `DATAUNIF.csv` | `u11,u12,z11,z12,` | 500 numeric rows; final empty header field | Two experiments: `u11→z11`, `u12→z12` |

There are five experiments and 2,500 input/output pairs in total. The technical report specifies a 20 ms sampling interval, or 50 Hz. The nominal run duration is 10 s; a zero-based 500-sample timeline runs from 0 through 9.98 s. That timeline is derived from the reported sampling interval because no timestamp column is present.

The recorded inputs represent summed motor voltages in V; independent terminal-voltage metrology was not established. Outputs are measured pulley-speed signals from a direction-insensitive pulse counter. Small negative observations do occur in the files, so do not take absolute values or clamp them. The physical output scale was not established in the inspected files/report, so the exported dictionary deliberately uses **dataset units**, not rpm or rad/s. This is useful for validating a sequence-model training pipeline, but its parameters cannot calibrate a 100 m habitat. The benchmark page requests a citation; an explicit data license was not verified. Do not apply a third-party loader's software license to the dataset.

### 3.2 Real spacecraft telemetry: Rosetta

The [PDS catalog](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=RO-X-HK-3-RWL-V1.0) identifies four reaction wheels and version 1.0. The inspected archive index contains 820 product entries: 164 each for speed, measured angular momentum, estimated friction torque, friction coefficient, and wheel direction. These are product counts, not independent experiments. The full archive is advertised at about 1.6 GB and was not downloaded.

The representative files are:

| Filename | Label meaning | Actual rows | File size |
|---|---|---:|---:|
| `ros_hk_nacg0014_2004_q1.tab` | RW A Real Wheel Speed; rpm | 129,749 | 5,968,454 bytes |
| `ros_hk_nacw0g05_2004_q1.tab` | RW A Est Friction Torque; Nm | 132,525 | 6,096,150 bytes |

Both labels specify `TIME OF MEASUREMENT` and `VALUE`. Sampling commonly uses 16 s and 64 s intervals and includes long gaps. The timestamps differ across channels, so a row-number join is invalid. A time interpolation would create a derived association, not proof that friction estimates were independent measurements or causal control inputs. Wheel direction encoding has outstanding review notes; signed inertial habitat rotation must not be inferred from these files.

The [NASA catalog](https://data.nasa.gov/dataset/rosetta-reaction-wheel-engineering-data) marks public access and `us-pd`; preserve the ESA/Rosetta/PDS attribution and source record. The archive's review-status notes are retained. These data are best suited to telemetry handling and actuator-health research.

### 3.3 A promising further acquisition: OpenMCT motor logs

The [Mendeley record](https://data.mendeley.com/datasets/5xvg43r9r8/2) reports version 2, published 6 September 2026, under CC BY 4.0. It describes 14 raw logs with more than 60,000 sample rows from a terrestrial DC motor apparatus. Reported fields are `REF`, `MEAS`, `DT_ms`, `CURRENT_RAW`, `CURRENT_AVG`, `PWM`, `DMM_CURRENT_A`, `DMM_TIME_s`, `DMM_AGE_ms`, and `DMM_SAMPLE_ID`.

These are attractive because they reportedly connect actuation, speed and current. The raw files were not retrieved, so the inventory does not claim to verify their values, units, synchronization or row counts. A documented follow-up file is `04_continuous_PID_validation/20_ms/raw_data_20ms.txt`. Obtain it with `DATA_DESCRIPTION.md` and `DATASET_METADATA.md` before using it. The multimeter stream may repeat readings and have asynchronous timing. This educational OpenMCT kit is unrelated to NASA's similarly named Open MCT visualization software.

### 3.4 Human evidence: the four AGBRESA supplements

All four source files returned HTTP 200 and total 32,951 bytes. They are included unchanged with hashes. The [publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0239228) and its CC BY 4.0 statement provide study context and attribution.

| File | Inspection result | Use |
|---|---|---|
| `pone.0239228.s001.xlsx` | Sheet `AGBR_Quest_ results`; 65 rows, 34 columns, 24 coded participants | Questionnaire outcomes including ESS, MSAQ and PANAS; three numerical groups with eight subjects each |
| `pone.0239228.s002.xlsx` | Sheet `Medical_Data`; 48 rows, 21 columns, 16 participants × days 1, 30, 60 | Pre/post and early-run HR/BP summaries |
| `pone.0239228.s003.csv` | Semicolon delimiter, UTF-8 BOM; 60 rows × 16 active score columns; 937 observed values, 23 blanks | Daily motion-sickness score table |
| `pone.0239228.s004.csv` | Tab delimiter despite `.csv`; 60 rows × 16 score columns; 960 numeric values | Daily perceived-exertion table |

Counts were computed from the files. S3/S4 have repeated group headers but no explicit participant IDs, day field or timestamps. Their rows are compatible with the study's daily schedule; assigning day numbers or linking columns to S1/S2 subjects is not independently documented. Even matching column positions between S3 and S4 are not a verified participant crosswalk. Preserve separate positional series identifiers for each file. S1/S2 numerical treatment, sex and age-group codes do not come with a complete in-file decoding dictionary. S2's early-run fields refer to the first during-centrifugation measurement scheduled two minutes after the +1 Gz plateau; the exact aggregation window is not encoded in the file.

Quality issues requiring flags, not silent repair:

- S1 has an extra nominal examination for one participant, partial date strings and an entirely blank variable column. A stored delta can be zero despite missing source values.
- S2 contains `SBP_peri_2min=14` paired with diastolic 67. The source really contains 14; do not replace it with a guessed value.
- S3's 23 blanks cannot be equated to the publication's terminated sessions.
- S4 contains one score of 5 while the paper describes a 6–20 exertion scale.
- Conventional HR/BP units are not explicitly encoded in the supplied sheets; verify the codebook before treating the file as calibrated physiological training data.

The experiment describes supine ground centrifugation, a center-of-mass gravity target, limited head/body movement, and repeated daily exposures. It does not establish comfortable long-duration walking at a particular spacecraft rpm. Continuous ECG collection described in the methods does not mean continuous waveforms are present in S2. The source-specific spin-ramp descriptions in the 2020 study and [2022 protocol](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.976926/full) differ; keep them distinct instead of inventing one universal limit.

### 3.5 OSDR: verified gravity-exposure data, with explicit biological scope

NASA's [OSDR overview](https://science.nasa.gov/biological-physical/data/osdr/) links its repository, documented BIOdata API, environmental application and hardware catalog. The study inventories were retrieved through the documented dataset/file endpoints, then original downloads followed. OSDR's [terms](https://science.nasa.gov/reference/osdr-help-terms-and-conditions/) describe open access, the SMD information policy and dataset citation expectations; no blanket Creative Commons license is invented here.

**OSD-595**, version 1, DOI `10.26030/vsya-2v22`: *Artificial gravity partially protects space-induced neurological deficits in Drosophila melanogaster*. The transformed file `LSDS-42_Video Recording_LSDS-42_video_mhatreTRANSFORMED.csv` has 11 rows and 99 columns. Three columns identify module and condition; 96 describe video-derived intensity sums over mission days 13–28. There are six 1G modules and five microgravity-comparator modules. These are 1,056 repeated module/video summaries, not 1,056 independent organisms or experiments. The protocol gives centrifuge settings of 68 rpm and 2.2 rpm, respectively; the latter comparator is not exactly zero rotation. Motor actions, sampled RPM response and delivered torque are absent from the inspected results.

**OSD-596**, version 2, DOI `10.26030/eybf-jv71`: the companion behaviour/climbing study. `LSDS-43_Locomotion_LSDS-43_Locomotion_mhatreTRANSFORMED.csv` has 83 rows with `Source Name`, `Spaceflight`, `Altered Gravity`, `sex`, and `percent_climbed_to_5_centimeter`. The protocol describes vial groups and averaging three climbing trials. The records comprise 27 ground-control, 29 flight-1G and 27 flight-microgravity entries. Preserve the shared exposure and vial-level experimental structure.

Original workbooks and ISA metadata archives are included for both studies. They are alternate representations and metadata, not extra independent samples.

**OSD-686**, version 8, DOI `10.26030/vf2f-v411`: *Impact of spaceflight and artificial gravity on sulfur metabolism in mouse liver: sulfur metabolomic and transcriptomic analysis*. Nine RNA-seq sample entries and 178 file entries were observed in metadata/inventory. No assay values were downloaded. The title does not justify labelling every listed file metabolomics; the inspected OSDR assay is transcription profiling. This is a separate biological follow-up, not a control-data acquisition priority.

OSDR's [EDA](https://visualization.osdr.nasa.gov/eda/) provides cabin/hardware environmental variables such as temperature, CO2 and humidity, plus radiation context. [SPLASH](https://visualization.osdr.nasa.gov/splash/) catalogs experiment hardware. Their availability does not establish that they contain the ring's required torque/RPM histories. An additional [mouse dose-response experiment record, OS-840](https://osdr.nasa.gov/bio/repo/data/experiments/OS-840), stated results were in progress with no associated studies at inspection.

### 3.6 NASA PIMS: inspect the header, not just the numbers

Two complete segments from 27 October 2026 were decoded as little-endian float32 records with four columns. The [official guide](https://gipoc.grc.nasa.gov/pims/plots/user/mohan/how2read_pad_binary_files.html) identifies time in seconds and x/y/z acceleration in g. The files' data frame is `SSAnalysis`.

| Segment | Samples | Rate | Cutoff | Interpretation |
|---|---:|---:|---:|---|
| MAMS `ossbtmf`, approximately two hours | 450 | 0.0625 Hz | 0.01 Hz | Quasi-steady, trimmed-mean-filtered and bias-compensated product |
| SAMS `121f02`, approximately 17 seconds | 8,500 | 500 Hz | 200 Hz | Vibratory acceleration; archive instructs demeaning to remove bias |

Follow the included [PAD timing primer](https://gipoc.grc.nasa.gov/pims/pub/plots/user/mohan/PADprimer.pdf): reconstruct timing using the filename start and header rate rather than the stored time column or filename end. The observed MAMS last relative time differs from the prescribed grid by about 2.981 s. Do not infer a missing-sample mechanism from that discrepancy alone. The archive's contiguous/gap markers matter when joining files.

Both headers contain an apparently stale `ISSConfiguration` entry; it must not be used as verified 2026 event attribution. SAMS has unverified `BiasCoeff` values 1.23/4.46/7.89; do not subtract them as if their units and calibration meaning were established. Coordinate metadata exists, but position/Euler units were not verified from the sampled XML. Preserve the header and avoid an invented geometric transform. Public download access was verified; a specific reuse/ML license was not established in the reviewed pages.

The segments were acquired at different times and are not synchronized measurements of the same event. They support data ingestion and optional disturbance stress tests. They neither measure the future habitat's structural response nor separate instrument noise from physical ISS vibration.

## 4. Variable coverage and valid joins

The CSV dictionary is the detailed mapping. The main coverage is:

| Required variable family | Verified source/field | Remaining gap |
|---|---|---|
| Input/action to later speed | CED `u1→z1`, `u2→z2`, `u3→z3`, `u11→z11`, `u12→z12` | No ring torque, calibrated signed ring rate, hub state or gravity outcome |
| Spacecraft wheel speed | Rosetta `VALUE` under the real-wheel-speed label | No verified absolute inertial habitat rate |
| Friction | Rosetta estimated-friction `VALUE` in Nm | No habitat bearing law, applied motor torque or thermal dependence |
| Acceleration and sensor metadata | PIMS binary columns 2–4 and XML `SampleRate`, `DataCoordinateSystem`, `DataQualityMeasure` | No ring-specific frame/transfer model or command history |
| Human symptoms and cardiovascular outcomes | AGBRESA questionnaire, HR/BP and positional daily scores | No action-conditioned control labels; sparse timing and codebook gaps |
| Biological exposure/outcome | OSDR condition fields, protocol RPM and climbing/video outcomes | No astronaut-specific limits or motor dynamics |
| Ring/hub geometry, inertia, mass and center of mass | None for this design | Obtain CAD/bench parameters or declare assumptions |
| Commanded versus delivered torque, delays, current, voltage, temperature | No complete inspected target-system sequence | Bench data and simulator actuator model required; OpenMCT is a further candidate |
| Thrust vector, floor orientation, crew position, local gravity, jerk, saturation and faults | None as one synchronized measured sequence | Generate and log in the coupled simulator |

Valid integration keys are local to an experiment: CED experiment-pair plus row index; Rosetta wheel and timestamp; PIMS sensor and reconstructed time; OSDR sample/module IDs plus assay labels; documented AGBRESA participant/examination identifiers. Repeated human score columns lack a validated identity mapping. A shared numerical timestamp does not make different missions, species or hardware one coherent trajectory.

Use these sources as separate evidence streams. Only join them when physical identity, coordinate frames, clocks and sampling semantics are established. A biological label must never be copied onto an unrelated motor trace as a control reward.

## 5. Physical model needed before generating training data

### 5.1 Define the controlled quantity

Use apparent acceleration at a named point on the crew floor. Record its three components in the local floor frame, the normal component and lateral components, not just its magnitude. The target 1G applies at that reference location; a rotating habitat has a radial gradient.

The [NASA artificial-gravity chapter](https://ntrs.nasa.gov/api/citations/20070001008/downloads/20070001008.pdf) is a physics reference. The model below specifies how to use the rotating-frame terms in this project's dataset.

For a frame origin with inertial acceleration **A_O**, absolute angular velocity **Ω**, crew location **r**, and relative crew velocity **v_rel**, express all vectors in one frame:

\[
\mathbf g_{app}=\mathbf g_{true}(\mathbf r)-\mathbf A_O
-\dot{\boldsymbol\Omega}\times\mathbf r
-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf r)
-2\boldsymbol\Omega\times\mathbf v_{rel}.
\]

For a stationary occupant, contact force per unit mass balances this vector with opposite sign. For a moving occupant, the relative acceleration and contact dynamics must also be included; the Coriolis-containing vector alone is not a complete walking model.

If the spacecraft origin is falling under the same planetary field, the shared gravitational acceleration cancels in `g_true - A_O`; only the local field difference and non-gravitational acceleration remain. Do not add Earth's orbital gravity directly as an extra floor load.

### 5.2 The original RPM relation is an equilibrium condition

With a fixed spin axis, stationary reference occupant, negligible tidal terms and signed thrust acceleration `a_z` parallel to the axis:

\[
\mathbf g_{app}=R\omega^2\mathbf e_r-R\dot\omega\mathbf e_\phi-a_z\mathbf e_z,
\qquad
\|\mathbf g_{app}\|^2=R^2\omega^4+R^2\dot\omega^2+a_z^2.
\]

When spin is steady, the target is

\[
\omega_{eq}=\left(\frac{g_*^2-a_z^2}{R^2}\right)^{1/4},\qquad
\mathrm{RPM}=\frac{60\omega_{eq}}{2\pi},\qquad |a_z|\le g_*.
\]

For the reference 100 m radius and `g*=9.80665 m/s²`, calculated equilibria are:

| Signed axial thrust magnitude | Required steady spin |
|---|---:|
| 0G | 2.9904 rpm |
| 0.3G | 2.9207 rpm |
| 0.5G | 2.7829 rpm |
| 0.8G | 2.3164 rpm |
| 1.0G | 0 rpm |

Reversing the axial thrust sign reverses the axial apparent-down component and floor-tilt requirement, while the equilibrium speed magnitude stays the same. Spin direction need not reverse.

The Euler term `-R*dot(omega)` is present during changes in spin. Instantaneously moving from zero thrust to 0.8G while the ring is still at its coasting rate produces approximately `sqrt(1+0.8²)=1.2806G`, even before considering a spin-change contribution. A real actuator cannot instantly remove angular momentum. Exact constant 1G therefore requires feasible coordinated transitions; it cannot be promised for every arbitrary thrust command.

Within this geometry, axial thrust exceeding 1G makes a 1G total magnitude infeasible. At exactly 1G axial thrust, exact magnitude tracking requires both zero spin and zero spin acceleration. The equilibrium reference becomes increasingly sensitive near this limit. Use a reference governor and explicit infeasibility flags rather than silently clamping the square root.

A floor normal aligned only in the radial/axial plane cannot remove tangential Euler loading. If motorized decks are modelled, log their orientation, rate and actuator limits. For stationary shear-free support the upward floor normal must oppose `g_app`. Merely achieving the right vector magnitude is insufficient.

### 5.3 Model the hub reaction and real actuator

For constant coaxial inertias, with positive internal motor torque applied to the ring and opposing friction `tau_f`:

\[
I_r\dot\omega_r=\tau_m-\tau_f+\tau_{ext,r},\qquad
I_h\dot\omega_h=-\tau_m+\tau_f+\tau_{ext,h}.
\]

The encoder measures relative rate. In the coaxial case,

\[
\omega_{ring,abs}=\omega_{hub,abs}+\dot\theta_{ring/hub}.
\]

Gravity uses the absolute ring angular velocity. In 3D, compose vectors in a common frame rather than adding arbitrary scalar channels. If inertia changes during a run, include the angular-momentum transport; do not simply replace an inertia parameter while leaving momentum unexplained.

Basilisk's OneDOF effector provides a suitable rigid-body starting component. Use `motorTorqueInMsg.motorTorque[0]`, its coupled hub and its angle/rate outputs. Avoid prescribed perfect motion as a substitute for actual torque response. Pin the software version/commit, integration settings and configuration in every generated dataset.

Add finite torque, torque slew and speed limits, command delay, actuator lag, bearing friction, external torque, sensor uncertainty, and any required structural modes. Electrical power requires a motor/electrical model or calibrated current and voltage; do not call `torque * relative angular speed` total bus power without accounting for efficiency and losses.

## 6. The missing dataset to generate

### 6.1 Three synchronized logs and one configuration record

| Record | Required fields |
|---|---|
| Configuration | `configuration_id`, radius/geometry, masses, inertia tensors, center of mass, motor limits, friction model, sensor locations/orientations, delays, numerical step, simulator version and random seeds |
| Truth | `episode_id`, `t_s`, true hub and ring states, relative encoder angle/rate, delivered thrust, applied torque, floor/crew geometry, apparent-gravity vector, contact-load assumptions |
| Observations | Measurement and arrival timestamps, noisy encoder/IMU/current observations, estimated state, bias/noise definitions, validity/dropout flags, measurement age |
| Actions and outcomes | Requested torque, delivered torque after limits/delay, requested/actual floor angle, setpoints, saturation, power, gravity error, lateral loads, mode changes, fault and feasibility flags |

Preserve the simulator truth for supervision and evaluation without making it an input that a real controller would not have. A future thrust schedule may be supplied only when the mission actually provides that preview. Never leak future fault times or future sensor readings to the controller.

The learned-dynamics problem is a sequence of histories and actions predicting subsequent state/observation distributions. A learned controller is evaluated by closed-loop tracking and constraint outcomes. An LLM can retrieve the physics, explain decisions or propose bounded supervisory settings. These are different training tasks; a literature collection is not a replacement for numeric action-conditioned trajectories.

### 6.2 Scenario coverage

Start with a coupled coaxial system before claiming a general 3D controller. Cover coasting, initial spin-up, scheduled axial thrust ramps, braking, return to coast, thrust sign reversal, actuator saturation, delay, sensor bias/dropout, and an explicitly infeasible request. Include repeated excitation at several amplitudes and rates so the learner can distinguish inertia, damping and actuator delay.

Then add off-axis thrust, hub attitude changes, precession, flexible modes, changing payload distribution with momentum accounting, movable decks and moving crew. These effects cannot be validated by varying the steady-state formula alone.

There is no evidence-derived universal minimum number of trajectories. A concrete **pilot design**, not a validation guarantee, is:

- Ten declared scenario families, each run under ten physically consistent parameter configurations and two stochastic seeds: **200 trajectories**.
- Allocate six configurations to training, two to validation and two to test before generating windows: **120 / 40 / 40 trajectories**.
- Include an additional challenge set with previously unseen fault combinations or manoeuvre shapes; report it separately instead of tuning on it.
- Choose scenario duration from the slowest settling time and full transition/recovery, not animation duration. Choose integration and logging intervals from the fastest modelled actuator/sensor dynamics and document convergence checks.

This is a starting budget for learning and debugging. Repeated synthetic samples do not turn uncertain model parameters into measured facts. Parameter ranges must come from a design/bench basis where available; otherwise label them as assumptions and show sensitivity to them. Do not randomly choose unphysical mass/inertia combinations.

### 6.3 Baselines and meaningful evaluation

Implement an analytic equilibrium reference plus a torque-limited conventional feedback controller first. Compare a learned dynamics model against the physics baseline on one-step prediction and free rollouts. Compare learned or optimized control against the same baseline under identical configurations and disturbances.

Report gravity error distributions, peak and time-integrated lateral loading, time outside a declared target band, transition/recovery time, torque/speed/power constraint violations, hub angular momentum, estimation error, and fault handling. Keep normal tracking separate from correctly identified infeasible cases. Do not label constraint-violating actions “optimal” because they reduce a single scalar loss.

Before fitting, verify zero-external-torque angular-momentum conservation, finite-torque spin-up/down, known steady equilibria, frame/sign conventions, sensor latency and saturation. Repeat numerical-resolution checks only where they change the conclusion. After fitting, hold out complete episodes/configurations. For any later human-response analysis, hold out participants or campaigns; randomly splitting repeated observations from the same subject will exaggerate generalization.

## 7. Exact first files and preprocessing order

1. **CED `DATAPRBS.csv` and `DATAUNIF.csv`.** Parse the true header order, remove the trailing empty column and create five independent experiment IDs. Derive relative time explicitly. Train the first input/output dynamics baseline in the dataset's observed output units; do not claim that it learns gravity.
2. **The pinned Basilisk source/documentation plus a declared habitat configuration.** Implement torque-driven ring/hub dynamics and record the logs in Section 6. No synthetic trajectory data were generated in this research pass.
3. **PIMS two binary/header pairs.** Decode with the verified dtype, reconstruct time from the official rule, preserve frame metadata, apply documented bias handling and keep these as labelled optional disturbance examples.
4. **AGBRESA S1–S4.** Parse each actual delimiter/sheet, preserve missing values and original scores, and retain positional IDs where explicit IDs are absent. Use them for a distinct human-response analysis or reference corpus.
5. **OSD-595/596 transformed CSVs plus ISA ZIPs.** Use them only for an explicit organism-response question or demonstration of gravity-exposure metadata. They are not necessary to train the first mechanical controller.
6. **Rosetta's two products plus their labels.** Begin actuator-health and irregular-telemetry analysis if it is in scope; do not pretend the estimated friction channel is a motor action. Avoid downloading the full ~1.6 GB archive until a mission-wide question requires it.
7. **Acquire the OpenMCT raw logs next if reachable through the public repository.** Verify files and current/timing semantics before promoting the record beyond metadata-only status.

The smallest defensible start is therefore **the five measured CED trajectories as an external pipeline check plus a newly generated, clearly labelled finite-torque ring/hub dataset with a held-out configuration split**. The human and OSDR data remain valuable supporting evidence, but are not prerequisites for that first numerical experiment.

## 8. Search coverage, access limits and negative findings

The investigation covered NASA OSDR studies and documented APIs, EDA/SPLASH, NASA PIMS and NTRS, NASA/ESA PDS, NASA NLSP/LSDA, DLR records, official ESA campaign material, university control benchmarks, and discovery queries for Zenodo/Mendeley/other original deposits. Searches included artificial gravity, centrifugation, variable/partial/hypergravity, RPM, motor torque/speed, reaction-wheel data, motor system identification, vestibular/motion-sickness and cardiovascular response. A dataset was shortlisted for useful evidence rather than merely containing one keyword.

The [NASA NLSP page](https://www.nasa.gov/hrp/nlsp/) distinguishes open UTMB bed-rest campaigns 1, 3 and 11 from a broader request workflow. Their destination records were identified but their file schemas were not readable in this check. This is not evidence that every NLSP dataset is restricted or that every record is directly downloadable. Additional AGBRESA raw data described as author-available are recorded as access-request-required; no researchers were contacted and no request was submitted.

The [DLR tolerability](https://elib.dlr.de/136486/) and [protocol](https://elib.dlr.de/190283/) records provide articles, not additional verified raw-data releases. ESA's campaign material supplies context. The old PIMS example host failed; current samples were located through NASA's official how-to page and downloaded successfully. The Mendeley raw-file access limitation is kept separate from its readable metadata. Tool-specific access failures are not interpreted as proof a dataset is absent.

No complete measured rotating-habitat control corpus was verified. No full-scale ring inertia, motor command-to-torque calibration, hub-coupled trajectory, or synchronized astronaut-floor load label was found for this 100 m reference design. No public biological result was reclassified as motor-control training data. No unmeasured synthetic value has been represented as an observation.

## 9. Reuse and reproducibility

Keep the original files, exact repository identifiers, version, retrieval date, hashes and preprocessing history. “Publicly downloadable” and “explicit reuse license verified” are separate manifest fields. PLOS/AGBRESA and the OpenMCT repository state Creative Commons terms; Basilisk states ISC software terms; OSDR states its NASA open-access/SMD policy; the NASA Rosetta catalog marks `us-pd`; explicit CED/PIMS dataset terms were not verified in the inspected material.

The sample package preserves numerical values and metadata. It does not supply invented group-code meanings, repaired physiological scores, synchronized unrelated studies, real spacecraft controller labels, or a trained model. Those distinctions are the basis for a reproducible next experiment.
