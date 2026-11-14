# Proposed scientific experiments

**E1 — measured actuator system identification:** Obtain Uppsala Coupled Electric Drives CSV originals, fit one-step voltage→response predictors, and hold out experiments. This validates the pipeline but does not calibrate a spacecraft ring. Source: https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives.

**E2 — spacecraft physics/controller:** Vary ring/hub inertias, friction, thrust ramps and sensor bias in whole-episode splits. Compare analytical reference with torque-limited PID and a constrained MPC using gravity-vector tracking, actuator power and lateral loads. Current working E2 example is in `outputs/simulations/`.

**E3 — Earth physiology:** Acquire public E4 wearable data; establish next-window heart-rate forecasting baselines with subject holdouts. Keep original device, cohort and task labels.

**E4 — NHANES skeletal baseline:** Estimate available BMD outcomes cross-sectionally. This is an association model, not a longitudinal causal model. Follow survey interpretation for population analyses.

**E5 — artificial-gravity biological response:** Acquire consented longitudinal gravity exposure and repeated health outcomes if accessible. Estimate subject-level response curves with confounding controls and appropriate uncertainty; do not infer a treatment from datasets without exposure changes. Compare hypotheses in exploratory simulations before any prospective medical/flight research.
