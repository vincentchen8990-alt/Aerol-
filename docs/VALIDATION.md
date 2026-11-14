# Verification of the delivered research prototype

## Executed (11 Nov 2026)

- Validated all four user CSV schemas, SHA-256 hashes and record counts (9 / 26 / 11 / 16).
- Inspected the optional original Excel workbook and confirmed eight sheets.
- Recovered prior research manifest (40 records) and variable dictionary (102 mappings).
- Generated 70 synthetic dynamical episodes × 120 transitions = 8,400 samples.
- Trained a PyTorch MLP with 49 training, 10 validation and 11 **held-out spacecraft episodes**.
- Generated 600 mission telemetry samples for a physical 1,200 second example.
- Validated API routes, basic data import behavior, steady theoretical RPM, action/reaction angular momentum, transient gravity vector and source-separation rules using automated tests.
- Ran `python -m pytest -q`; output: **18 passed**.

## Sample numerical results

The bundled `outputs/metrics/dynamics_training.json` has full training history and held-out MAE/RMSE per feature. The held-out one-step learned dynamics MAE was about 0.00287 rad/s in ring speed and about 0.59 m/s² in radial apparent acceleration. This is an approximation to the synthetic simulator, not external validation.

The example PID mission has about 0.0143G mean absolute local gravity error and 0.1406G maximum absolute sampled error. These figures depend on the assumed torque limit, inertia and transient thrust schedule; they are not human safety thresholds or mission performance predictions.

## Unverified / blocked

- Outbound DNS/network access prevented downloads of actual participants' raw E4/NHANES data.
- Therefore **no real-human physiological or BMD checkpoints were trained**. Their scripts exist and explicitly reject nonexistent files.
- React/Three.js dependency installation/build not performed because npm registry was not reachable. The dependency-free 3D viewer was served by FastAPI and has API/static route tests.
- Real actuator/hub measurements and actual human artificial-gravity dose-response data are not included.
- No clinician review, hardware-in-the-loop test, system identification against a real ring, safety certification or flight certification performed.

## Future validation

1. Acquire data in an authorized Internet-enabled environment, respecting study licensing.
2. Validate raw file formats and create participant-held-out Earth physiology models.
3. Acquire actual human gravity-exposure and longitudinal outcome data; test causal identifiability before making personal gravity claims.
4. Validate mechanical model against hardware with synchronized commanded torque, applied torque, inertial ring speed, hub response and floor IMUs.
5. Add full 6-DOF dynamics, robust sensor fusion, safety limits and formal verification before considering any physical deployment.
