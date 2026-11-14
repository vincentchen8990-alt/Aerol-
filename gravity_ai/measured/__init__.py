"""Archived measured-ISS components, kept separate from the simulated 100 m ring.

* MARS-S centrifuge RPM (JAXA, ISS Kibo CBEF): MHU-1 2016 / MHU-2 2017 flight logs, one-step
  next-second RPM forecaster (8 measured RPM samples in, 1 RPM value out).
* NASA PIMS / MAMS OSS BTMF filtered ISS acceleration: 1-11 November 2026 UTC, SSAnalysis
  frame, one-step 16-second forecaster ((8, 3) samples in g, 3 values in g out).

Neither model is a torque-response model, a controller or a human-health model, and neither is
loaded into the 13-input / 5-output simulated dynamics checkpoint (outputs/runs/selected_dynamics.json).
Inference uses the exported JSON weights and NumPy only.
"""
