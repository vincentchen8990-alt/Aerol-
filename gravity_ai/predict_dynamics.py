"""Reproducible inference example for the learned spacecraft dynamics predictor (simulated data).

  python -m gravity_ai.predict_dynamics                 # latest completed run
  python -m gravity_ai.predict_dynamics --run outputs/runs/<run_id>

Loads the checkpoint from disk, validates the input schema, applies the saved preprocessing,
runs the network in eval mode, converts back to physical units and compares with the analytical
simulator for the same input. Prediction only; no motor command is selected.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.dynamics_predictor import DynamicsPredictor, OutOfEnvelopeError, SchemaError
from gravity_ai.models.dynamics_schema import OUTPUTS, rad_s_to_rpm
from gravity_ai.simulation.physics import G0, SpacecraftParams, SpacecraftState, equilibrium_omega, step
from gravity_ai.train.dynamics import LATEST_POINTER


SELECTED_POINTER = 'selected_dynamics.json'
ENV_RUN_DIR = 'GRAVITY_DYNAMICS_RUN_DIR'


def latest_run_dir(root: Path) -> Path | None:
    pointer = root / 'outputs' / 'runs' / LATEST_POINTER
    if not pointer.is_file():
        return None
    return Path(json.loads(pointer.read_text(encoding='utf-8'))['run_dir'])


def selected_run_dir(root: Path) -> tuple[Path | None, str]:
    """The run the application serves, and where that choice came from.

    Order: GRAVITY_DYNAMICS_RUN_DIR environment variable, then the deliberate selection in
    outputs/runs/selected_dynamics.json (not touched by training), then the most recently
    completed run (outputs/runs/latest_dynamics.json, updated by every training run).
    """
    env = os.environ.get(ENV_RUN_DIR)
    if env:
        p = Path(env)
        return (p if p.is_absolute() else root / p), f'environment variable {ENV_RUN_DIR}'
    pointer = root / 'outputs' / 'runs' / SELECTED_POINTER
    if pointer.is_file():
        p = Path(json.loads(pointer.read_text(encoding='utf-8'))['run_dir'])
        return (p if p.is_absolute() else root / p), f'outputs/runs/{SELECTED_POINTER}'
    latest = latest_run_dir(root)
    if latest is not None:
        return latest, f'outputs/runs/{LATEST_POINTER} (no explicit selection)'
    return None, 'no selected or completed run'


def example_input():
    """App/API default spacecraft at 1 g, 0.8 g thrust, +4 MN*m command, dt = 2 s."""
    p = SpacecraftParams()  # radius 100 m, ring 4e9, hub 1e10, drag 2e5, torque 8e6, power 3e6, slew 8e5
    s = SpacecraftState(ring_omega_rad_s=equilibrium_omega(p.radius_m, G0), hub_omega_rad_s=0.0, applied_torque_nm=1.0e6)
    cmd, axial, dt = 4.0e6, 0.8 * G0, 2.0
    raw = {'ring_omega_rad_s': s.ring_omega_rad_s, 'hub_omega_rad_s': s.hub_omega_rad_s,
           'applied_torque_nm': s.applied_torque_nm, 'requested_torque_nm': cmd, 'axial_accel_m_s2': axial,
           'ring_inertia_kg_m2': p.ring_inertia_kg_m2, 'hub_inertia_kg_m2': p.hub_inertia_kg_m2,
           'bearing_drag_nms': p.bearing_drag_nms, 'radius_m': p.radius_m, 'dt_s': dt,
           'motor_torque_limit_nm': p.motor_torque_limit_nm, 'motor_power_limit_w': p.motor_power_limit_w,
           'motor_slew_nm_s': p.motor_slew_nm_s}
    nxt, t = step(p, s, cmd, axial, dt)
    truth = {'next_ring_omega_rad_s': nxt.ring_omega_rad_s, 'next_hub_omega_rad_s': nxt.hub_omega_rad_s,
             'crew_radial_m_s2': t['crew_gravity_radial_m_s2'], 'crew_tangential_m_s2': t['crew_gravity_tangential_m_s2'],
             'crew_axial_m_s2': t['crew_gravity_axial_m_s2']}
    return raw, truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', help='run directory (default: the run the application serves)')
    args = ap.parse_args(argv)
    root = project_root()
    run_dir = Path(args.run) if args.run else selected_run_dir(root)[0]
    if run_dir is None:
        raise SystemExit('No trained run found; run python -m gravity_ai.train.dynamics first')
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    predictor = DynamicsPredictor.load(run_dir)
    print(f'Loaded {predictor.path} (run {predictor.run_id}, epoch {predictor.checkpoint["epoch"]}, '
          f'target_mode {predictor.target_mode}, model in eval mode: {not predictor.model.training})')
    raw, truth = example_input()
    print('Input (physical units):')
    for k, v in raw.items():
        print(f'  {k:<24} {v:.6g}')
    pred = predictor.predict(raw)  # raises on schema or envelope violations
    print(f'Inside training envelope: {not bool(pred["outside_training_envelope"][0])}')
    print(f'{"output":<24}{"predicted":>14}{"simulator":>14}{"error":>12}  unit')
    for name, unit, _ in OUTPUTS:
        p, t = float(pred[name][0]), truth[name]
        print(f'{name:<24}{p:>14.6g}{t:>14.6g}{p - t:>12.3g}  {unit}')
    print(f'ring speed: predicted {float(rad_s_to_rpm(pred["next_ring_omega_rad_s"][0])):.4f} rpm, '
          f'simulator {float(rad_s_to_rpm(truth["next_ring_omega_rad_s"])):.4f} rpm')
    print('Input checks:')
    bad = dict(raw)
    bad.pop('dt_s')
    try:
        predictor.predict(bad)
    except SchemaError as e:
        print(f'  missing feature -> SchemaError: {e}')
    odd = dict(raw, dt_s=3.0)
    try:
        predictor.predict(odd)
    except OutOfEnvelopeError as e:
        print(f'  dt=3.0 s -> OutOfEnvelopeError: {e}')


if __name__ == '__main__':
    main()
