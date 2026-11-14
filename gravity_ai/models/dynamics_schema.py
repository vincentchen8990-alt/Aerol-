"""Feature/target contract for the learned spacecraft dynamics predictor (schema dynamics-v2).

SIMULATED SPACECRAFT DYNAMICS ONLY. Nothing in this module describes human physiology.

Data flow
---------
1. The generator stores every transition in physical SI units (columns named below).
2. Model input  = raw_value / physical_scale  (fixed scales listed in FEATURES), then a
   standardisation (mean/std) fitted on TRAINING episodes only and saved in the checkpoint.
3. Model target = physical_output - reference(inputs)  (target_mode="residual")
              or  physical_output                     (target_mode="absolute", the archived
   formulation), then standardised with training-only mean/std.
4. Predictions are decoded in reverse: un-standardise, then add the reference back.

The axial crew-acceleration component equals -axial_accel exactly in this simulator, so in
residual mode its learned residual is identically zero: its error is not evidence of learning.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from gravity_ai.simulation.physics import G0

SCHEMA_VERSION = 'dynamics-v2'
DATA_TYPE = 'SIMULATED_SPACECRAFT_DYNAMICS'
TARGET_MODES = ('residual', 'absolute')

# (raw column = feature name, unit, physical scale applied before standardisation, meaning)
FEATURES = (
    ('ring_omega_rad_s', 'rad/s', 1.0, 'current inertial ring angular velocity'),
    ('hub_omega_rad_s', 'rad/s', 1.0, 'current inertial hub angular velocity'),
    ('applied_torque_nm', 'N*m', 1e7, 'motor torque applied during the previous step (actuator state)'),
    ('requested_torque_nm', 'N*m', 1e7, 'motor torque command for this step'),
    ('axial_accel_m_s2', 'm/s^2', G0, 'spacecraft axial (thrust) acceleration during this step; feature expressed in g'),
    ('ring_inertia_kg_m2', 'kg*m^2', 1e10, 'ring moment of inertia'),
    ('hub_inertia_kg_m2', 'kg*m^2', 1e10, 'hub moment of inertia'),
    ('bearing_drag_nms', 'N*m*s/rad', 1e6, 'bearing drag coefficient on relative ring-hub speed'),
    ('radius_m', 'm', 100.0, 'crew floor radius'),
    ('dt_s', 's', 1.0, 'integration time step'),
    ('motor_torque_limit_nm', 'N*m', 1e7, 'motor torque magnitude limit'),
    ('motor_power_limit_w', 'W', 1e7, 'motor power limit'),
    ('motor_slew_nm_s', 'N*m/s', 1e6, 'motor torque slew-rate limit'),
)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_SCALES = np.array([f[2] for f in FEATURES], dtype=np.float64)

# Physical outputs (simulator truth columns use the same names), in model output order.
OUTPUTS = (
    ('next_ring_omega_rad_s', 'rad/s', 'ring angular velocity after the step'),
    ('next_hub_omega_rad_s', 'rad/s', 'hub angular velocity after the step'),
    ('crew_radial_m_s2', 'm/s^2', 'apparent gravity, radial component at the floor (r*omega_next^2)'),
    ('crew_tangential_m_s2', 'm/s^2', 'apparent gravity, tangential component (-r*alpha_ring)'),
    ('crew_axial_m_s2', 'm/s^2', 'apparent gravity, axial component (-axial_accel)'),
)
OUTPUT_NAMES = tuple(o[0] for o in OUTPUTS)
# Truth that is stored but not a model output: needed to check actuator propagation.
EXTRA_TRUTH = ('next_applied_torque_nm',)

RESIDUAL_REFERENCES = {
    'next_ring_omega_rad_s': 'ring_omega_rad_s',
    'next_hub_omega_rad_s': 'hub_omega_rad_s',
    'crew_radial_m_s2': 'radius_m * ring_omega_rad_s**2',
    'crew_tangential_m_s2': '0',
    'crew_axial_m_s2': '-axial_accel_m_s2',
}


def _as_columns(raw, names):
    return np.stack([np.asarray(raw[n], dtype=np.float64) for n in names], axis=1)


def encode_features(raw) -> np.ndarray:
    """Raw physical columns -> physically scaled features (before standardisation)."""
    return _as_columns(raw, FEATURE_NAMES) / FEATURE_SCALES


def reference_outputs(raw, target_mode: str) -> np.ndarray:
    if target_mode not in TARGET_MODES:
        raise ValueError(f'target_mode must be one of {TARGET_MODES}, got {target_mode!r}')
    n = len(np.asarray(raw['ring_omega_rad_s']))
    if target_mode == 'absolute':
        return np.zeros((n, len(OUTPUT_NAMES)))
    ring = np.asarray(raw['ring_omega_rad_s'], dtype=np.float64)
    return np.stack([
        ring,
        np.asarray(raw['hub_omega_rad_s'], dtype=np.float64),
        np.asarray(raw['radius_m'], dtype=np.float64) * ring**2,
        np.zeros(n),
        -np.asarray(raw['axial_accel_m_s2'], dtype=np.float64),
    ], axis=1)


def truth_outputs(data) -> np.ndarray:
    return _as_columns(data, OUTPUT_NAMES)


def encode_targets(data, target_mode: str) -> np.ndarray:
    return truth_outputs(data) - reference_outputs(data, target_mode)


def decode_outputs(model_targets: np.ndarray, raw, target_mode: str) -> np.ndarray:
    return np.asarray(model_targets, dtype=np.float64) + reference_outputs(raw, target_mode)


def actuator_torque(applied_nm, requested_nm, rel_omega_rad_s, dt_s, torque_limit_nm, power_limit_w, slew_nm_s):
    """Vectorised copy of the motor model inside physics.step (clamp, slew, power limit).

    Used to advance the actuator state in multi-step rollouts. tests/test_dynamics_pipeline.py
    checks it is identical to physics.step's applied torque.
    """
    applied = np.asarray(applied_nm, dtype=np.float64)
    limit = np.asarray(torque_limit_nm, dtype=np.float64)
    req = np.maximum(-limit, np.minimum(limit, np.asarray(requested_nm, dtype=np.float64)))
    allowed = np.asarray(slew_nm_s, dtype=np.float64) * np.asarray(dt_s, dtype=np.float64)
    slewed = np.maximum(applied - allowed, np.minimum(applied + allowed, req))
    max_power_torque = np.asarray(power_limit_w, dtype=np.float64) / np.maximum(np.abs(rel_omega_rad_s), .005)
    out = np.maximum(-max_power_torque, np.minimum(max_power_torque, slewed))
    return np.maximum(-limit, np.minimum(limit, out))


def schema_dict(target_mode: str) -> dict:
    return {
        'schema_version': SCHEMA_VERSION,
        'data_type': DATA_TYPE,
        'features': [{'name': n, 'unit': u, 'physical_scale': s, 'description': d} for n, u, s, d in FEATURES],
        'outputs': [{'name': n, 'unit': u, 'description': d} for n, u, d in OUTPUTS],
        'target_mode': target_mode,
        'target_definition': ('model target = output - reference' if target_mode == 'residual'
                              else 'model target = output'),
        'residual_references': RESIDUAL_REFERENCES if target_mode == 'residual' else None,
        'preprocessing': ['feature = raw / physical_scale',
                          'standardise features with training-split mean/std (std==0 -> 1)',
                          'standardise targets with training-split mean/std (std==0 -> 1)'],
        'not_a_human_model': True,
    }


def fingerprint(arrays: dict) -> str:
    """Content hash of named arrays (stable across re-saves; .npz bytes embed timestamps)."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def rad_s_to_rpm(x):
    return np.asarray(x) * 60.0 / (2 * math.pi)
