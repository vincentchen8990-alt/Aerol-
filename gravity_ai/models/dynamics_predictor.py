"""Load a trained dynamics checkpoint and predict in physical units (SIMULATED spacecraft dynamics).

Prediction only: nothing here selects motor commands. The PID/MPC controllers remain in charge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gravity_ai.models.dynamics import DynamicsNet
from gravity_ai.models.dynamics_schema import (FEATURE_NAMES, OUTPUT_NAMES, SCHEMA_VERSION, actuator_torque,
                                               decode_outputs, encode_features)

CHECKPOINT_FORMAT = 'gravity_ai.dynamics_predictor/1'
POSITIVE = ('ring_inertia_kg_m2', 'hub_inertia_kg_m2', 'radius_m', 'dt_s', 'motor_torque_limit_nm',
            'motor_power_limit_w', 'motor_slew_nm_s')
PARAM_KEYS = ('ring_inertia_kg_m2', 'hub_inertia_kg_m2', 'bearing_drag_nms', 'radius_m', 'motor_torque_limit_nm',
              'motor_power_limit_w', 'motor_slew_nm_s')


class SchemaError(ValueError):
    pass


class OutOfEnvelopeError(ValueError):
    pass


class DynamicsPredictor:
    def __init__(self, checkpoint: dict, path: Path | None = None, device: str = 'cpu'):
        if checkpoint.get('format') != CHECKPOINT_FORMAT:
            raise SchemaError(f'unsupported checkpoint format {checkpoint.get("format")!r}')
        schema = checkpoint['schema']
        if schema['schema_version'] != SCHEMA_VERSION:
            raise SchemaError(f'checkpoint schema {schema["schema_version"]} != code schema {SCHEMA_VERSION}')
        if tuple(f['name'] for f in schema['features']) != FEATURE_NAMES:
            raise SchemaError('checkpoint feature order differs from the code schema')
        if tuple(o['name'] for o in schema['outputs']) != OUTPUT_NAMES:
            raise SchemaError('checkpoint output order differs from the code schema')
        self.path = Path(path) if path else None
        self.checkpoint = checkpoint
        self.schema = schema
        self.target_mode = schema['target_mode']
        self.run_id = checkpoint.get('run_id')
        mc = checkpoint['model_config']
        self.model = DynamicsNet(len(FEATURE_NAMES), len(OUTPUT_NAMES), mc['hidden_width'], mc['hidden_layers'])
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device).eval()
        self.device = device
        self.x_mean = np.asarray(checkpoint['input_mean'], dtype=np.float64)
        self.x_std = np.asarray(checkpoint['input_std'], dtype=np.float64)
        self.y_mean = np.asarray(checkpoint['target_mean'], dtype=np.float64)
        self.y_std = np.asarray(checkpoint['target_std'], dtype=np.float64)
        self.envelope = {k: tuple(v) for k, v in checkpoint['feature_envelope'].items()}
        self.dt_values = tuple(checkpoint['dt_values_trained'])

    @classmethod
    def load(cls, path, device: str = 'cpu'):
        path = Path(path)
        if path.is_dir():
            path = path / 'checkpoint_best.pt'
        ckpt = torch.load(path, map_location=device, weights_only=True)
        return cls(ckpt, path, device)

    # ---- input validation -------------------------------------------------------------------
    def validate(self, raw) -> dict:
        """Schema check: exact feature names, equal lengths, finite values, physically valid."""
        keys = set(raw)
        missing, extra = set(FEATURE_NAMES) - keys, keys - set(FEATURE_NAMES)
        if missing or extra:
            raise SchemaError(f'feature names mismatch; missing={sorted(missing)} unexpected={sorted(extra)}')
        arrays = {k: np.atleast_1d(np.asarray(raw[k], dtype=np.float64)) for k in FEATURE_NAMES}
        n = max(a.shape[0] for a in arrays.values())
        for k, a in arrays.items():
            if a.ndim != 1 or a.shape[0] not in (1, n):
                raise SchemaError(f'{k}: expected a scalar or a 1-D array of length {n}')
            if not np.all(np.isfinite(a)):
                raise SchemaError(f'{k}: non-finite value')
            arrays[k] = np.broadcast_to(a, (n,)).copy()
        for k in POSITIVE:
            if np.any(arrays[k] <= 0):
                raise SchemaError(f'{k}: must be > 0')
        if np.any(arrays['bearing_drag_nms'] < 0):
            raise SchemaError('bearing_drag_nms: must be >= 0')
        return arrays

    def envelope_report(self, arrays) -> dict:
        """Which rows lie outside the training envelope (per-feature train min/max, trained dt values)."""
        n = len(arrays['ring_omega_rad_s'])
        outside = np.zeros(n, dtype=bool)
        features = {}
        for k, (lo, hi) in self.envelope.items():
            tol = 1e-9 * max(abs(lo), abs(hi), 1e-12)
            bad = (arrays[k] < lo - tol) | (arrays[k] > hi + tol)
            if bad.any():
                features[k] = int(bad.sum())
            outside |= bad
        dt_ok = np.isclose(arrays['dt_s'][:, None], np.asarray(self.dt_values)[None, :], rtol=0, atol=1e-9).any(axis=1)
        if not dt_ok.all():
            features['dt_s_not_trained'] = int((~dt_ok).sum())
        outside |= ~dt_ok
        return {'rows_outside': int(outside.sum()), 'rows': n, 'by_feature': features, 'mask': outside}

    def envelope_details(self, arrays) -> dict:
        """Per out-of-range feature: rows outside, observed min/max, trained min/max (physical units)."""
        units = {f['name']: f['unit'] for f in self.schema['features']}
        out = {}
        for k, (lo, hi) in self.envelope.items():
            v = arrays[k]
            tol = 1e-9 * max(abs(lo), abs(hi), 1e-12)
            bad = (v < lo - tol) | (v > hi + tol)
            if bad.any():
                out[k] = {'rows_outside': int(bad.sum()), 'value_min': float(v.min()), 'value_max': float(v.max()),
                          'trained_min': lo, 'trained_max': hi, 'unit': units[k]}
        dt_bad = ~np.isclose(arrays['dt_s'][:, None], np.asarray(self.dt_values)[None, :], rtol=0, atol=1e-9).any(axis=1)
        if dt_bad.any():
            out['dt_s'] = {'rows_outside': int(dt_bad.sum()), 'value_min': float(arrays['dt_s'].min()),
                           'value_max': float(arrays['dt_s'].max()), 'trained_values': list(self.dt_values), 'unit': 's'}
        return out

    # ---- prediction ---------------------------------------------------------------------------
    def _predict_arrays(self, arrays) -> np.ndarray:
        x = (encode_features(arrays) - self.x_mean) / self.x_std
        with torch.no_grad():
            z = self.model(torch.as_tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy().astype(np.float64)
        return decode_outputs(z * self.y_std + self.y_mean, arrays, self.target_mode)

    def predict(self, raw, on_out_of_envelope: str = 'error') -> dict:
        """One-step prediction. Returns physical outputs keyed by OUTPUT_NAMES plus envelope info.

        on_out_of_envelope: 'error' raises OutOfEnvelopeError, 'flag' predicts and reports.
        """
        if on_out_of_envelope not in ('error', 'flag'):
            raise ValueError("on_out_of_envelope must be 'error' or 'flag'")
        arrays = self.validate(raw)
        env = self.envelope_report(arrays)
        if env['rows_outside'] and on_out_of_envelope == 'error':
            raise OutOfEnvelopeError(f'{env["rows_outside"]} row(s) outside the training envelope: {env["by_feature"]}')
        y = self._predict_arrays(arrays)
        out = {name: y[:, i] for i, name in enumerate(OUTPUT_NAMES)}
        out['outside_training_envelope'] = env['mask']
        return out

    def step_fn(self, arrays) -> np.ndarray:
        """Unchecked batched step used inside rollouts (inputs already validated)."""
        return self._predict_arrays(arrays)


def rollout(step_fn, ring0, hub0, applied0, requested, axial, params: dict, dt):
    """Recursive multi-step prediction, vectorised over B independent rollouts.

    Supplied: the initial state (ring, hub, applied torque) at the start, the KNOWN planned torque
    commands and axial acceleration for every step (B, n), fixed spacecraft parameters and dt.
    Never supplied: any future simulator state. Velocities come from the previous prediction;
    the actuator state is advanced with the explicit motor model (clamp, slew, power limit).
    Returns physical outputs of shape (B, n, 5) and the propagated applied torque (B, n).
    """
    requested = np.atleast_2d(np.asarray(requested, dtype=np.float64))
    axial = np.atleast_2d(np.asarray(axial, dtype=np.float64))
    b, n = requested.shape
    ring = np.broadcast_to(np.asarray(ring0, dtype=np.float64), (b,)).copy()
    hub = np.broadcast_to(np.asarray(hub0, dtype=np.float64), (b,)).copy()
    applied = np.broadcast_to(np.asarray(applied0, dtype=np.float64), (b,)).copy()
    p = {k: np.broadcast_to(np.asarray(params[k], dtype=np.float64), (b,)) for k in PARAM_KEYS}
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (b,))
    outputs = np.empty((b, n, len(OUTPUT_NAMES)))
    applied_hist = np.empty((b, n))
    for j in range(n):
        arrays = {'ring_omega_rad_s': ring, 'hub_omega_rad_s': hub, 'applied_torque_nm': applied,
                  'requested_torque_nm': requested[:, j], 'axial_accel_m_s2': axial[:, j], 'dt_s': dt_arr, **p}
        y = step_fn(arrays)
        outputs[:, j] = y
        applied = actuator_torque(applied, requested[:, j], ring - hub, dt_arr, p['motor_torque_limit_nm'],
                                  p['motor_power_limit_w'], p['motor_slew_nm_s'])
        applied_hist[:, j] = applied
        ring, hub = y[:, 0], y[:, 1]
    return outputs, applied_hist


def persistence_step(arrays) -> np.ndarray:
    """Baseline: speeds unchanged, radial from current speed, no tangential, axial from known thrust."""
    ring = arrays['ring_omega_rad_s']
    return np.stack([ring, arrays['hub_omega_rad_s'], arrays['radius_m'] * ring**2,
                     np.zeros_like(ring), -arrays['axial_accel_m_s2']], axis=1)
