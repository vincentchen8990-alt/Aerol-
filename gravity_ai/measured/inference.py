"""Portable NumPy inference and input validation for the two measured-ISS models.

The arithmetic mirrors the handoff's reference implementations exactly
(real_spacecraft_measurements/rpm_predict.py and outputs/pims/inference.py) so the exported
inference test vectors are reproduced. Only one-step prediction is provided here: the RPM MLP
failed recursive (autonomous) forecasting, and this integration exposes no rollout.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

G0 = 9.80665  # standard gravity, m/s^2

RPM_HISTORY_SAMPLES = 8
RPM_CADENCE_S = 1.0
PIMS_HISTORY_SAMPLES = 8
PIMS_AXES = ('ax_g', 'ay_g', 'az_g')
PIMS_CADENCE_S = 16.0
TIMESTAMP_TOLERANCE_S = 1e-3
# The filtered MAMS OSS BTMF product spans roughly +/-5e-6 g. Values above 1e-3 g are not this
# product in standard-gravity units -- most likely micro-g passed where g is required.
PIMS_UNITS_GUARD_ABS_G = 1e-3


class MeasuredInputError(ValueError):
    """Invalid caller input (shape, units, values or timestamps). Never answered with a substitute."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.extra = extra

    def detail(self) -> dict:
        return {'code': self.code, 'message': str(self), **self.extra}


def _numeric_array(value, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
            raise MeasuredInputError('invalid_value', f'{name} must contain only numbers')
        return value.astype(np.float64)

    def check(v):
        if isinstance(v, (list, tuple)):
            for item in v:
                check(item)
        elif isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise MeasuredInputError('invalid_value', f'{name} must contain only numbers (got {type(v).__name__})')

    check(value)
    try:
        return np.asarray(value, dtype=np.float64)
    except ValueError as e:  # ragged nesting
        raise MeasuredInputError('invalid_shape', f'{name} has an irregular (ragged) shape') from e


def rpm_histories(values) -> np.ndarray:
    """Validate measured RPM histories: shape (8,) or (N, 8), oldest to newest -> (N, 8)."""
    arr = _numeric_array(values, 'history_rpm')
    if arr.ndim == 1 and arr.shape[0] == RPM_HISTORY_SAMPLES:
        arr = arr[None, :]
    elif not (arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] == RPM_HISTORY_SAMPLES):
        raise MeasuredInputError('invalid_shape', 'RPM history must be a flat array of eight scalar RPM readings, '
                                 f'oldest to newest: shape (8,) or batched (N, 8); got shape {tuple(arr.shape)}',
                                 expected_shape=[RPM_HISTORY_SAMPLES], received_shape=list(arr.shape))
    if not np.isfinite(arr).all():
        raise MeasuredInputError('nonfinite_value', 'RPM history contains a missing or non-finite reading')
    if (arr < 0).any():
        raise MeasuredInputError('negative_measured_rpm', 'Measured rotor RPM cannot be negative; the recorded '
                                 'MARS-S logs range from 0 to 78.5641 RPM')
    return arr


def pims_histories(values) -> np.ndarray:
    """Validate measured acceleration histories in g: shape (8, 3) or (N, 8, 3) -> (N, 8, 3)."""
    arr = _numeric_array(values, 'history_g')
    shape = (PIMS_HISTORY_SAMPLES, len(PIMS_AXES))
    if arr.shape == shape:
        arr = arr[None, ...]
    elif not (arr.ndim == 3 and arr.shape[0] >= 1 and arr.shape[1:] == shape):
        raise MeasuredInputError('invalid_shape', 'PIMS history must have shape (8, 3): eight three-axis '
                                 '[ax_g, ay_g, az_g] samples in standard gravity g, oldest to newest '
                                 f'(or batched (N, 8, 3)); got shape {tuple(arr.shape)}',
                                 expected_shape=list(shape), received_shape=list(arr.shape))
    if not np.isfinite(arr).all():
        raise MeasuredInputError('nonfinite_value', 'PIMS history contains a missing or non-finite value')
    peak = float(np.max(np.abs(arr)))
    if peak > PIMS_UNITS_GUARD_ABS_G:
        raise MeasuredInputError('units_suspect', f'|acceleration| up to {peak:g} exceeds {PIMS_UNITS_GUARD_ABS_G:g} g. '
                                 'This input is in standard gravity g, not micro-g: the recorded November 2026 '
                                 'MAMS values are about +/-5e-6 g. Divide micro-g values by 1e6.',
                                 max_abs_value=peak, required_units='g')
    return arr


class RpmModel:
    """MARS-S next-second RPM MLP (8 measured RPM samples -> next RPM). Raw output is unclipped."""

    def __init__(self, doc: dict):
        if doc.get('kind') != 'mlp_residual' or doc.get('activation') != 'relu':
            raise ValueError('expected the relu mlp_residual RPM model')
        if (doc.get('context_samples'), doc.get('sample_interval_seconds'), doc.get('horizon_seconds')) != (8, 1, 1):
            raise ValueError('RPM model must use 8 samples at 1 s cadence and a 1 s horizon')
        self.doc = doc
        self.weights = [np.asarray(w, dtype=np.float64) for w in doc['weights']]
        self.biases = [np.asarray(b, dtype=np.float64) for b in doc['biases']]
        shapes = [w.shape for w in self.weights]
        if shapes[0][0] != 8 or shapes[-1][1] != 1 or any(a[1] != b[0] for a, b in zip(shapes, shapes[1:])):
            raise ValueError(f'unexpected RPM weight shapes {shapes}')
        self.x_mean = np.asarray(doc['x_mean'], dtype=np.float64)
        self.x_scale = np.asarray(doc['x_scale'], dtype=np.float64)
        self.y_mean = float(doc['y_mean'])
        self.y_scale = float(doc['y_scale'])
        self.radius_m = float(doc['nominal_radius_m'])
        self.name = doc['name']

    @staticmethod
    def features(history: np.ndarray) -> np.ndarray:
        return np.column_stack((history[:, -1], np.diff(history, axis=1)))

    def predict_raw(self, history_rpm) -> np.ndarray:
        """One-step forecast(s) in RPM, shape (N,). Negative values are returned unclipped."""
        history = rpm_histories(history_rpm)
        value = (self.features(history) - self.x_mean) / self.x_scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight + bias
            if index < len(self.weights) - 1:
                value = np.maximum(value, 0)
        residual = value[:, 0]
        return history[:, -1] + residual * self.y_scale + self.y_mean


class PimsModel:
    """NASA MAMS one-step forecaster: (8, 3) measured g, 16 s apart -> next (3,) in g."""

    def __init__(self, doc: dict):
        if doc.get('format') != 'iss-mams-forecast-v1' or doc.get('kind') != 'mlp_residual':
            raise ValueError('expected the iss-mams-forecast-v1 mlp_residual model')
        if doc.get('units') != 'g' or float(doc.get('dt_s')) != PIMS_CADENCE_S or doc.get('input_shape') != [8, 3]:
            raise ValueError('PIMS model must take (8, 3) inputs in g at 16 s cadence')
        if doc.get('axis_order') != list(PIMS_AXES) or doc.get('output_order') != list(PIMS_AXES):
            raise ValueError('unexpected PIMS axis order')
        self.doc = doc
        self.layers = [(np.asarray(layer['weights'], dtype=np.float64), np.asarray(layer['bias'], dtype=np.float64))
                       for layer in doc['layers']]
        if self.layers[0][0].shape[0] != 24 or self.layers[-1][0].shape[1] != 3:
            raise ValueError('unexpected PIMS layer shapes')
        self.activation = doc['activation']
        if self.activation not in ('tanh', 'relu'):
            raise ValueError('unsupported hidden activation')
        self.x_mean = np.asarray(doc['feature_scaler']['mean'], dtype=np.float64)
        self.x_scale = np.asarray(doc['feature_scaler']['scale'], dtype=np.float64)
        self.y_mean = np.asarray(doc['residual_scaler']['mean'], dtype=np.float64)
        self.y_scale = np.asarray(doc['residual_scaler']['scale'], dtype=np.float64)
        self.name = doc['kind']

    def predict(self, history_g) -> np.ndarray:
        """Next three-axis sample(s) in g, shape (N, 3)."""
        history = pims_histories(history_g)
        last = history[:, -1, :]
        x = (history.reshape(len(history), 24) - self.x_mean) / self.x_scale
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if index < len(self.layers) - 1:
                x = np.tanh(x) if self.activation == 'tanh' else np.maximum(x, 0.0)
        delta_g = x * self.y_scale + self.y_mean
        return last + delta_g


def nominal_radial_g(rpm, radius_m: float = 0.15):
    """Derived nominal radial g = r*(2*pi*RPM/60)^2/9.80665. None for a negative or missing speed.

    A physics calculation at the published nominal radius -- not a measured acceleration,
    crew exposure or learned health prediction."""
    if rpm is None:
        return None
    rpm = float(rpm)
    if not np.isfinite(rpm) or rpm < 0:
        return None
    return float(radius_m * (rpm * 2 * np.pi / 60) ** 2 / G0)


def parse_timestamps(values, count: int, cadence_s: float, *, require_utc: bool, name: str = 'timestamps'):
    """Parse ISO-8601 timestamps and require exactly `cadence_s` between consecutive samples.

    A gap, duplicate, reordering or irregular interval raises MeasuredInputError so that
    non-consecutive observations can never be presented to a model as consecutive."""
    if values is None:
        raise MeasuredInputError('timestamps_required', f'{name} are required: one timestamp per history sample, '
                                 'so the cadence can be checked')
    if not isinstance(values, (list, tuple)) or len(values) != count:
        got = len(values) if isinstance(values, (list, tuple)) else type(values).__name__
        raise MeasuredInputError('invalid_timestamps', f'{name} must list exactly {count} timestamps, one per history '
                                 f'sample (got {got})')
    parsed = []
    for i, text in enumerate(values):
        if not isinstance(text, str):
            raise MeasuredInputError('invalid_timestamps', f'{name}[{i}] must be an ISO-8601 string')
        try:
            parsed.append(datetime.fromisoformat(text.strip()))
        except ValueError as e:
            raise MeasuredInputError('invalid_timestamps', f'{name}[{i}] is not an ISO-8601 timestamp: {text!r}') from e
    aware = {t.utcoffset() is not None for t in parsed}
    if len(aware) > 1:
        raise MeasuredInputError('timestamp_timezone', f'{name} mix timezone-aware and naive values')
    if require_utc and (not aware.pop() or any(t.utcoffset().total_seconds() != 0 for t in parsed)):
        raise MeasuredInputError('timestamp_timezone', f'{name} must be explicit UTC (for example '
                                 '2026-11-10T00:00:07.070Z): the NASA PIMS samples are UTC records')
    for i in range(1, count):
        interval = (parsed[i] - parsed[i - 1]).total_seconds()
        if abs(interval - cadence_s) > TIMESTAMP_TOLERANCE_S:
            kind = 'gap' if interval > cadence_s else ('out-of-order or duplicate samples' if interval <= 0 else 'short interval')
            raise MeasuredInputError(
                'invalid_cadence',
                f'{name}[{i - 1}] and {name}[{i}] are {interval:g} s apart ({kind}); this model requires exactly '
                f'{cadence_s:g} s between consecutive measured samples. A gap cannot be treated as consecutive '
                'observations, so no forecast is made.',
                index=i, interval_s=interval, required_interval_s=cadence_s)
    return parsed


def format_timestamp(t: datetime) -> str:
    text = t.isoformat(timespec='seconds' if t.microsecond == 0 else 'milliseconds')
    return text.replace('+00:00', 'Z')
