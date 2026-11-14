"""Archived measured-ISS services used by the API: model status, one-step forecasts and replays.

Rules enforced here (from the handoff's usage notes):
* one-step forecasts only, each from fresh *measured* history whose timestamps prove the cadence;
* a gap resets the history -- nothing is interpolated and predictions never feed back as inputs;
* a negative RPM forecast is marked invalid, its derived g is suppressed, and the raw unclipped
  model output is kept in diagnostics;
* a missing/altered model or replay file yields an explicit "unavailable" status, never a substitute.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from gravity_ai.measured import assets
from gravity_ai.measured.assets import AssetError
from gravity_ai.measured.inference import (PIMS_AXES, PIMS_CADENCE_S, PIMS_HISTORY_SAMPLES, RPM_CADENCE_S,
                                           RPM_HISTORY_SAMPLES, MeasuredInputError, PimsModel, RpmModel,
                                           format_timestamp, nominal_radial_g, parse_timestamps, pims_histories,
                                           rpm_histories)

MODE = 'ARCHIVED_MEASUREMENTS'
RPM_MODEL_ID = 'mars_s_rpm_next_second_mlp'
PIMS_MODEL_ID = 'iss_mams_ossbtmf_next_16s_mlp'
DERIVED_G_LABEL = 'derived nominal radial g'
DERIVED_G_FORMULA = 'r*(2*pi*RPM/60)^2/9.80665 with r = 0.15 m (published nominal radius)'
DERIVED_G_NOTE = ('Physics calculation at the nominal radius; not directly measured gravity, crew exposure '
                  'or a learned human-health prediction.')
SEPARATION_NOTE = ('Separate from the hypothetical 100 m ring simulator and its 13-input/5-output simulated '
                   'dynamics checkpoint (outputs/runs/selected_dynamics.json), which is not modified or reused.')
EVALUATION_NOTE = ('The MHU-2 mission (RPM test set) and the 10-11 Nov 2026 PIMS test days have already been '
                   'examined. Any retraining or architecture change needs a new, reserved evaluation dataset '
                   'before new performance claims are made.')
STATUS_FORECAST = 'one_step_from_fresh_measured_history'
STATUS_INVALID_NEGATIVE = 'invalid_negative_rpm'
STATUS_INSUFFICIENT = 'insufficient_contiguous_measured_history'
STATUS_MODEL_UNAVAILABLE = 'model_unavailable'
CLIP_LABELS = {'initial_recorded_spin_up': 'Initial recorded spin-up',
               'early_steady_operation': 'Early steady operation',
               'first_recorded_slowdown': 'First recorded slowdown'}


class ModelUnavailable(Exception):
    def __init__(self, status: dict):
        super().__init__(status.get('load_error') or 'model unavailable')
        self.status = status


# ---- model loading ---------------------------------------------------------------------------------
_models: dict = {}


def _load(kind: str):
    rel, cls, model_id = {'rpm': (assets.RPM_MODEL, RpmModel, RPM_MODEL_ID),
                          'pims': (assets.PIMS_MODEL, PimsModel, PIMS_MODEL_ID)}[kind]
    status = {'model_id': model_id, 'file': rel, 'root': assets.display_path(assets.measured_root()),
              'available': False, 'load_error': None, 'load_error_code': None}
    try:
        doc, info = assets.load_verified(rel)
        key = (kind, info['sha256'], info['root'])
        if key not in _models:
            _models[key] = cls(doc)
        status.update(available=True, sha256=info['sha256'], verified_against=info['verified_against'])
        return _models[key], status
    except AssetError as e:
        status.update(load_error=str(e), load_error_code=e.code)
    except (ValueError, KeyError, TypeError, IndexError) as e:
        status.update(load_error=f'model file could not be interpreted: {type(e).__name__}: {e}',
                      load_error_code='model_invalid')
    return None, status


def load_rpm_model():
    return _load('rpm')


def load_pims_model():
    return _load('pims')


def _report(rel):
    try:
        return assets.load_verified(rel)[0]
    except AssetError:
        return None


def _pims_source_manifest():
    try:
        return assets.load_verified(assets.PIMS_SOURCE_MANIFEST)[0]
    except AssetError:
        return None


# ---- descriptions ----------------------------------------------------------------------------------
def hardware() -> dict:
    doc, info = assets.load_verified(assets.HARDWARE)
    mars = doc['mars_s_mhu1_mhu2']
    radius, setting = mars['nominal_radius_m'], mars['nominal_rotation_setting_rpm']
    return {'data_kind': 'PUBLISHED_REFERENCE_SPECIFICATIONS', 'profiles': doc, 'provenance': info,
            'derived': {'mars_s_nominal_setting_radial_g': nominal_radial_g(setting, radius),
                        'label': f'{DERIVED_G_LABEL} at the nominal {setting} RPM setting and {radius} m radius',
                        'formula': DERIVED_G_FORMULA, 'note': DERIVED_G_NOTE},
            'visualization_note': ('The ISS dimensions constrain a schematic proportional view only. The 109 m '
                                   'solar-array span is a span, not a ring radius; none of these dimensions are '
                                   'command-response training labels.')}


def _rpm_metrics(report):
    if not report:
        return None
    test = report['test_metrics']
    rec = report['recursive_metrics']['groups']['uniform_10_minute_origins']['models']
    return {'test_mission': 'MHU-2 (independent 2017 mission)', 'unit': 'RPM',
            'one_step_rmse': {k: test[k]['all']['rmse_rpm'] for k in
                              ('mlp_residual', 'ridge_alpha_1', 'moving_mean_8', 'persistence')},
            'one_step_mae': {k: test[k]['all']['mae_rpm'] for k in
                             ('mlp_residual', 'ridge_alpha_1', 'moving_mean_8', 'persistence')},
            'changing_history_rmse': {k: test[k]['active_past_history']['rmse_rpm'] for k in ('mlp_residual', 'persistence')},
            'changing_history_windows': test['mlp_residual']['active_past_history']['count'],
            'test_windows': test['mlp_residual']['all']['count'],
            'negative_one_step_predictions': test['mlp_residual']['all']['negative_predictions'],
            'recursive_60s_rmse': {k: rec[k]['60']['rmse_rpm'] for k in ('mlp_residual', 'persistence')},
            'recursive_status': 'FAILED: recursive forecasts are unstable; not exposed by this integration'}


def _pims_metrics(report):
    if not report:
        return None
    test = report['one_step_test']
    return {'test_period_utc': [report['splits']['test']['first_utc'], report['splits']['test']['last_utc']],
            'unit': 'micro-g', 'test_windows': test['mlp_residual']['samples'],
            'one_step_rmse_micro_g': {k: test[k]['overall_rmse_micro_g'] for k in ('mlp_residual', 'ridge_residual', 'persistence')},
            'one_step_mae_micro_g': {k: test[k]['overall_mae_micro_g'] for k in ('mlp_residual', 'ridge_residual', 'persistence')},
            'comparison_note': ('The selected MLP and Ridge are almost tied (Ridge has the lower MAE); this is not '
                                'evidence of neural superiority over Ridge.')}


def rpm_model_status() -> dict:
    model, status = load_rpm_model()
    doc = model.doc if model else {}
    status.update({
        'name': doc.get('name', 'mlp_residual'), 'role': 'one-step prediction only; never a motor command',
        'hardware': 'JAXA MARS-S centrifuge, ISS Kibo CBEF (MHU-1/MHU-2 configuration)',
        'input': 'eight consecutive measured RPM samples, oldest to newest: shape (8,) or (N, 8)',
        'input_shape': [RPM_HISTORY_SAMPLES], 'cadence_s': RPM_CADENCE_S, 'history_span_s': 7.0, 'horizon_s': 1.0,
        'output': 'next measured RPM (1 s ahead)', 'output_unit': 'RPM',
        'training_mission': 'MHU-1 (2016)', 'test_mission': 'MHU-2 (2017)',
        'one_step_only': True, 'used_for_control': False, 'torque_response_model': False,
        'scope': doc.get('scope', 'Observed closed-loop RPM forecaster; not torque-response identification or a controller'),
        'unknown_inputs': doc.get('unknown_inputs'), 'use_decision_after_test': doc.get('use_decision_after_test'),
        'metrics': _rpm_metrics(_report(assets.RPM_REPORT)), 'separation': SEPARATION_NOTE})
    return status


def pims_model_status() -> dict:
    model, status = load_pims_model()
    manifest = _pims_source_manifest()
    frame = manifest['validation']['postparse']['data_coordinate_system'] if manifest else None
    sensor = manifest['files'][0]['sensor_coordinate_system'] if manifest else None
    status.update({
        'name': model.name if model else 'mlp_residual', 'role': 'one-step prediction only',
        'instrument': 'NASA MAMS OSS bias-compensated trimmed-mean-filtered acceleration (PIMS archive, sensor ossbtmf)',
        'input': 'eight three-axis [ax_g, ay_g, az_g] measurements in standard gravity g, exactly 16 s apart',
        'input_shape': [PIMS_HISTORY_SAMPLES, len(PIMS_AXES)], 'axes': list(PIMS_AXES), 'units': 'g',
        'display_units': 'micro-g = g x 1e6', 'cadence_s': PIMS_CADENCE_S, 'history_span_s': 112.0, 'horizon_s': 16.0,
        'recorded_dates_utc': ['2026-11-01', '2026-11-11'],
        'coordinate_frame': frame['name'] if frame else 'SSAnalysis',
        'coordinate_frame_comment': frame.get('comment') if frame else None,
        'sensor_location_comment': sensor.get('comment') if sensor else None,
        'used_for_control': False,
        'metrics': _pims_metrics(_report(assets.PIMS_REPORT)),
        'separation': ('Separate dataset from the 2016/2017 MARS-S logs: different dates, instrument and location. '
                       'Not simultaneous or co-located telemetry; not upsampled.')})
    return status


def models_summary() -> dict:
    """Compact entry for /api/models. Never raises."""
    rpm, pims = load_rpm_model()[1], load_pims_model()[1]
    keep = ('model_id', 'available', 'load_error', 'file', 'sha256')
    return {'data_type': 'ARCHIVED_MEASURED_ISS_MODELS', 'live_telemetry': False, 'used_for_control': False,
            'mars_s_rpm': {k: rpm.get(k) for k in keep}, 'iss_pims_acceleration': {k: pims.get(k) for k in keep},
            'separation': SEPARATION_NOTE}


def status() -> dict:
    hw_error = None
    try:
        hw = hardware()
    except AssetError as e:
        hw, hw_error = None, str(e)
    return {'mode': MODE, 'live_telemetry': False,
            'description': 'Archived ISS measurements replayed with one-step measured-history models',
            'hardware': hw, 'hardware_error': hw_error,
            'models': {'mars_s_rpm': rpm_model_status(), 'iss_pims_acceleration': pims_model_status()},
            'replays': {'mars_s_rpm': _replay_availability(assets.measured_root() / assets.RPM_REPLAY),
                        'iss_pims_acceleration': _replay_availability(assets.pims_replay_path())},
            'evaluation_note': EVALUATION_NOTE, 'separation': SEPARATION_NOTE}


def _replay_availability(path):
    return {'file': assets.display_path(path), 'available': path.is_file()}


# ---- one-step forecasts ----------------------------------------------------------------------------
def rpm_one_step(history_rpm, timestamps) -> dict:
    model, mstatus = load_rpm_model()
    history = _single(history_rpm, 'rpm')
    stamps = parse_timestamps(timestamps, RPM_HISTORY_SAMPLES, RPM_CADENCE_S, require_utc=False)
    if model is None:
        raise ModelUnavailable(mstatus)
    raw = float(model.predict_raw(history)[0])
    valid = raw >= 0
    latest = float(history[-1])
    target = stamps[-1] + timedelta(seconds=RPM_CADENCE_S)
    return {
        'available': True, 'valid': valid, 'status': STATUS_FORECAST if valid else STATUS_INVALID_NEGATIVE,
        'mode': MODE, 'live_telemetry': False,
        'model': {'id': RPM_MODEL_ID, 'name': model.name, 'file': mstatus['file'], 'sha256': mstatus['sha256'],
                  'role': 'one-step prediction only; never a motor command'},
        'units': {'rpm': 'RPM (revolutions per minute)', 'derived_nominal_radial_g': 'g (standard gravity), derived'},
        'cadence_s': RPM_CADENCE_S, 'horizon_s': RPM_CADENCE_S, 'history_samples': RPM_HISTORY_SAMPLES,
        'history_timestamps': [format_timestamp(t) for t in stamps], 'forecast_timestamp': format_timestamp(target),
        'timestamp_timezone': _timezone_text(stamps),
        'history_rpm': history.tolist(),
        'forecast_rpm': raw if valid else None,
        'derived_nominal_radial_g': {
            'label': DERIVED_G_LABEL, 'formula': DERIVED_G_FORMULA, 'note': DERIVED_G_NOTE, 'radius_m': model.radius_m,
            'from_latest_measured_rpm': nominal_radial_g(latest, model.radius_m),
            'from_forecast_rpm': nominal_radial_g(raw, model.radius_m) if valid else None,
            'suppressed_reason': None if valid else 'forecast speed is negative (invalid); no derived g is shown'},
        'diagnostics': {'raw_model_output_rpm': raw, 'clipped': False, 'latest_measured_rpm': latest,
                        'predicted_change_rpm': raw - latest,
                        'invalid_reason': None if valid else 'negative rotor speed is outside the measured operating regime',
                        'input_outside_recorded_range': bool(history.max() > 78.5641)},
        'provenance': {'model_training': 'MHU-1 2016 flight log (Otsuki et al. 2026, Supplementary Data 12)',
                       'independent_test': 'MHU-2 2017 flight log (Supplementary Data 13)',
                       'publication': 'https://doi.org/10.1038/s41467-026-76349-y',
                       'history_source': 'supplied by the caller'},
        'limitations': ['One second ahead from eight fresh measured samples; not a self-running trajectory.',
                        'Future start/stop commands, torque, current and inertia are unobserved.',
                        'Not a torque-response model, controller or human-health model.'],
    }


def pims_one_step(history_g, timestamps_utc) -> dict:
    model, mstatus = load_pims_model()
    history = _single(history_g, 'pims')
    stamps = parse_timestamps(timestamps_utc, PIMS_HISTORY_SAMPLES, PIMS_CADENCE_S, require_utc=True,
                              name='timestamps_utc')
    if model is None:
        raise ModelUnavailable(mstatus)
    forecast = model.predict(history)[0]
    target = stamps[-1] + timedelta(seconds=PIMS_CADENCE_S)
    outside = _outside_recorded_range(history)
    return {
        'available': True, 'valid': True, 'status': STATUS_FORECAST, 'mode': MODE, 'live_telemetry': False,
        'model': {'id': PIMS_MODEL_ID, 'name': model.name, 'file': mstatus['file'], 'sha256': mstatus['sha256'],
                  'role': 'one-step prediction only'},
        'units': 'g', 'axes': list(PIMS_AXES), 'display_units': 'micro-g: multiply g values by 1e6',
        'coordinate_frame': 'SSAnalysis',
        'cadence_s': PIMS_CADENCE_S, 'horizon_s': PIMS_CADENCE_S, 'history_span_s': 112.0,
        'history_timestamps_utc': [format_timestamp(t) for t in stamps], 'forecast_timestamp_utc': format_timestamp(target),
        'history_g': history.tolist(), 'forecast_g': forecast.tolist(),
        'diagnostics': {'predicted_change_g': (forecast - history[-1]).tolist(),
                        'input_outside_recorded_nov_2026_range': outside},
        'provenance': {'source': 'NASA GRC PIMS acceleration archive, MAMS OSS BTMF (sensor ossbtmf)',
                       'archive': 'https://gipoc.grc.nasa.gov/wp/pims/acceleration-archives/',
                       'training_data': '1-7 Nov 2026 UTC; validation 8-9 Nov; held-out test 10-11 Nov',
                       'history_source': 'supplied by the caller'},
        'limitations': ['Filtered quasi-steady acceleration at one sensor; not whole-ISS rigid-body motion or all vibration.',
                        'Not simultaneous or co-located with the 2016/2017 MARS-S records.'],
    }


def _single(values, kind):
    arr = rpm_histories(values) if kind == 'rpm' else pims_histories(values)
    if arr.shape[0] != 1:
        raise MeasuredInputError('invalid_shape', 'this endpoint takes one history per request '
                                 + ('(shape (8,))' if kind == 'rpm' else '(shape (8, 3))'))
    return arr[0]


def _timezone_text(stamps):
    if stamps[0].utcoffset() is None:
        return 'unspecified in the source; preserved as recorded (not relabelled UTC)'
    return 'UTC offset ' + str(stamps[0].utcoffset())


def _outside_recorded_range(history):
    manifest = _pims_source_manifest()
    if not manifest:
        return None
    post = manifest['validation']['postparse']
    lo, hi = np.asarray(post['axis_min_g']), np.asarray(post['axis_max_g'])
    return bool(((history < lo) | (history > hi)).any())


# ---- replays ---------------------------------------------------------------------------------------
def _contiguous_targets(seconds: np.ndarray, cadence: float, history: int) -> np.ndarray:
    """Indices i whose `history` preceding points and i itself are exactly `cadence` apart (no gaps)."""
    if len(seconds) and np.any(np.diff(seconds) <= 0):
        raise AssetError('replay_invalid', 'replay timestamps are not strictly increasing')
    ok = np.abs(np.diff(seconds) - cadence) <= 1e-3
    return np.array([i for i in range(history, len(seconds)) if ok[i - history:i].all()], dtype=int)


def _gaps(seconds, cadence, stamps):
    out = []
    for i in np.flatnonzero(np.abs(np.diff(seconds) - cadence) > 1e-3):
        interval = float(seconds[i + 1] - seconds[i])
        out.append({'after': stamps[i], 'before': stamps[i + 1], 'interval_s': interval,
                    'missing_samples': int(round(interval / cadence)) - 1})
    return out


def rpm_replay() -> dict:
    doc, info = assets.load_verified(assets.RPM_REPLAY)
    model, mstatus = load_rpm_model()
    radius = float(doc.get('nominal_radius_m', 0.15))
    clips = []
    for clip in doc['clips']:
        pts = clip['points']
        stamps = [datetime.fromisoformat(p['timestamp_as_recorded']) for p in pts]
        seconds = np.array([(t - stamps[0]).total_seconds() for t in stamps])
        rpm = np.array([p['measured_rpm'] for p in pts], dtype=np.float64)
        targets = _contiguous_targets(seconds, RPM_CADENCE_S, RPM_HISTORY_SAMPLES)
        raw = {}
        if model is not None and len(targets):
            raw = dict(zip(targets.tolist(), model.predict_raw(np.stack([rpm[i - 8:i] for i in targets])).tolist()))
        rows, gap_before = [], {g['before']: g for g in _gaps(seconds, RPM_CADENCE_S, [p['timestamp_as_recorded'] for p in pts])}
        eligible, last_gap = set(targets.tolist()), None
        for i, p in enumerate(pts):
            gap = gap_before.get(p['timestamp_as_recorded'])
            if gap:
                last_gap = i
            row = {'timestamp_as_recorded': p['timestamp_as_recorded'], 'elapsed_s': float(seconds[i]),
                   'measured_rpm': float(rpm[i]),
                   'derived_nominal_radial_g_from_measured': nominal_radial_g(rpm[i], radius),
                   'missing_samples_before': gap['missing_samples'] if gap else 0,
                   'exported_forecast_rpm': p.get('forecast_for_this_timestamp_rpm'),
                   'exported_forecast_status': p.get('forecast_status')}
            if i in raw:
                value = raw[i]
                valid = value >= 0
                row.update(forecast_status=STATUS_FORECAST if valid else STATUS_INVALID_NEGATIVE,
                           forecast_rpm=value if valid else None, raw_model_output_rpm=value,
                           derived_nominal_radial_g_from_forecast=nominal_radial_g(value, radius) if valid else None,
                           forecast_error_rpm=(value - rpm[i]) if valid else None,
                           forecast_issued_at=pts[i - 1]['timestamp_as_recorded'],
                           history_start=pts[i - 8]['timestamp_as_recorded'])
            else:
                is_eligible = i in eligible
                row.update(forecast_status=STATUS_MODEL_UNAVAILABLE if is_eligible else STATUS_INSUFFICIENT,
                           forecast_rpm=None, raw_model_output_rpm=None, derived_nominal_radial_g_from_forecast=None,
                           forecast_error_rpm=None, forecast_issued_at=None, history_start=None,
                           history_reset_reason=None if is_eligible else ('after_gap' if last_gap is not None else 'clip_start'))
            rows.append(row)
        counts = {}
        for r in rows:
            counts[r['forecast_status']] = counts.get(r['forecast_status'], 0) + 1
        clips.append({'name': clip['name'], 'label': CLIP_LABELS.get(clip['name'], clip['name']),
                      'first_timestamp': pts[0]['timestamp_as_recorded'], 'last_timestamp': pts[-1]['timestamp_as_recorded'],
                      'points_count': len(rows), 'gaps': list(gap_before.values()), 'status_counts': counts,
                      'points': rows})
    mission = _rpm_mission_metadata()
    archive = assets.measured_root() / doc['source_archive']
    archive_sha = assets.file_sha256_cached(archive)
    return {
        'data_kind': 'ARCHIVED_MEASUREMENTS_REPLAY', 'mode': MODE, 'live_telemetry': False,
        'description': doc['data_kind'], 'mission': doc['mission'], 'flight_year': doc['flight_year'],
        'hardware': doc['hardware_scope'], 'nominal_radius_m': radius,
        'mission_recorded_span': mission, 'timestamp_timezone': doc['timestamp_timezone'],
        'cadence_s': RPM_CADENCE_S, 'units': {'measured_rpm': 'RPM', 'forecast_rpm': 'RPM',
                                              'derived_nominal_radial_g': 'g (standard gravity), derived'},
        'forecast_mode': ('one-step: each forecast predicts the recorded second from the eight measured samples '
                          'immediately before it; predictions are never fed back as inputs'),
        'display_rule': doc['display_rule'],
        'derived_g': {'label': DERIVED_G_LABEL, 'formula': DERIVED_G_FORMULA, 'note': DERIVED_G_NOTE},
        'model': mstatus,
        'provenance': {'replay_file': info, 'generated_by': 'real_spacecraft_measurements/export_rpm_replay.py',
                       'source_archive': doc['source_archive'],
                       'source_url': (mission or {}).get('source_url'), 'source_sha256_recorded': (mission or {}).get('sha256'),
                       'source_archive_present': archive_sha is not None,
                       'source_archive_sha256_matches': None if archive_sha is None else archive_sha == (mission or {}).get('sha256'),
                       'publication': 'https://doi.org/10.1038/s41467-026-76349-y (Supplementary Data 13)'},
        'clips': clips,
    }


def _rpm_mission_metadata():
    report = _report(assets.RPM_REPORT)
    if not report:
        return None
    for m in report['source_missions']:
        if m['mission'] == 'MHU-2':
            return {k: m[k] for k in ('mission', 'first_timestamp_as_recorded', 'last_timestamp_as_recorded',
                                      'source_url', 'sha256', 'archive_member', 'timestamp_timezone')}
    return None


def pims_replay() -> dict:
    path = assets.pims_replay_path()
    if not path.is_file():
        raise AssetError('replay_missing', f'{assets.display_path(path)} not found; build it with '
                                           'python -m scripts.build_pims_replay after reacquiring the NASA table')
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except ValueError as e:
        raise AssetError('replay_invalid', f'{assets.display_path(path)} is not valid JSON: {e}') from e
    model, mstatus = load_pims_model()
    clips = []
    for clip in doc['clips']:
        pts = clip['points']
        stamps = [datetime.fromisoformat(p['timestamp_utc'].replace('Z', '+00:00')) for p in pts]
        if any(t.utcoffset() is None or t.utcoffset().total_seconds() != 0 for t in stamps):
            raise AssetError('replay_invalid', 'PIMS replay timestamps must be explicit UTC')
        seconds = np.array([(t - stamps[0]).total_seconds() for t in stamps])
        values = np.array([[p[a] for a in PIMS_AXES] for p in pts], dtype=np.float64)
        targets = _contiguous_targets(seconds, PIMS_CADENCE_S, PIMS_HISTORY_SAMPLES)
        forecasts = {}
        if model is not None and len(targets):
            forecasts = dict(zip(targets.tolist(), model.predict(np.stack([values[i - 8:i] for i in targets]))))
        gap_before = {g['before']: g for g in _gaps(seconds, PIMS_CADENCE_S, [p['timestamp_utc'] for p in pts])}
        eligible, rows, seen_gap = set(targets.tolist()), [], False
        for i, p in enumerate(pts):
            gap = gap_before.get(p['timestamp_utc'])
            seen_gap = seen_gap or bool(gap)
            row = {'timestamp_utc': p['timestamp_utc'], 'elapsed_s': float(seconds[i]),
                   'measured_g': values[i].tolist(), 'segment_id': p['segment_id'], 'split': p['split'],
                   'missing_samples_before': gap['missing_samples'] if gap else 0}
            if i in forecasts:
                f = forecasts[i]
                row.update(forecast_status=STATUS_FORECAST, forecast_g=f.tolist(), forecast_error_g=(f - values[i]).tolist(),
                           forecast_issued_at_utc=pts[i - 1]['timestamp_utc'], history_start_utc=pts[i - 8]['timestamp_utc'])
            else:
                row.update(forecast_status=STATUS_MODEL_UNAVAILABLE if i in eligible else STATUS_INSUFFICIENT,
                           forecast_g=None, forecast_error_g=None, forecast_issued_at_utc=None, history_start_utc=None,
                           history_reset_reason=None if i in eligible else ('after_gap' if seen_gap else 'clip_start'))
            rows.append(row)
        counts = {}
        for r in rows:
            counts[r['forecast_status']] = counts.get(r['forecast_status'], 0) + 1
        clips.append({'name': clip['name'], 'label': clip.get('label', clip['name']), 'split': clip.get('split'),
                      'description': clip.get('description'),
                      'first_timestamp_utc': pts[0]['timestamp_utc'], 'last_timestamp_utc': pts[-1]['timestamp_utc'],
                      'points_count': len(rows), 'gaps': list(gap_before.values()), 'status_counts': counts, 'points': rows})
    return {
        'data_kind': 'ARCHIVED_MEASUREMENTS_REPLAY', 'mode': MODE, 'live_telemetry': False,
        'description': doc.get('description'), 'instrument': doc.get('instrument'),
        'recorded_dates_utc': doc.get('recorded_dates_utc'), 'coordinate_frame': doc.get('coordinate_frame'),
        'coordinate_frame_comment': doc.get('coordinate_frame_comment'), 'sensor_location_comment': doc.get('sensor_location_comment'),
        'cadence_s': PIMS_CADENCE_S, 'units': 'g', 'axes': list(PIMS_AXES),
        'display_units': 'micro-g: displayed values are g x 1e6',
        'forecast_mode': ('one-step: each forecast predicts the sample 16 s after the eight measured samples '
                          '(112 s span) immediately before it; predictions are never fed back as inputs'),
        'display_rule': doc.get('display_rule'),
        'model': mstatus, 'provenance': {**doc.get('provenance', {}), 'replay_file': assets.display_path(path),
                                         'replay_sha256': assets.file_sha256_cached(path)},
        'separation': ('Separate dataset from the 2016/2017 MARS-S RPM logs: not simultaneous, not co-located, '
                       'not upsampled to sub-16-second dynamics.'),
        'clips': clips,
    }
