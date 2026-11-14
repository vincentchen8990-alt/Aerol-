"""Learned spacecraft-dynamics predictions for the API and the Orbital Lab viewer.

PREDICTION ONLY. The PID/MPC controller selects every motor command; the learned model is run
afterwards on the same mission to show how its predictions compare with the simulated physics.
The reference is the analytical simulator (simulated, not measured spacecraft data).
"""
from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import Field

from gravity_ai.api.schemas import SimRequest
from gravity_ai.controllers.mpc import GravityMPC
from gravity_ai.controllers.pid import TorquePID
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.dynamics_predictor import DynamicsPredictor, rollout
from gravity_ai.models.dynamics_schema import FEATURES, rad_s_to_rpm
from gravity_ai.predict_dynamics import selected_run_dir
from gravity_ai.simulation.physics import G0, SpacecraftParams, equilibrium_omega
from gravity_ai.simulation.spacecraft import MissionConfig, simulate

router = APIRouter()
UNITS = {name: unit for name, unit, _, _ in FEATURES}
MAX_COMPARE_STEPS = 20000
_cache: dict = {}


def _rel(path):
    if path is None:
        return None
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def load_selected_predictor():
    """Return (predictor or None, status dict). Never raises: the simulator must stay usable."""
    run_dir, source = selected_run_dir(project_root())
    status = {'selection_source': source, 'run_dir': _rel(run_dir), 'checkpoint': None,
              'checkpoint_exists': False, 'checkpoint_loads': False, 'load_error': None}
    if run_dir is None:
        status['load_error'] = 'no trained dynamics run is selected or recorded'
        return None, status
    ck = run_dir / 'checkpoint_best.pt'
    status['checkpoint'] = _rel(ck)
    if not ck.is_file():
        status['load_error'] = f'checkpoint file not found: {_rel(ck)}'
        return None, status
    status['checkpoint_exists'] = True
    key = (str(ck.resolve()), ck.stat().st_mtime_ns)
    if key not in _cache:
        try:
            predictor = DynamicsPredictor.load(ck)
        except Exception as e:  # report, never break the simulation endpoints
            status['load_error'] = f'{type(e).__name__}: {e}'
            return None, status
        horizons = []
        ev = run_dir / 'evaluation' / 'evaluation.json'
        if ev.is_file():
            try:
                horizons = sorted({float(v['horizon_s']) for v in json.loads(ev.read_text(encoding='utf-8'))['rollouts'].values()})
            except (ValueError, KeyError):
                horizons = []
        meta = {'run_id': predictor.run_id, 'epoch': predictor.checkpoint.get('epoch'),
                'target_mode': predictor.target_mode, 'supported_dt_s': list(predictor.dt_values),
                'training_ranges': {k: {'min': lo, 'max': hi, 'unit': UNITS[k]} for k, (lo, hi) in predictor.envelope.items()},
                'evaluated_forecast_horizons_s': horizons,
                'evaluation_report': _rel(ev) if ev.is_file() else None}
        _cache.clear()
        _cache[key] = (predictor, meta)
    status['checkpoint_loads'] = True
    predictor, meta = _cache[key]
    return predictor, {**status, **meta}


def model_status() -> dict:
    _, status = load_selected_predictor()
    return {**status, 'data_type': 'SIMULATED_SPACECRAFT_DYNAMICS',
            'used_by_api_simulate': False,
            'used_by_viewer': 'Orbital Lab "Learned Model vs Simulated Physics" panel (display only)',
            'used_for_predictions': 'POST /api/dynamics/compare, shown in the Orbital Lab viewer panel',
            'used_for_control': False, 'command_selecting_controllers': ['pid', 'mpc']}


class CompareRequest(SimRequest):
    forecast_start_s: float = Field(default=0.0, ge=0)
    forecast_horizon_s: float | None = Field(default=60.0, gt=0, description='None = rest of the mission')
    allow_unsupported_time_step: bool = False


def _whole_steps(value, dt, name):
    k = int(round(value / dt))
    if abs(k * dt - value) > 1e-6 * max(1.0, value):
        raise HTTPException(422, {'code': 'invalid_forecast', 'message': f'{name} must be a whole number of {dt:g} s time steps'})
    return k


def _g(a):
    return np.linalg.norm(a[:, 2:5], axis=1) / G0


def _subsample(n, max_points):
    every = max(1, n // max_points)
    sel = list(range(0, n, every))
    if sel[-1] != n - 1:
        sel.append(n - 1)
    return sel


@router.post('/api/dynamics/compare')
def dynamics_compare(req: CompareRequest):
    """Run the normal PID/MPC mission, then the learned model on the same mission:
    one-step predictions for every step, an open-loop forecast window, and a whole-mission forecast."""
    if req.controller not in ('pid', 'mpc'):
        raise HTTPException(422, 'controller must be pid or mpc')
    n = round(req.duration_s / req.dt_s)
    if n > MAX_COMPARE_STEPS:
        raise HTTPException(422, {'code': 'too_many_steps', 'message': f'comparison limited to {MAX_COMPARE_STEPS} steps; '
                                  'increase the time step or shorten the mission'})
    predictor, status = load_selected_predictor()
    if predictor is None:
        raise HTTPException(503, {'code': 'model_unavailable', 'message': f'learned dynamics model unavailable: {status["load_error"]}',
                                  **status})
    supported = list(predictor.dt_values)
    dt_ok = any(abs(req.dt_s - d) < 1e-9 for d in supported)
    if not dt_ok and not req.allow_unsupported_time_step:
        raise HTTPException(422, {'code': 'unsupported_time_step', 'supported_dt_s': supported,
                                  'message': f'The learned model was trained at time steps of '
                                             f'{", ".join(f"{d:g}" for d in supported)} s; {req.dt_s:g} s is not supported. '
                                             'The simulation itself is unaffected.'})
    k0 = _whole_steps(req.forecast_start_s, req.dt_s, 'forecast_start_s')
    if k0 >= n:
        raise HTTPException(422, {'code': 'invalid_forecast', 'message': 'forecast_start_s must be before the end of the mission'})
    nh = n - k0 if req.forecast_horizon_s is None else _whole_steps(req.forecast_horizon_s, req.dt_s, 'forecast_horizon_s')
    truncated = k0 + nh > n
    nh = min(nh, n - k0)

    p = SpacecraftParams(radius_m=req.radius_m, ring_inertia_kg_m2=req.ring_inertia_kg_m2,
                         hub_inertia_kg_m2=req.hub_inertia_kg_m2, motor_torque_limit_nm=req.motor_torque_limit_nm)
    m = MissionConfig(duration_s=req.duration_s, dt_s=req.dt_s, peak_thrust_g=req.peak_thrust_g,
                      target_gravity_g=req.target_gravity_g)
    try:
        rows = simulate(p, m, controller=TorquePID() if req.controller == 'pid' else GravityMPC(), max_points=None)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(422, str(e))
    ring0 = equilibrium_omega(p.radius_m, m.target_gravity_g * G0, 0.)  # simulate() starts here (initialize_spinning)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    t = col('time_s')
    ring_next, hub_next, applied_next = col('ring_omega_rad_s'), col('hub_omega_rad_s'), col('applied_motor_torque_nm')
    requested, axial = col('requested_motor_torque_nm'), col('axial_accel_m_s2')
    params = {'ring_inertia_kg_m2': p.ring_inertia_kg_m2, 'hub_inertia_kg_m2': p.hub_inertia_kg_m2,
              'bearing_drag_nms': p.bearing_drag_nms, 'radius_m': p.radius_m, 'motor_torque_limit_nm': p.motor_torque_limit_nm,
              'motor_power_limit_w': p.motor_power_limit_w, 'motor_slew_nm_s': p.motor_slew_nm_s}
    ring_pre, hub_pre, applied_pre = np.r_[ring0, ring_next[:-1]], np.r_[0., hub_next[:-1]], np.r_[0., applied_next[:-1]]
    raw = {'ring_omega_rad_s': ring_pre, 'hub_omega_rad_s': hub_pre, 'applied_torque_nm': applied_pre,
           'requested_torque_nm': requested, 'axial_accel_m_s2': axial, 'dt_s': req.dt_s, **params}
    arrays = predictor.validate(raw)
    env = predictor.envelope_report(arrays)
    details = predictor.envelope_details(arrays)
    one = predictor.predict(raw, on_out_of_envelope='flag')
    names = ('next_ring_omega_rad_s', 'next_hub_omega_rad_s', 'crew_radial_m_s2', 'crew_tangential_m_s2', 'crew_axial_m_s2')
    truth = np.stack([ring_next, hub_next, col('crew_gravity_radial_m_s2'), col('crew_gravity_tangential_m_s2'),
                      col('crew_gravity_axial_m_s2')], 1)
    pred1 = np.stack([one[k] for k in names], 1)
    whole, _ = rollout(predictor.step_fn, ring0, 0., 0., requested[None], axial[None], params, req.dt_s)
    whole = whole[0]

    # Forecast window: start from the simulated state at time k0*dt, then only the model's own predictions.
    f_ring0, f_hub0, f_app0 = ring_pre[k0], hub_pre[k0], applied_pre[k0]
    fc, fc_applied = rollout(predictor.step_fn, f_ring0, f_hub0, f_app0, requested[None, k0:k0 + nh],
                             axial[None, k0:k0 + nh], params, req.dt_s)
    fc, fc_applied = fc[0], fc_applied[0]
    fc_inputs = {**{k: np.broadcast_to(v, (nh,)) for k, v in params.items()},
                 'ring_omega_rad_s': np.r_[f_ring0, fc[:-1, 0]], 'hub_omega_rad_s': np.r_[f_hub0, fc[:-1, 1]],
                 'applied_torque_nm': np.r_[f_app0, fc_applied[:-1]], 'requested_torque_nm': requested[k0:k0 + nh],
                 'axial_accel_m_s2': axial[k0:k0 + nh], 'dt_s': np.full(nh, req.dt_s)}
    fc_env = predictor.envelope_details(fc_inputs)
    fc_truth = truth[k0:k0 + nh]
    rpm_err = rad_s_to_rpm(fc[:, 0] - fc_truth[:, 0])
    g_err = _g(fc) - _g(fc_truth)

    one_step = {k: {'mae': float(np.mean(np.abs(pred1[:, i] - truth[:, i])))} for i, k in enumerate(names)}
    one_step['ring_speed_mae_rpm'] = float(rad_s_to_rpm(one_step['next_ring_omega_rad_s']['mae']))
    one_step['gravity_magnitude_mae_g'] = float(np.mean(np.abs(_g(pred1) - _g(truth))))
    one_step['description'] = ('each step predicted from the simulated state one step earlier '
                               '(the model never builds on its own predictions)')

    def at(sec):
        i = int(round(sec / req.dt_s)) - 1
        if i < 0 or i >= n:
            return None
        return {'time_s': float(t[i]), 'ring_speed_abs_error_rpm': float(abs(rad_s_to_rpm(whole[i, 0] - truth[i, 0]))),
                'gravity_abs_error_g': float(abs(_g(whole[i:i + 1])[0] - _g(truth[i:i + 1])[0]))}

    sel = _subsample(n, req.max_points)
    fsel = _subsample(nh, req.max_points)
    return {
        'data_type': 'SYNTHETIC_SPACECRAFT_DYNAMICS',
        'reference': 'simulated physics (analytical simulator), not measured spacecraft data',
        'ground_truth': 'analytical simulator (same as /api/simulate)',
        'mission': req.model_dump(),
        'active_controller': req.controller,
        'controller_note': 'Every motor command was selected by the PID/MPC controller. The learned model only predicts.',
        'learned_model': {'run_id': predictor.run_id, 'epoch': status.get('epoch'), 'checkpoint': status['checkpoint'],
                          'selection_source': status['selection_source'], 'role': 'prediction_only',
                          'supported_dt_s': supported, 'time_step_supported': dt_ok},
        'steps': n, 'dt_s': req.dt_s,
        'training_envelope': {'rows_outside': env['rows_outside'], 'rows': n, 'by_feature': env['by_feature'],
                              'details': details, 'inside_all_ranges': env['rows_outside'] == 0,
                              'note': 'Inside the recorded training ranges is not a guarantee of accuracy; '
                                      'outside them the predictions are extrapolation.'},
        'one_step': one_step,
        'forecast': {
            'description': 'open-loop: starts from the simulated state at start_s, then feeds the model its own '
                           'predicted state for every later step; never reset to the simulator',
            'command_source': f'{req.controller.upper()} torque commands recorded in the reference simulation and the planned '
                              'thrust schedule, replayed open-loop (not re-planned from the forecast state)',
            'start_s': float(k0 * req.dt_s), 'requested_horizon_s': req.forecast_horizon_s,
            'horizon_s': float(nh * req.dt_s), 'steps': nh, 'truncated_at_mission_end': truncated,
            'start_state': {'source': 'simulated physics at start_s', 'ring_rpm': float(rad_s_to_rpm(f_ring0)),
                            'hub_rpm': float(rad_s_to_rpm(f_hub0)), 'applied_torque_nm': float(f_app0)},
            'error_at_end': {'time_s': float(t[k0 + nh - 1]), 'ring_speed_error_rpm': float(rpm_err[-1]),
                             'gravity_error_g': float(g_err[-1])},
            'max_abs_error': {'ring_speed_rpm': float(np.max(np.abs(rpm_err))), 'gravity_g': float(np.max(np.abs(g_err)))},
            'forecast_states_outside_training_range': fc_env,
            'series': [{'time_s': float(t[k0 + j]), 'simulator_ring_rpm': float(rad_s_to_rpm(fc_truth[j, 0])),
                        'forecast_ring_rpm': float(rad_s_to_rpm(fc[j, 0])),
                        'simulator_gravity_g': float(_g(fc_truth[j:j + 1])[0]), 'forecast_gravity_g': float(_g(fc[j:j + 1])[0])}
                       for j in fsel],
        },
        'rollout': {'description': 'open-loop recursive prediction of the whole mission from the initial state, '
                                   'replaying the controller commands and thrust schedule; no simulator states fed back',
                    'abs_error_at': {f'{s}s': at(s) for s in (10, 30, 60)}, 'abs_error_at_end': at(n * req.dt_s)},
        'series': [{'time_s': float(t[i]), 'simulator_ring_rpm': float(rad_s_to_rpm(truth[i, 0])),
                    'one_step_ring_rpm': float(rad_s_to_rpm(pred1[i, 0])), 'rollout_ring_rpm': float(rad_s_to_rpm(whole[i, 0])),
                    'simulator_gravity_g': float(_g(truth[i:i + 1])[0]), 'one_step_gravity_g': float(_g(pred1[i:i + 1])[0]),
                    'rollout_gravity_g': float(_g(whole[i:i + 1])[0])} for i in sel],
    }
