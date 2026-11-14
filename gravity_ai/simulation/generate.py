"""Generate SIMULATED spacecraft-dynamics transitions for the learned predictor (schema dynamics-v2).

Labels come from the analytical simulator gravity_ai.simulation.physics.step. This is synthetic
spacecraft data, never measured NASA data and never human physiology.

Reproducibility and frozen splits
---------------------------------
* Episode i is generated from its own random stream default_rng([seed, 0, i, attempt]), so its
  content does not depend on how many episodes are generated.
* Episode i uses dt_values_s[i % len(dt_values_s)]. Splits are stratified by time step: within
  each time step's sequence of episodes, every block of 20 gets a seeded permutation into 14 train,
  3 validation, 3 test (exactly 70/15/15 per complete block). Generating more episodes only appends;
  earlier episodes keep both their content and their split, so validation/test trajectories never
  move into training.
* Each episode is one scenario with its own spacecraft, initial state and command schedule, and is
  simulated at one time step, so there are no shared trajectory variants across splits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gravity_ai.controllers.pid import TorquePID
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.dynamics_schema import (DATA_TYPE, EXTRA_TRUTH, FEATURE_NAMES, OUTPUT_NAMES, SCHEMA_VERSION,
                                               fingerprint)
from gravity_ai.simulation.mission import axial_acceleration_g
from gravity_ai.simulation.physics import G0, SpacecraftParams, SpacecraftState, equilibrium_omega, step
from gravity_ai.train.dynamics_config import DEFAULT_CONFIG_PATH, ConfigError, describe, load_config

SPLIT_NAMES = ('train', 'val', 'test')
SPLIT_BLOCK = 20
SPLIT_COUNTS = (14, 3, 3)
SCENARIO_CODES = {'mission_pid': 0, 'setpoint_steps': 1, 'random_torque': 2}
MAX_ATTEMPTS = 25
META_COLUMNS = ('episode_id', 'step_index', 'time_s', 'scenario_code', 'split_code')


def split_for_episode(episode_id: int, seed: int, n_dt: int) -> int:
    """Stratified by time step: episode i has dt index i % n_dt and position j = i // n_dt within
    that time step's sequence; every block of 20 positions gets exactly 14/3/3 train/val/test."""
    dt_index, j = int(episode_id) % n_dt, int(episode_id) // n_dt
    block, pos = divmod(j, SPLIT_BLOCK)
    rank = int(np.random.default_rng([seed, 1, dt_index, block]).permutation(SPLIT_BLOCK)[pos])
    return 0 if rank < SPLIT_COUNTS[0] else (1 if rank < SPLIT_COUNTS[0] + SPLIT_COUNTS[1] else 2)


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _piecewise(rng, steps, min_len, max_len, sampler):
    out = np.empty(steps)
    k = 0
    while k < steps:
        n = int(rng.integers(min_len, max_len + 1))
        out[k:k + n] = sampler()
        k += n
    return out


def _axial_profile(rng, steps):
    return _piecewise(rng, steps, 20, 60, lambda: 0.0 if rng.random() < 0.4 else rng.uniform(-0.8, 0.8) * G0)


def _sample_params(rng, ranges):
    return SpacecraftParams(
        radius_m=_loguniform(rng, *ranges['radius_m']),
        ring_inertia_kg_m2=_loguniform(rng, *ranges['ring_inertia_kg_m2']),
        hub_inertia_kg_m2=_loguniform(rng, *ranges['hub_inertia_kg_m2']),
        bearing_drag_nms=_loguniform(rng, *ranges['bearing_drag_nms']),
        motor_torque_limit_nm=_loguniform(rng, *ranges['motor_torque_limit_nm']),
        motor_power_limit_w=_loguniform(rng, *ranges['motor_power_limit_w']),
        motor_slew_nm_s=_loguniform(rng, *ranges['motor_slew_nm_s']),
    ).validate()


def _omega_for_g(radius_m, g):
    return equilibrium_omega(radius_m, g * G0, 0.0)


def _simulate_attempt(rng, scenario, p: SpacecraftParams, steps, dt):
    """Return per-step dict rows; commands depend only on the current simulated state."""
    limit = p.motor_torque_limit_nm
    if scenario == 'mission_pid':
        # App-like mission: PID tracks a target gravity while thrust accelerates, coasts and brakes.
        target_g = rng.uniform(0.3, 1.5)
        peak_g = rng.uniform(0.0, 1.2)
        duration = steps * dt
        axial = np.array([axial_acceleration_g(k * dt, duration, peak_g) * G0 for k in range(steps)])
        targets = np.full(steps, target_g)
        s = SpacecraftState(ring_omega_rad_s=_omega_for_g(p.radius_m, target_g) * rng.uniform(0.9, 1.1),
                            hub_omega_rad_s=rng.uniform(-0.005, 0.005), applied_torque_nm=0.0)
        info = {'target_g': target_g, 'peak_thrust_g': peak_g}
    elif scenario == 'setpoint_steps':
        # Spin-up / spin-down: PID follows piecewise gravity set-points under changing thrust.
        targets = _piecewise(rng, steps, 15, 40, lambda: rng.uniform(0.1, 1.6))
        axial = _axial_profile(rng, steps)
        s = SpacecraftState(ring_omega_rad_s=_omega_for_g(p.radius_m, targets[0]) * rng.uniform(0.7, 1.3),
                            hub_omega_rad_s=rng.uniform(-0.015, 0.015),
                            applied_torque_nm=rng.uniform(-0.5, 0.5) * limit)
        info = {}
    elif scenario == 'random_torque':
        # Open-loop excitation: random piecewise torque commands incl. saturation, kept inside a
        # 0.15-1.8 g band by flipping the command sign when the band is left.
        levels = _piecewise(rng, steps, 3, 15, lambda: 0.0 if rng.random() < 0.15 else rng.uniform(-1, 1) * limit)
        axial = _axial_profile(rng, steps)
        targets = None
        s = SpacecraftState(ring_omega_rad_s=_omega_for_g(p.radius_m, rng.uniform(0.3, 1.5)),
                            hub_omega_rad_s=rng.uniform(-0.015, 0.015),
                            applied_torque_nm=rng.uniform(-0.5, 0.5) * limit)
        info = {}
    else:
        raise ValueError(f'unknown scenario {scenario!r}')

    pid = TorquePID()
    rows = []
    for k in range(steps):
        if targets is not None:
            cmd, _ = pid.command(p, s, axial[k], targets[k] * G0, dt)
        else:
            g_now = p.radius_m * s.ring_omega_rad_s**2 / G0
            cmd = levels[k]
            if g_now > 1.8:
                cmd = -abs(cmd)
            elif g_now < 0.15:
                cmd = abs(cmd)
        nxt, t = step(p, s, cmd, axial[k], dt)
        rows.append({
            'step_index': k, 'time_s': s.time_s,
            'ring_omega_rad_s': s.ring_omega_rad_s, 'hub_omega_rad_s': s.hub_omega_rad_s,
            'applied_torque_nm': s.applied_torque_nm, 'requested_torque_nm': float(cmd),
            'axial_accel_m_s2': float(axial[k]),
            'ring_inertia_kg_m2': p.ring_inertia_kg_m2, 'hub_inertia_kg_m2': p.hub_inertia_kg_m2,
            'bearing_drag_nms': p.bearing_drag_nms, 'radius_m': p.radius_m, 'dt_s': dt,
            'motor_torque_limit_nm': p.motor_torque_limit_nm, 'motor_power_limit_w': p.motor_power_limit_w,
            'motor_slew_nm_s': p.motor_slew_nm_s,
            'next_ring_omega_rad_s': nxt.ring_omega_rad_s, 'next_hub_omega_rad_s': nxt.hub_omega_rad_s,
            'next_applied_torque_nm': nxt.applied_torque_nm,
            'crew_radial_m_s2': t['crew_gravity_radial_m_s2'], 'crew_tangential_m_s2': t['crew_gravity_tangential_m_s2'],
            'crew_axial_m_s2': t['crew_gravity_axial_m_s2'], '_crew_g': t['crew_gravity_g'],
        })
        s = nxt
    return rows, info


def _check_episode(rows, p, max_g):
    """Reasons an episode is physically invalid for training, or None if valid."""
    cols = FEATURE_NAMES + OUTPUT_NAMES + EXTRA_TRUTH
    if not all(np.isfinite(r[c]) for r in rows for c in cols):
        return 'non_finite'
    if min(min(r['ring_omega_rad_s'], r['next_ring_omega_rad_s']) for r in rows) <= 0:
        return 'ring_reversed_or_stopped'
    if max(r['_crew_g'] for r in rows) > max_g:
        return 'crew_gravity_above_limit'
    # Internal motor and drag torques only: total angular momentum must be conserved.
    l0 = p.ring_inertia_kg_m2 * rows[0]['ring_omega_rad_s'] + p.hub_inertia_kg_m2 * rows[0]['hub_omega_rad_s']
    for r in rows:
        l1 = p.ring_inertia_kg_m2 * r['next_ring_omega_rad_s'] + p.hub_inertia_kg_m2 * r['next_hub_omega_rad_s']
        if abs(l1 - l0) > 1e-9 * abs(l0):
            return 'angular_momentum_not_conserved'
    # Input/target alignment: the next state of step k must be the current state of step k+1.
    for a, b in zip(rows, rows[1:]):
        if (a['next_ring_omega_rad_s'] != b['ring_omega_rad_s'] or a['next_hub_omega_rad_s'] != b['hub_omega_rad_s']
                or a['next_applied_torque_nm'] != b['applied_torque_nm']):
            return 'misaligned_transition'
    return None


def simulate_episode(episode_id: int, cfg_data: dict):
    seed, steps = cfg_data['seed'], cfg_data['steps_per_episode']
    dts = cfg_data['dt_values_s']
    dt = float(dts[episode_id % len(dts)])
    names = list(cfg_data['scenario_mix'])
    weights = np.array([cfg_data['scenario_mix'][n] for n in names], dtype=float)
    rejected = {}
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([seed, 0, episode_id, attempt])
        scenario = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        p = _sample_params(rng, cfg_data['ranges'])
        rows, info = _simulate_attempt(rng, scenario, p, steps, dt)
        reason = _check_episode(rows, p, cfg_data['max_crew_gravity_g'])
        if reason is None:
            return rows, {'episode_id': episode_id, 'scenario': scenario, 'dt_s': dt, 'attempt': attempt,
                          'split': SPLIT_NAMES[split_for_episode(episode_id, seed, len(dts))], **info,
                          'radius_m': p.radius_m, 'ring_inertia_kg_m2': p.ring_inertia_kg_m2,
                          'hub_inertia_kg_m2': p.hub_inertia_kg_m2, 'bearing_drag_nms': p.bearing_drag_nms,
                          'motor_torque_limit_nm': p.motor_torque_limit_nm, 'motor_power_limit_w': p.motor_power_limit_w,
                          'motor_slew_nm_s': p.motor_slew_nm_s}, rejected
        rejected[reason] = rejected.get(reason, 0) + 1
    raise RuntimeError(f'episode {episode_id}: no valid scenario after {MAX_ATTEMPTS} attempts ({rejected})')


def generate_dataset(cfg_data: dict, output_dir: Path, overwrite: bool = False, log=print) -> dict:
    output_dir = Path(output_dir)
    target = output_dir / 'transitions.npz'
    if target.exists() and not overwrite:
        raise FileExistsError(f'{target} exists. Datasets are frozen benchmarks; choose a new output_dir '
                              'or pass --overwrite deliberately.')
    started = time.perf_counter()
    columns = {c: [] for c in META_COLUMNS + FEATURE_NAMES + OUTPUT_NAMES + EXTRA_TRUTH}
    episodes, rejected_total = [], {}
    n = cfg_data['episodes']
    for ep in range(n):
        rows, meta, rejected = simulate_episode(ep, cfg_data)
        for reason, count in rejected.items():
            rejected_total[reason] = rejected_total.get(reason, 0) + count
        split_code = SPLIT_NAMES.index(meta['split'])
        for r in rows:
            for c in FEATURE_NAMES + OUTPUT_NAMES + EXTRA_TRUTH:
                columns[c].append(r[c])
            columns['episode_id'].append(ep)
            columns['step_index'].append(r['step_index'])
            columns['time_s'].append(r['time_s'])
            columns['scenario_code'].append(SCENARIO_CODES[meta['scenario']])
            columns['split_code'].append(split_code)
        episodes.append(meta)
        if (ep + 1) % max(1, n // 10) == 0 or ep + 1 == n:
            log(f'  generated {ep + 1}/{n} episodes')
    arrays = {c: np.asarray(v, dtype=np.int32 if c in ('episode_id', 'step_index', 'scenario_code', 'split_code')
                            else np.float64) for c, v in columns.items()}
    fp = fingerprint(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)

    split_manifest = {
        'scheme': (f'stratified by dt: episode i -> dt index i % n_dt, position i // n_dt; per block of '
                   f'{SPLIT_BLOCK} positions a seeded permutation -> {SPLIT_COUNTS} train/val/test'),
        'seed': cfg_data['seed'], 'block_size': SPLIT_BLOCK, 'counts_per_block': dict(zip(SPLIT_NAMES, SPLIT_COUNTS)),
        'episodes': {str(m['episode_id']): m['split'] for m in episodes},
    }
    (output_dir / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2), encoding='utf-8')

    def count(key, value_fn):
        out = {}
        for m in episodes:
            k = value_fn(m)
            out.setdefault(k, {s: 0 for s in SPLIT_NAMES})[m['split']] += 1
        return out

    split_codes = arrays['split_code']
    report = {
        'data_type': DATA_TYPE, 'schema_version': SCHEMA_VERSION,
        'note': 'Synthetic spacecraft dynamics from the analytical simulator. Not measured data; not human physiology.',
        'observation_model': 'clean simulator truth (no sensor noise, delay or faults)',
        'labels_from': 'gravity_ai.simulation.physics.step',
        'dataset_fingerprint_sha256': fp,
        'files': {'transitions': str(target), 'split_manifest': str(output_dir / 'split_manifest.json')},
        'config': cfg_data,
        'episodes': n, 'steps_per_episode': cfg_data['steps_per_episode'], 'rows': int(len(split_codes)),
        'split_episode_counts': {s: sum(m['split'] == s for m in episodes) for s in SPLIT_NAMES},
        'split_row_counts': {s: int((split_codes == i).sum()) for i, s in enumerate(SPLIT_NAMES)},
        'episodes_by_dt_and_split': count('dt', lambda m: str(m['dt_s'])),
        'episodes_by_scenario_and_split': count('scenario', lambda m: m['scenario']),
        'rejected_attempts': rejected_total,
        'value_ranges': {c: [float(arrays[c].min()), float(arrays[c].max())] for c in FEATURE_NAMES + OUTPUT_NAMES},
        'crew_gravity_g_range': [float(min(np.hypot(np.hypot(arrays['crew_radial_m_s2'], arrays['crew_tangential_m_s2']),
                                                    arrays['crew_axial_m_s2'])) / G0),
                                 float(max(np.hypot(np.hypot(arrays['crew_radial_m_s2'], arrays['crew_tangential_m_s2']),
                                                    arrays['crew_axial_m_s2'])) / G0)],
        'episode_metadata': episodes,
        'elapsed_s': round(time.perf_counter() - started, 2),
    }
    (output_dir / 'generation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description='Generate simulated spacecraft-dynamics transitions (dynamics-v2).')
    ap.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='YAML config (data section); "none" = built-in defaults')
    ap.add_argument('--episodes', type=int)
    ap.add_argument('--steps', type=int, dest='steps_per_episode')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--output-dir')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing dataset in output-dir')
    args = ap.parse_args(argv)
    root = project_root()
    try:
        cfg, sources = load_config(None if args.config == 'none' else args.config, root=root, overrides={
            'data.episodes': args.episodes, 'data.steps_per_episode': args.steps_per_episode,
            'data.seed': args.seed, 'data.output_dir': args.output_dir})
    except ConfigError as e:
        raise SystemExit(f'Configuration error: {e}')
    print(describe(cfg, sources, sections=('data',)))
    out = Path(cfg['data']['output_dir'])
    out = out if out.is_absolute() else root / out
    try:
        report = generate_dataset(cfg['data'], out, overwrite=args.overwrite)
    except FileExistsError as e:
        raise SystemExit(str(e))
    summary = {k: report[k] for k in ('data_type', 'rows', 'episodes', 'split_episode_counts', 'split_row_counts',
                                      'episodes_by_dt_and_split', 'episodes_by_scenario_and_split', 'rejected_attempts',
                                      'crew_gravity_g_range', 'dataset_fingerprint_sha256', 'files', 'elapsed_s')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
