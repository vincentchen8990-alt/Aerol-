"""Fresh evaluation of a trained dynamics run, loaded from its saved checkpoint on disk.

  python -m gravity_ai.evaluate_dynamics --run outputs/runs/<run_id>     (or --run latest)

Ground truth is the analytical simulator (synthetic). Reports one-step and multi-step (rollout)
errors on the frozen held-out TEST episodes, against a persistence baseline and a ridge baseline,
and writes evaluation.json plus plots to <run_dir>/evaluation/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.dynamics_predictor import PARAM_KEYS, DynamicsPredictor, persistence_step, rollout
from gravity_ai.models.dynamics_schema import (FEATURE_NAMES, OUTPUT_NAMES, OUTPUTS, decode_outputs, encode_features,
                                               encode_targets, rad_s_to_rpm, truth_outputs)
from gravity_ai.simulation.physics import G0
from gravity_ai.train.dynamics import LATEST_POINTER, SPLIT_NAMES, load_dataset
from gravity_ai.train.dynamics_config import load_config

UNITS = {name: unit for name, unit, _ in OUTPUTS}
ACCEL = slice(2, 5)


class RidgeBaseline:
    """Linear baseline on the same standardised features, predicting residual targets."""

    def __init__(self, x_mean, x_std, alpha=1.0):
        self.x_mean, self.x_std, self.alpha = x_mean, x_std, alpha

    def fit(self, arrays, mask):
        from sklearn.linear_model import Ridge
        x = (encode_features(arrays)[mask] - self.x_mean) / self.x_std
        self.model = Ridge(alpha=self.alpha).fit(x, encode_targets({k: v[mask] for k, v in arrays.items()}, 'residual'))
        return self

    def step_fn(self, arrays):
        x = (encode_features(arrays) - self.x_mean) / self.x_std
        return decode_outputs(self.model.predict(x), arrays, 'residual')


def _subset(arrays, mask):
    return {k: v[mask] for k, v in arrays.items()}


def _err_stats(err):
    err = np.asarray(err, dtype=np.float64)
    return {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err**2)))}


def one_step_metrics(pred, truth) -> dict:
    out = {}
    for i, name in enumerate(OUTPUT_NAMES):
        s = _err_stats(pred[:, i] - truth[:, i])
        s['unit'] = UNITS[name]
        if i < 2:
            s['mae_rpm'], s['rmse_rpm'] = float(rad_s_to_rpm(s['mae'])), float(rad_s_to_rpm(s['rmse']))
        else:
            s['mae_g'], s['rmse_g'] = s['mae'] / G0, s['rmse'] / G0
        out[name] = s
    mag_p = np.linalg.norm(pred[:, ACCEL], axis=1)
    mag_t = np.linalg.norm(truth[:, ACCEL], axis=1)
    s = _err_stats(mag_p - mag_t)
    out['crew_gravity_magnitude'] = {**s, 'unit': 'm/s^2', 'mae_g': s['mae'] / G0, 'rmse_g': s['rmse'] / G0,
                                     'definition': '| |a_pred| - |a_true| | (total apparent gravity)'}
    vec = np.linalg.norm(pred[:, ACCEL] - truth[:, ACCEL], axis=1)
    out['crew_gravity_vector_error'] = {'mean': float(vec.mean()), 'rmse': float(np.sqrt(np.mean(vec**2))),
                                        'unit': 'm/s^2', 'mean_g': float(vec.mean() / G0),
                                        'definition': '|a_pred - a_true| (3-component vector error)'}
    return out


def _episode_table(arrays, mask):
    """Per test episode arrays ordered by step: (episode ids, dict of (E, steps) arrays)."""
    eps = np.unique(arrays['episode_id'][mask])
    rows = {}
    for e in eps:
        idx = np.where(arrays['episode_id'] == e)[0]
        idx = idx[np.argsort(arrays['step_index'][idx])]
        rows[int(e)] = idx
    return eps, rows


def evaluate_rollouts(arrays, test_mask, horizons, stride, models: dict) -> dict:
    """Recursive predictions from start points inside test episodes; nothing from the future
    trajectory is fed back except the known command and thrust schedules."""
    eps, ep_rows = _episode_table(arrays, test_mask)
    truth_all = truth_outputs(arrays)
    results, curves = {}, {}
    dts = sorted({float(arrays['dt_s'][ep_rows[int(e)][0]]) for e in eps})
    for dt_s in dts:
        dt_eps = [int(e) for e in eps if float(arrays['dt_s'][ep_rows[int(e)][0]]) == dt_s]
        for horizon in horizons:
            n = int(round(horizon / dt_s))
            if n < 1 or abs(n * dt_s - horizon) > 1e-9:
                continue
            starts = []
            for e in dt_eps:
                idx = ep_rows[e]
                if len(idx) < n:
                    continue
                starts += [idx[k0:k0 + n] for k0 in range(0, len(idx) - n + 1, stride)]
            if not starts:
                continue
            win = np.stack(starts)  # (B, n) row indices
            first = win[:, 0]
            params = {k: arrays[k][first] for k in PARAM_KEYS}
            truth = truth_all[win]  # (B, n, 5)
            key = f'dt={dt_s:g}s,horizon={horizon:g}s'
            results[key] = {'dt_s': dt_s, 'horizon_s': horizon, 'steps': n, 'rollouts': int(len(win)),
                            'episodes': len(dt_eps), 'models': {}}
            for name, step_fn in models.items():
                pred, applied = rollout(step_fn, arrays['ring_omega_rad_s'][first], arrays['hub_omega_rad_s'][first],
                                        arrays['applied_torque_nm'][first], arrays['requested_torque_nm'][win],
                                        arrays['axial_accel_m_s2'][win], params, dt_s)
                end_p, end_t = pred[:, -1], truth[:, -1]
                m = {
                    'ring_omega_rad_s': _err_stats(end_p[:, 0] - end_t[:, 0]),
                    'hub_omega_rad_s': _err_stats(end_p[:, 1] - end_t[:, 1]),
                    'crew_radial_m_s2': _err_stats(end_p[:, 2] - end_t[:, 2]),
                    'crew_tangential_m_s2': _err_stats(end_p[:, 3] - end_t[:, 3]),
                    'crew_axial_m_s2': _err_stats(end_p[:, 4] - end_t[:, 4]),
                    'crew_gravity_magnitude_m_s2': _err_stats(np.linalg.norm(end_p[:, ACCEL], axis=1)
                                                              - np.linalg.norm(end_t[:, ACCEL], axis=1)),
                }
                m['ring_omega_rpm'] = {k: float(rad_s_to_rpm(v)) for k, v in m['ring_omega_rad_s'].items()}
                m['crew_gravity_magnitude_g'] = {k: v / G0 for k, v in m['crew_gravity_magnitude_m_s2'].items()}
                if name == 'learned_model':
                    m['actuator_torque_propagation_nm'] = _err_stats(applied[:, -1] - arrays['next_applied_torque_nm'][win[:, -1]])
                results[key]['models'][name] = m
                if horizon == max(horizons):
                    curves.setdefault(f'{dt_s:g}', {})[name] = {
                        'time_s': ((np.arange(n) + 1) * dt_s).tolist(),
                        'ring_rpm_mae': rad_s_to_rpm(np.mean(np.abs(pred[:, :, 0] - truth[:, :, 0]), axis=0)).tolist(),
                        'gravity_g_mae': (np.mean(np.abs(np.linalg.norm(pred[:, :, ACCEL], axis=2)
                                                         - np.linalg.norm(truth[:, :, ACCEL], axis=2)), axis=0) / G0).tolist(),
                    }
    return {'results': results, 'error_growth': curves}


def _plots(out_dir, manifest, arrays, test_mask, pred_test, truth_test, curves, example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = {}
    h = manifest['history']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in h], [r['train_loss'] for r in h], label='training')
    ax.plot([r['epoch'] for r in h], [r['validation_loss'] for r in h], label='validation')
    ax.axvline(manifest['best_epoch'], color='grey', ls=':', label=f'best epoch {manifest["best_epoch"]}')
    ax.set(yscale='log', xlabel='epoch', ylabel='MSE (standardised targets)', title=f'Loss - {manifest["run_id"]}')
    ax.legend()
    fig.tight_layout()
    files['training_loss'] = str(out_dir / 'training_loss.png')
    fig.savefig(files['training_loss'], dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(truth_test), size=min(6000, len(truth_test)), replace=False)
    cur = np.stack([arrays['ring_omega_rad_s'][test_mask], arrays['hub_omega_rad_s'][test_mask]], 1)[sel]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for i, label in enumerate(('ring', 'hub')):
        t, p = truth_test[sel, i], pred_test[sel, i]
        ax = axes[0, i]
        ax.scatter(rad_s_to_rpm(t), rad_s_to_rpm(p), s=3, alpha=.4)
        lim = [rad_s_to_rpm(min(t.min(), p.min())), rad_s_to_rpm(max(t.max(), p.max()))]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(xlabel=f'simulator next {label} speed (rpm)', ylabel='predicted (rpm)', title=f'Next {label} speed')
        ax = axes[1, i]
        dt_, dp = rad_s_to_rpm(t - cur[:, i]), rad_s_to_rpm(p - cur[:, i])
        ax.scatter(dt_, dp, s=3, alpha=.4)
        lim = [min(dt_.min(), dp.min()), max(dt_.max(), dp.max())]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(xlabel=f'simulator per-step change (rpm)', ylabel='predicted change (rpm)', title=f'{label} speed change per step')
    fig.suptitle('Held-out test episodes, one-step predictions')
    fig.tight_layout()
    files['pred_vs_actual_speeds'] = str(out_dir / 'pred_vs_actual_speeds.png')
    fig.savefig(files['pred_vs_actual_speeds'], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    comps = [(2, 'radial'), (3, 'tangential'), (4, 'axial')]
    for ax, (i, label) in zip(axes.flat, comps):
        t, p = truth_test[sel, i] / G0, pred_test[sel, i] / G0
        ax.scatter(t, p, s=3, alpha=.4)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(xlabel=f'simulator {label} (g)', ylabel='predicted (g)', title=f'Crew acceleration, {label}')
    t = np.linalg.norm(truth_test[sel][:, ACCEL], axis=1) / G0
    p = np.linalg.norm(pred_test[sel][:, ACCEL], axis=1) / G0
    ax = axes[1, 1]
    ax.scatter(t, p, s=3, alpha=.4)
    ax.plot([t.min(), t.max()], [t.min(), t.max()], 'k--', lw=1)
    ax.set(xlabel='simulator |a| (g)', ylabel='predicted |a| (g)', title='Total apparent gravity magnitude')
    fig.suptitle('Held-out test episodes, one-step predictions')
    fig.tight_layout()
    files['pred_vs_actual_accel'] = str(out_dir / 'pred_vs_actual_accel.png')
    fig.savefig(files['pred_vs_actual_accel'], dpi=120)
    plt.close(fig)

    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        styles = {'learned_model': '-', 'ridge_baseline': '--', 'persistence': ':'}
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for c, (dt_key, by_model) in zip(colors, sorted(curves.items(), key=lambda kv: float(kv[0]))):
            for name, cv in by_model.items():
                axes[0].plot(cv['time_s'], cv['ring_rpm_mae'], styles[name], color=c, label=f'{name}, dt={dt_key}s')
                axes[1].plot(cv['time_s'], cv['gravity_g_mae'], styles[name], color=c, label=f'{name}, dt={dt_key}s')
        axes[0].set(yscale='log', xlabel='rollout time (s)', ylabel='MAE ring speed (rpm)', title='Ring speed error growth')
        axes[1].set(yscale='log', xlabel='rollout time (s)', ylabel='MAE |gravity| (g)', title='Apparent gravity error growth')
        axes[1].legend(fontsize=7, ncol=2)
        fig.suptitle('Recursive rollouts on held-out test episodes (known commands, no future states)')
        fig.tight_layout()
        files['rollout_error_growth'] = str(out_dir / 'rollout_error_growth.png')
        fig.savefig(files['rollout_error_growth'], dpi=120)
        plt.close(fig)

    if example:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tt = example['time_s']
        for name, series in example['series'].items():
            axes[0].plot(tt, series['ring_rpm'], label=name)
            axes[1].plot(tt, series['gravity_g'], label=name)
        axes[0].set(xlabel='time (s)', ylabel='ring speed (rpm)', title='Ring speed')
        axes[1].set(xlabel='time (s)', ylabel='|gravity| (g)', title='Apparent gravity')
        axes[1].legend()
        fig.suptitle(f'Example test episode {example["episode_id"]} ({example["scenario"]}, dt={example["dt_s"]:g}s): '
                     'open-loop rollout of the full episode')
        fig.tight_layout()
        files['rollout_example'] = str(out_dir / 'rollout_example.png')
        fig.savefig(files['rollout_example'], dpi=120)
        plt.close(fig)
    return files


def _example_rollout(arrays, test_mask, models, report, prefer_dt=2.0):
    eps, ep_rows = _episode_table(arrays, test_mask)
    scen = {m['episode_id']: m['scenario'] for m in report['episode_metadata']}
    cands = [int(e) for e in eps if float(arrays['dt_s'][ep_rows[int(e)][0]]) == prefer_dt and scen[int(e)] == 'mission_pid']
    if not cands:
        cands = [int(e) for e in eps]
    e = cands[0]
    idx = ep_rows[e]
    truth = truth_outputs(arrays)[idx]
    first = idx[:1]
    series = {'simulator': {'ring_rpm': rad_s_to_rpm(truth[:, 0]).tolist(),
                            'gravity_g': (np.linalg.norm(truth[:, ACCEL], axis=1) / G0).tolist()}}
    for name, fn in models.items():
        pred, _ = rollout(fn, arrays['ring_omega_rad_s'][first], arrays['hub_omega_rad_s'][first],
                          arrays['applied_torque_nm'][first], arrays['requested_torque_nm'][idx][None],
                          arrays['axial_accel_m_s2'][idx][None], {k: arrays[k][first] for k in PARAM_KEYS},
                          float(arrays['dt_s'][idx[0]]))
        series[name] = {'ring_rpm': rad_s_to_rpm(pred[0, :, 0]).tolist(),
                        'gravity_g': (np.linalg.norm(pred[0][:, ACCEL], axis=1) / G0).tolist()}
    return {'episode_id': e, 'scenario': scen[e], 'dt_s': float(arrays['dt_s'][idx[0]]),
            'time_s': (arrays['time_s'][idx] + arrays['dt_s'][idx]).tolist(), 'series': series}


def evaluate_run(run_dir: Path, horizons=None, stride=None, log=print) -> dict:
    started = time.perf_counter()
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed':
        raise RuntimeError(f'run {run_dir} is not completed (status={manifest.get("status")})')
    eval_cfg = manifest['config']['evaluation']
    horizons = horizons or eval_cfg['rollout_horizons_s']
    stride = stride or eval_cfg['rollout_start_stride_steps']

    predictor = DynamicsPredictor.load(run_dir / 'checkpoint_best.pt')  # fresh load from disk
    ds = load_dataset(Path(manifest['dataset']['dir']))
    if ds['fingerprint'] != predictor.checkpoint['dataset_fingerprint']:
        raise ValueError('dataset on disk differs from the one this checkpoint was trained on')
    if ds['split_manifest_sha256'] != predictor.checkpoint['split_manifest_sha256']:
        raise ValueError('split manifest differs from the one used in training')
    arrays, masks = ds['arrays'], ds['masks']
    feats = {k: arrays[k] for k in FEATURE_NAMES}
    ridge = RidgeBaseline(predictor.x_mean, predictor.x_std).fit(arrays, masks[0])
    step_models = {'learned_model': predictor.step_fn, 'persistence': persistence_step, 'ridge_baseline': ridge.step_fn}

    one_step, per_dt = {}, {}
    pred_test = truth_test = None
    for s_i in (1, 2):
        split = SPLIT_NAMES[s_i]
        sub = _subset(feats, masks[s_i])
        truth = truth_outputs(_subset(arrays, masks[s_i]))
        env = predictor.envelope_report(predictor.validate(sub))
        one_step[split] = {'rows': int(masks[s_i].sum()),
                           'episodes': int(len(np.unique(arrays['episode_id'][masks[s_i]]))),
                           'rows_outside_training_envelope': env['rows_outside'], 'models': {}}
        for name, fn in step_models.items():
            pred = fn(sub)
            one_step[split]['models'][name] = one_step_metrics(pred, truth)
            if split == 'test':
                if name == 'learned_model':
                    pred_test, truth_test = pred, truth
                for dt_v in sorted(set(sub['dt_s'].tolist())):
                    m = sub['dt_s'] == dt_v
                    per_dt.setdefault(f'{dt_v:g}', {'rows': int(m.sum()), 'models': {}})['models'][name] = \
                        one_step_metrics(pred[m], truth[m])

    log('Running rollouts...')
    ro = evaluate_rollouts(arrays, masks[2], horizons, stride, step_models)
    example = _example_rollout(arrays, masks[2], step_models, ds['report'])
    out_dir = run_dir / 'evaluation'
    out_dir.mkdir(exist_ok=True)
    plots = _plots(out_dir, manifest, arrays, masks[2], pred_test, truth_test, ro['error_growth'], example)

    result = {
        'run_id': manifest['run_id'], 'evaluated': dt.datetime.now().isoformat(timespec='seconds'),
        'data_type': manifest['data_type'],
        'ground_truth': 'analytical simulator gravity_ai.simulation.physics.step (synthetic, clean, no sensor noise)',
        'checkpoint': str(predictor.path), 'checkpoint_epoch': predictor.checkpoint['epoch'],
        'loaded_from_disk': True, 'target_mode': predictor.target_mode,
        'dataset_fingerprint_sha256': ds['fingerprint'], 'split_manifest_sha256': ds['split_manifest_sha256'],
        'baselines': {'persistence': 'next speeds = current speeds; radial = r*omega^2 of current speed; '
                                     'tangential = 0; axial = -known thrust acceleration',
                      'ridge_baseline': 'ridge regression (alpha=1) on the same standardised features, '
                                        'residual targets, fitted on training episodes only'},
        'one_step': one_step, 'one_step_test_by_dt_s': per_dt,
        'rollout_assumptions': ['initial ring/hub speed and applied torque taken from the simulator at the start point',
                                'known planned torque commands and axial thrust schedule supplied for every step '
                                '(recorded commands replayed open-loop; closed-loop feedback is not simulated)',
                                'fixed spacecraft parameters and dt supplied',
                                'velocities fed back from the previous prediction; no future simulator state used',
                                'applied (actuator) torque advanced by the explicit motor model (clamp, slew, power limit)'],
        'rollout_start_stride_steps': stride, 'rollouts': ro['results'], 'rollout_error_growth': ro['error_growth'],
        'rollout_example': {k: example[k] for k in ('episode_id', 'scenario', 'dt_s')},
        'plots': plots, 'elapsed_s': round(time.perf_counter() - started, 2),
    }
    (out_dir / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    log(f'Wrote {out_dir / "evaluation.json"}')
    return result


def print_summary(r, log=print):
    log(f'\nRun {r["run_id"]} - checkpoint {r["checkpoint"]} (epoch {r["checkpoint_epoch"]}, loaded from disk)')
    test = r['one_step']['test']
    log(f'One-step TEST ({test["rows"]} rows, {test["episodes"]} episodes) MAE [learned | persistence | ridge]:')
    names = OUTPUT_NAMES + ('crew_gravity_magnitude',)
    for n in names:
        vals = [test['models'][m][n]['mae'] for m in ('learned_model', 'persistence', 'ridge_baseline')]
        log(f'  {n:<24} {vals[0]:>11.4g} | {vals[1]:>11.4g} | {vals[2]:>11.4g}  {test["models"]["learned_model"][n]["unit"]}')
    log('Rollout end-of-horizon MAE, ring speed (rpm) and |gravity| (g) [learned | persistence | ridge]:')
    for key, v in r['rollouts'].items():
        ms = v['models']
        rpm = [ms[m]['ring_omega_rpm']['mae'] for m in ('learned_model', 'persistence', 'ridge_baseline')]
        gg = [ms[m]['crew_gravity_magnitude_g']['mae'] for m in ('learned_model', 'persistence', 'ridge_baseline')]
        log(f'  {key:<24} n={v["rollouts"]:<4} rpm {rpm[0]:.4g} | {rpm[1]:.4g} | {rpm[2]:.4g}   '
            f'g {gg[0]:.4g} | {gg[1]:.4g} | {gg[2]:.4g}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Evaluate a dynamics run from its saved checkpoint.')
    ap.add_argument('--run', default='latest', help='run directory, run id under outputs/runs, or "latest"')
    ap.add_argument('--config', default=None, help='optional YAML to override evaluation settings')
    args = ap.parse_args(argv)
    root = project_root()
    runs_root = root / 'outputs' / 'runs'
    if args.run == 'latest':
        pointer = runs_root / LATEST_POINTER
        if not pointer.is_file():
            raise SystemExit('No completed run recorded; train first: python -m gravity_ai.train.dynamics')
        run_dir = Path(json.loads(pointer.read_text(encoding='utf-8'))['run_dir'])
    else:
        run_dir = Path(args.run)
        run_dir = run_dir if run_dir.is_absolute() else (root / run_dir if (root / run_dir).exists() else runs_root / run_dir)
    horizons = stride = None
    if args.config:
        cfg, _ = load_config(args.config, root=root)
        horizons, stride = cfg['evaluation']['rollout_horizons_s'], cfg['evaluation']['rollout_start_stride_steps']
    print_summary(evaluate_run(run_dir, horizons, stride))


if __name__ == '__main__':
    main()
