"""Train the spacecraft dynamics predictor on SIMULATED transitions (schema dynamics-v2).

Not a human-physiology model. Splits come from the dataset's frozen split manifest; normalisation
is fitted on training episodes only. Each run writes a self-contained directory:

  outputs/runs/<run_id>/checkpoint_best.pt      best-validation weights + preprocessing + schema
  outputs/runs/<run_id>/training_state_last.pt  full state for --resume (model, optimizer, RNG, history)
  outputs/runs/<run_id>/run_manifest.json       config, dataset/split ids, seeds, versions, device, history
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.dynamics import DynamicsNet
from gravity_ai.models.dynamics_predictor import CHECKPOINT_FORMAT
from gravity_ai.models.dynamics_schema import (DATA_TYPE, FEATURE_NAMES, OUTPUT_NAMES, SCHEMA_VERSION, encode_features,
                                               encode_targets, fingerprint, schema_dict)
from gravity_ai.train.dynamics_config import DEFAULT_CONFIG_PATH, ConfigError, describe, load_config

SPLIT_NAMES = ('train', 'val', 'test')
LATEST_POINTER = 'latest_dynamics.json'


def _resolve(root: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def load_dataset(dataset_dir: Path) -> dict:
    """Load transitions, verify content fingerprint and the frozen split manifest."""
    dataset_dir = Path(dataset_dir)
    npz = dataset_dir / 'transitions.npz'
    if not npz.is_file():
        raise FileNotFoundError(f'{npz} not found. Generate it first: python -m gravity_ai.simulation.generate')
    with np.load(npz) as f:
        arrays = {k: f[k] for k in f.files}
    report = json.loads((dataset_dir / 'generation_report.json').read_text(encoding='utf-8'))
    fp = fingerprint(arrays)
    if fp != report['dataset_fingerprint_sha256']:
        raise ValueError(f'{npz} content does not match its generation report fingerprint')
    if report.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'dataset schema {report.get("schema_version")} != {SCHEMA_VERSION}')
    manifest_text = (dataset_dir / 'split_manifest.json').read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    episode_split = {int(k): SPLIT_NAMES.index(v) for k, v in manifest['episodes'].items()}
    expected = np.array([episode_split[int(e)] for e in arrays['episode_id']])
    if not np.array_equal(expected, arrays['split_code']):
        raise ValueError('row split codes disagree with split_manifest.json')
    sets = [set(arrays['episode_id'][arrays['split_code'] == i].tolist()) for i in range(3)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('an episode appears in more than one split')
    return {'arrays': arrays, 'report': report, 'fingerprint': fp, 'dir': dataset_dir,
            'split_manifest_sha256': hashlib.sha256(manifest_text.encode()).hexdigest(),
            'masks': [arrays['split_code'] == i for i in range(3)]}


def fit_standardizer(x: np.ndarray):
    """Mean/std on the given (training) rows; zero-variance columns get std 1."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def resolve_device(requested: str) -> torch.device:
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ConfigError('train.device: cuda requested but torch.cuda.is_available() is False '
                          f'(installed torch {torch.__version__})')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def environment_info(device: torch.device) -> dict:
    import sklearn
    info = {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'scikit_learn': sklearn.__version__, 'platform': platform.platform(),
            'device': str(device), 'cuda_available': torch.cuda.is_available(),
            'torch_cuda_build': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            'cpu': platform.processor()}
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
    return info


def _write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def train(cfg: dict, sources: dict, root: Path, run_id: str | None = None, resume: Path | None = None, log=print) -> dict:
    t_cfg, m_cfg = cfg['train'], cfg['model']
    device = resolve_device(t_cfg['device'])
    if t_cfg['num_threads'] > 0:
        torch.set_num_threads(t_cfg['num_threads'])

    resume_state = None
    if resume is not None:
        resume = Path(resume)
        state_path = resume / 'training_state_last.pt' if resume.is_dir() else resume
        resume_state = torch.load(state_path, map_location='cpu', weights_only=True)
        run_dir = state_path.parent
        run_id = resume_state['run_id']
        prev_manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
        for key in ('seed', 'batch_size', 'learning_rate', 'weight_decay', 'target_mode'):
            if prev_manifest['config']['train'][key] != t_cfg[key]:
                raise ConfigError(f'--resume: train.{key} differs from the original run '
                                  f'({prev_manifest["config"]["train"][key]} vs {t_cfg[key]})')
        if prev_manifest['config']['model'] != m_cfg:
            raise ConfigError('--resume: model settings differ from the original run')
        data_dir = Path(prev_manifest['dataset']['dir'])
    else:
        run_id = run_id or dt.datetime.now().strftime('dyn-%Y%m%d-%H%M%S')
        run_dir = _resolve(root, t_cfg['output_root']) / run_id
        if run_dir.exists():
            raise FileExistsError(f'{run_dir} already exists; choose another --run-id or use --resume')
        data_dir = _resolve(root, t_cfg['dataset_dir'])

    ds = load_dataset(data_dir)
    if resume_state is not None and resume_state['dataset_fingerprint'] != ds['fingerprint']:
        raise ValueError('--resume: dataset content changed since the original run')
    a, masks = ds['arrays'], ds['masks']
    if not all(m.any() for m in masks):
        raise ValueError('every split needs at least one episode')
    x_all = encode_features(a)
    y_all = encode_targets(a, t_cfg['target_mode'])
    if resume_state is None:
        x_mean, x_std = fit_standardizer(x_all[masks[0]])
        y_mean, y_std = fit_standardizer(y_all[masks[0]])
    else:  # never refit: restored exactly
        x_mean, x_std = (np.asarray(resume_state[k], dtype=np.float64) for k in ('input_mean', 'input_std'))
        y_mean, y_std = (np.asarray(resume_state[k], dtype=np.float64) for k in ('target_mean', 'target_std'))
    xs = ((x_all - x_mean) / x_std).astype(np.float32)
    ys = ((y_all - y_mean) / y_std).astype(np.float32)
    train_x, train_y = torch.from_numpy(xs[masks[0]]).to(device), torch.from_numpy(ys[masks[0]]).to(device)
    val_x, val_y = torch.from_numpy(xs[masks[1]]).to(device), torch.from_numpy(ys[masks[1]]).to(device)
    envelope = {k: [float(a[k][masks[0]].min()), float(a[k][masks[0]].max())] for k in FEATURE_NAMES}
    dt_values = sorted({float(v) for v in a['dt_s'][masks[0]]})

    torch.manual_seed(t_cfg['seed'])
    np.random.seed(t_cfg['seed'])
    model = DynamicsNet(len(FEATURE_NAMES), len(OUTPUT_NAMES), m_cfg['hidden_width'], m_cfg['hidden_layers']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg['learning_rate'], weight_decay=t_cfg['weight_decay'])
    gen = torch.Generator().manual_seed(t_cfg['seed'])
    history, best, best_epoch, bad, start_epoch, prior_elapsed = [], float('inf'), 0, 0, 0, 0.0
    resume_events = []
    if resume_state is not None:
        model.load_state_dict(resume_state['model_state'])
        opt.load_state_dict(resume_state['optimizer_state'])
        gen.set_state(resume_state['generator_state'])
        history, best, best_epoch, bad = (resume_state['history'], resume_state['best_val'],
                                          resume_state['best_epoch'], resume_state['bad_epochs'])
        start_epoch, prior_elapsed = resume_state['epoch'], resume_state['elapsed_s']
        resume_events = resume_state.get('resume_events', []) + [{
            'resumed_at': dt.datetime.now().isoformat(timespec='seconds'), 'from_epoch': start_epoch,
            'restored': ['model', 'optimizer', 'shuffle_generator', 'normalisation', 'history', 'early_stopping_counters']}]
        start_mode = f'resumed from {state_path} after epoch {start_epoch}'
    else:
        start_mode = 'from_scratch'
    run_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    split_eps = {s: int(len(np.unique(a['episode_id'][m]))) for s, m in zip(SPLIT_NAMES, masks)}
    split_rows = {s: int(m.sum()) for s, m in zip(SPLIT_NAMES, masks)}
    env = environment_info(device)
    manifest = {
        'run_id': run_id, 'status': 'running', 'data_type': DATA_TYPE,
        'note': 'Learned approximation of the analytical spacecraft simulator. Not a human or flight-qualified model.',
        'start_mode': start_mode, 'resume_events': resume_events,
        'created': (prev_manifest['created'] if resume_state is not None
                    else dt.datetime.now().isoformat(timespec='seconds')),
        'config': cfg, 'config_sources': sources, 'schema': schema_dict(t_cfg['target_mode']),
        'architecture': {'class': 'gravity_ai.models.dynamics.DynamicsNet', 'inputs': len(FEATURE_NAMES),
                         'outputs': len(OUTPUT_NAMES), **m_cfg, 'parameters': n_params},
        'dataset': {'dir': str(ds['dir']), 'fingerprint_sha256': ds['fingerprint'],
                    'split_manifest_sha256': ds['split_manifest_sha256'],
                    'generation_seed': ds['report']['config']['seed'], 'split_seed': ds['report']['config']['seed'],
                    'split_episode_counts': split_eps, 'split_row_counts': split_rows,
                    'dt_values_trained': dt_values},
        'seeds': {'train': t_cfg['seed'], 'data_generation': ds['report']['config']['seed']},
        'environment': env,
        'feature_envelope_train': envelope,
    }
    _write_json(run_dir / 'run_manifest.json', manifest)

    log(f'Run {run_id}: {start_mode}')
    log(f'Device: {device} (cuda available: {env["cuda_available"]}, torch {env["torch"]}, threads {env["torch_threads"]})')
    log(f'Dataset {ds["dir"]} fingerprint {ds["fingerprint"][:12]}...; episodes {split_eps}; rows {split_rows}')
    log(f'Model DynamicsNet {len(FEATURE_NAMES)}->{m_cfg["hidden_layers"]}x{m_cfg["hidden_width"]}->{len(OUTPUT_NAMES)} '
        f'({n_params} parameters), target_mode={t_cfg["target_mode"]}')

    def checkpoint_payload(state, epoch, val):
        return {'format': CHECKPOINT_FORMAT, 'run_id': run_id, 'epoch': epoch, 'val_loss': val,
                'schema': schema_dict(t_cfg['target_mode']), 'model_config': dict(m_cfg), 'model_state': state,
                'input_mean': x_mean.tolist(), 'input_std': x_std.tolist(),
                'target_mean': y_mean.tolist(), 'target_std': y_std.tolist(),
                'feature_envelope': envelope, 'dt_values_trained': dt_values,
                'dataset_fingerprint': ds['fingerprint'], 'split_manifest_sha256': ds['split_manifest_sha256']}

    started = time.perf_counter()
    stop_reason = 'epoch_limit'
    epochs = t_cfg['epochs']
    patience = t_cfg['early_stopping_patience']
    if start_epoch >= epochs:
        stop_reason = 'already_at_epoch_limit'
    elif bad >= patience:
        stop_reason = 'early_stopping_already_triggered'
        start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(train_x), generator=gen).to(device)
        total, count = 0.0, 0
        for idx in perm.split(t_cfg['batch_size']):
            pred = model(train_x[idx])
            loss = nn.functional.mse_loss(pred, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(val_x), val_y))
        improved = val < best
        if improved:
            best, best_epoch, bad = val, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(checkpoint_payload(best_state, epoch + 1, val), run_dir / 'checkpoint_best.pt')
        else:
            bad += 1
        history.append({'epoch': epoch + 1, 'train_loss': total / count, 'validation_loss': val,
                        'seconds': round(time.perf_counter() - t0, 3), 'improved': improved})
        elapsed = prior_elapsed + time.perf_counter() - started
        torch.save({'run_id': run_id, 'epoch': epoch + 1, 'model_state': model.state_dict(),
                    'optimizer_state': opt.state_dict(), 'generator_state': gen.get_state(),
                    'history': history, 'best_val': best, 'best_epoch': best_epoch, 'bad_epochs': bad,
                    'input_mean': x_mean.tolist(), 'input_std': x_std.tolist(),
                    'target_mean': y_mean.tolist(), 'target_std': y_std.tolist(),
                    'dataset_fingerprint': ds['fingerprint'], 'elapsed_s': elapsed, 'resume_events': resume_events},
                   run_dir / 'training_state_last.pt')
        log(f'epoch {epoch + 1:>3}/{epochs}  train {total / count:.6f}  val {val:.6f}  '
            f'{"*best*" if improved else f"no improvement {bad}/{patience}"}  ({history[-1]["seconds"]:.1f}s)')
        if bad >= patience:
            stop_reason = 'early_stopping'
            break

    if not (run_dir / 'checkpoint_best.pt').is_file():
        raise RuntimeError('training finished without a saved checkpoint')
    elapsed = prior_elapsed + time.perf_counter() - started
    manifest.update({'status': 'completed', 'completed': dt.datetime.now().isoformat(timespec='seconds'),
                     'stop_reason': stop_reason, 'epochs_completed': len(history), 'best_epoch': best_epoch,
                     'best_validation_mse_standardized': best, 'training_elapsed_s': round(elapsed, 2),
                     'history': history,
                     'artifacts': {'best_checkpoint': str(run_dir / 'checkpoint_best.pt'),
                                   'training_state': str(run_dir / 'training_state_last.pt'),
                                   'manifest': str(run_dir / 'run_manifest.json')}})
    _write_json(run_dir / 'run_manifest.json', manifest)
    _write_json(_resolve(root, t_cfg['output_root']) / LATEST_POINTER,
                {'run_id': run_id, 'run_dir': str(run_dir), 'completed': manifest['completed']})
    log(f'Finished: {stop_reason}; best epoch {best_epoch} (val {best:.6f}); {len(history)} epochs; {elapsed:.1f}s')
    log(f'Checkpoint: {run_dir / "checkpoint_best.pt"}')
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description='Train the spacecraft dynamics predictor (simulated data).')
    ap.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='YAML config; "none" = built-in defaults')
    ap.add_argument('--dataset-dir')
    ap.add_argument('--output-root')
    ap.add_argument('--run-id')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--learning-rate', type=float)
    ap.add_argument('--weight-decay', type=float)
    ap.add_argument('--patience', type=int, dest='early_stopping_patience')
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    ap.add_argument('--num-threads', type=int)
    ap.add_argument('--target-mode', choices=('residual', 'absolute'))
    ap.add_argument('--resume', help='run directory (or training_state_last.pt) to continue')
    args = ap.parse_args(argv)
    root = project_root()
    overrides = {f'train.{k}': getattr(args, k) for k in (
        'dataset_dir', 'output_root', 'seed', 'epochs', 'batch_size', 'learning_rate', 'weight_decay',
        'early_stopping_patience', 'device', 'num_threads', 'target_mode')}
    try:
        cfg, sources = load_config(None if args.config == 'none' else args.config, overrides=overrides, root=root)
        print(describe(cfg, sources, sections=('train', 'model')))
        manifest = train(cfg, sources, root, run_id=args.run_id,
                         resume=_resolve(root, args.resume) if args.resume else None)
    except (ConfigError, FileExistsError, FileNotFoundError) as e:
        raise SystemExit(f'Error: {e}')
    print(json.dumps({k: manifest[k] for k in ('run_id', 'status', 'start_mode', 'stop_reason', 'epochs_completed',
                                                'best_epoch', 'best_validation_mse_standardized',
                                                'training_elapsed_s', 'artifacts')}, indent=2))


if __name__ == '__main__':
    main()
