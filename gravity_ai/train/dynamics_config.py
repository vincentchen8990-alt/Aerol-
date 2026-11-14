"""Configuration for spacecraft-dynamics data generation, training and evaluation.

Precedence: built-in DEFAULTS < YAML file (--config) < explicit command-line overrides.
Unknown keys and invalid values raise ConfigError; nothing is silently ignored.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = 'configs/dynamics.yaml'

DEFAULTS = {
    'data': {
        'output_dir': 'data/synthetic/dynamics_v2',
        'seed': 49,
        'episodes': 500,
        'steps_per_episode': 120,
        'dt_values_s': [0.5, 1.0, 2.0, 5.0],
        'scenario_mix': {'mission_pid': 0.35, 'setpoint_steps': 0.35, 'random_torque': 0.30},
        'max_crew_gravity_g': 3.0,
        'ranges': {
            'radius_m': [40.0, 400.0],
            'ring_inertia_kg_m2': [1.0e9, 1.0e10],
            'hub_inertia_kg_m2': [2.0e9, 3.0e10],
            'bearing_drag_nms': [1.0e4, 1.5e6],
            'motor_torque_limit_nm': [2.0e6, 2.0e7],
            'motor_power_limit_w': [1.0e6, 1.0e7],
            'motor_slew_nm_s': [2.0e5, 4.0e6],
        },
    },
    'train': {
        'dataset_dir': 'data/synthetic/dynamics_v2',
        'output_root': 'outputs/runs',
        'seed': 17,
        'epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.0015,
        'weight_decay': 0.0001,
        'early_stopping_patience': 8,
        'device': 'auto',
        'num_threads': 0,
        'target_mode': 'residual',
    },
    'model': {'hidden_width': 64, 'hidden_layers': 3, 'activation': 'silu'},
    'evaluation': {'rollout_horizons_s': [10, 30, 60], 'rollout_start_stride_steps': 10},
}

SCENARIOS = ('mission_pid', 'setpoint_steps', 'random_torque')


class ConfigError(ValueError):
    pass


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check(cond, key, msg):
    if not cond:
        raise ConfigError(f'{key}: {msg}')


def _validate(cfg):
    d, t, m, e = cfg['data'], cfg['train'], cfg['model'], cfg['evaluation']
    for key in ('seed', 'episodes', 'steps_per_episode'):
        _check(_is_int(d[key]), f'data.{key}', 'must be an integer')
    _check(d['seed'] >= 0, 'data.seed', 'must be >= 0')
    _check(d['episodes'] >= 20, 'data.episodes', 'must be >= 20 (one split block of 20 episodes)')
    _check(d['steps_per_episode'] >= 2, 'data.steps_per_episode', 'must be >= 2')
    _check(isinstance(d['output_dir'], str) and d['output_dir'], 'data.output_dir', 'must be a non-empty path')
    dts = d['dt_values_s']
    _check(isinstance(dts, list) and dts and all(_is_num(x) and 0 < x <= 10 for x in dts),
           'data.dt_values_s', 'must be a non-empty list of numbers in (0, 10] (API range is 0.1-10 s)')
    _check(len(set(dts)) == len(dts), 'data.dt_values_s', 'values must be distinct')
    mix = d['scenario_mix']
    _check(isinstance(mix, dict) and set(mix) <= set(SCENARIOS) and mix, 'data.scenario_mix',
           f'keys must be a subset of {SCENARIOS}')
    _check(all(_is_num(v) and v >= 0 for v in mix.values()) and abs(sum(mix.values()) - 1) < 1e-6,
           'data.scenario_mix', 'weights must be >= 0 and sum to 1')
    _check(_is_num(d['max_crew_gravity_g']) and 1 < d['max_crew_gravity_g'] <= 10, 'data.max_crew_gravity_g', 'must be in (1, 10]')
    for key, rng in d['ranges'].items():
        _check(isinstance(rng, list) and len(rng) == 2 and all(_is_num(x) for x in rng) and 0 < rng[0] <= rng[1],
               f'data.ranges.{key}', 'must be [low, high] with 0 < low <= high')
    _check(d['ranges']['radius_m'][0] > 2.0, 'data.ranges.radius_m', 'low must exceed crew height (1.75 m)')

    for key in ('dataset_dir', 'output_root'):
        _check(isinstance(t[key], str) and t[key], f'train.{key}', 'must be a non-empty path')
    for key in ('seed', 'epochs', 'batch_size', 'early_stopping_patience', 'num_threads'):
        _check(_is_int(t[key]), f'train.{key}', 'must be an integer')
    _check(t['seed'] >= 0, 'train.seed', 'must be >= 0')
    _check(1 <= t['epochs'] <= 10000, 'train.epochs', 'must be in [1, 10000]')
    _check(1 <= t['batch_size'] <= 1_000_000, 'train.batch_size', 'must be in [1, 1000000]')
    _check(t['early_stopping_patience'] >= 1, 'train.early_stopping_patience', 'must be >= 1')
    _check(t['num_threads'] >= 0, 'train.num_threads', 'must be >= 0 (0 = PyTorch default)')
    _check(_is_num(t['learning_rate']) and 0 < t['learning_rate'] < 1, 'train.learning_rate', 'must be in (0, 1)')
    _check(_is_num(t['weight_decay']) and 0 <= t['weight_decay'] < 1, 'train.weight_decay', 'must be in [0, 1)')
    _check(t['device'] in ('auto', 'cpu', 'cuda'), 'train.device', "must be 'auto', 'cpu' or 'cuda'")
    _check(t['target_mode'] in ('residual', 'absolute'), 'train.target_mode', "must be 'residual' or 'absolute'")

    _check(_is_int(m['hidden_width']) and 4 <= m['hidden_width'] <= 4096, 'model.hidden_width', 'must be an integer in [4, 4096]')
    _check(_is_int(m['hidden_layers']) and 1 <= m['hidden_layers'] <= 16, 'model.hidden_layers', 'must be an integer in [1, 16]')
    _check(m['activation'] == 'silu', 'model.activation', "only 'silu' is implemented")

    hz = e['rollout_horizons_s']
    _check(isinstance(hz, list) and hz and all(_is_num(x) and x > 0 for x in hz), 'evaluation.rollout_horizons_s',
           'must be a non-empty list of positive seconds')
    _check(_is_int(e['rollout_start_stride_steps']) and e['rollout_start_stride_steps'] >= 1,
           'evaluation.rollout_start_stride_steps', 'must be an integer >= 1')
    return cfg


def _merge(base, update, path, sources, layer):
    for key, value in update.items():
        dotted = f'{path}.{key}' if path else key
        if key not in base:
            raise ConfigError(f'unknown setting {dotted!r}; allowed: {sorted(base)}')
        if isinstance(base[key], dict) and key not in ('scenario_mix',):
            if not isinstance(value, dict):
                raise ConfigError(f'{dotted}: expected a mapping')
            _merge(base[key], value, dotted, sources, layer)
        else:
            base[key] = copy.deepcopy(value)
            sources[dotted] = layer


def load_config(path: str | Path | None = DEFAULT_CONFIG_PATH, overrides: dict | None = None, root: Path | None = None):
    """Return (config, sources). `overrides` maps dotted keys (e.g. 'train.epochs') to values.

    A path of None uses built-in defaults only.
    """
    cfg = copy.deepcopy(DEFAULTS)
    sources = {}
    if path is not None:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = root / p
        if not p.is_file():
            raise ConfigError(f'config file not found: {p}')
        loaded = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f'{p}: top level must be a mapping')
        _merge(cfg, loaded, '', sources, f'file:{p.name}')
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        keys = dotted.split('.')
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise ConfigError(f'unknown setting {dotted!r}')
            node = node[k]
        if keys[-1] not in node:
            raise ConfigError(f'unknown setting {dotted!r}')
        node[keys[-1]] = value
        sources[dotted] = 'command-line'
    return _validate(cfg), sources


def describe(cfg, sources, sections=None):
    """Human-readable effective configuration with the source of each non-default value."""
    shown = {k: v for k, v in cfg.items() if sections is None or k in sections}
    lines = ['Effective configuration:', json.dumps(shown, indent=2)]
    relevant = {k: v for k, v in sources.items() if sections is None or k.split('.')[0] in sections}
    lines.append('Non-default values: ' + (', '.join(f'{k} <- {v}' for k, v in sorted(relevant.items())) or 'none (built-in defaults)'))
    return '\n'.join(lines)
