"""Focused checks for the dynamics-v2 pipeline: config handling, frozen splits, normalisation,
checkpoint loading, feature schema, actuator/rollout propagation, resume, evaluation and API."""
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from gravity_ai.models.dynamics import DynamicsNet
from gravity_ai.models.dynamics_predictor import (DynamicsPredictor, OutOfEnvelopeError, PARAM_KEYS, SchemaError,
                                                  rollout)
from gravity_ai.models.dynamics_schema import (FEATURE_NAMES, OUTPUT_NAMES, actuator_torque, encode_features,
                                               truth_outputs)
from gravity_ai.simulation.generate import generate_dataset, simulate_episode
from gravity_ai.simulation.physics import G0, SpacecraftParams, SpacecraftState, step
from gravity_ai.train.dynamics import load_dataset, train
from gravity_ai.train.dynamics_config import DEFAULTS, ConfigError, load_config


def small_cfg(tmp_path, episodes=80, steps=12, **train_overrides):
    overrides = {'data.episodes': episodes, 'data.steps_per_episode': steps, 'data.output_dir': str(tmp_path / 'data'),
                 'train.dataset_dir': str(tmp_path / 'data'), 'train.output_root': str(tmp_path / 'runs'),
                 'train.epochs': 2, 'train.batch_size': 64}
    overrides.update({f'train.{k}': v for k, v in train_overrides.items()})
    return load_config(None, overrides=overrides)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('dyn')
    cfg, sources = small_cfg(tmp)
    report = generate_dataset(cfg['data'], tmp / 'data', log=lambda *_: None)
    manifest = train(cfg, sources, tmp, run_id='t1', log=lambda *_: None)
    return {'tmp': tmp, 'cfg': cfg, 'report': report, 'manifest': manifest, 'run_dir': tmp / 'runs' / 't1'}


# ---- configuration --------------------------------------------------------------------------------
def test_config_precedence_defaults_file_cli(tmp_path):
    f = tmp_path / 'c.yaml'
    f.write_text('train:\n  epochs: 7\n  learning_rate: 0.01\n', encoding='utf-8')
    cfg, sources = load_config(f, overrides={'train.epochs': 3})
    assert cfg['train']['epochs'] == 3 and sources['train.epochs'] == 'command-line'
    assert cfg['train']['learning_rate'] == 0.01 and sources['train.learning_rate'] == 'file:c.yaml'
    assert cfg['train']['batch_size'] == DEFAULTS['train']['batch_size'] and 'train.batch_size' not in sources


@pytest.mark.parametrize('text', ['train:\n  epoch: 5\n', 'trian:\n  epochs: 5\n', 'train:\n  device: tpu\n',
                                  'train:\n  learning_rate: -1\n', 'model:\n  activation: relu\n',
                                  'data:\n  ranges:\n    radius_m: [1.0e9, 1.0e10]\n',
                                  'data:\n  ranges:\n    ring_inertia_kg_m2: [1.0e9, 1.0e10]\n'])
def test_config_rejects_unknown_or_invalid(tmp_path, text):
    # The last case is YAML 1.1 parsing 1.0e9 (no exponent sign) as a string: rejected, not ignored.
    f = tmp_path / 'bad.yaml'
    f.write_text(text, encoding='utf-8')
    with pytest.raises(ConfigError):
        load_config(f)


def test_repository_config_is_valid():
    cfg, _ = load_config('configs/dynamics.yaml')
    assert cfg == DEFAULTS


def test_cuda_request_without_cuda_is_rejected(tmp_path, trained):
    if torch.cuda.is_available():
        pytest.skip('CUDA present')
    cfg, sources = small_cfg(tmp_path, device='cuda', dataset_dir=str(trained['tmp'] / 'data'))
    with pytest.raises(ConfigError):
        train(cfg, sources, tmp_path, run_id='cuda', log=lambda *_: None)


# ---- data, splits, normalisation ------------------------------------------------------------------
def test_episode_content_and_split_stable_when_expanding(tmp_path):
    small, _ = small_cfg(tmp_path / 'a', episodes=40, steps=6)
    large, _ = small_cfg(tmp_path / 'b', episodes=80, steps=6)
    a = generate_dataset(small['data'], tmp_path / 'a', log=lambda *_: None)
    b = generate_dataset(large['data'], tmp_path / 'b', log=lambda *_: None)
    assert [m['split'] for m in a['episode_metadata']] == [m['split'] for m in b['episode_metadata'][:40]]
    da, db = np.load(tmp_path / 'a' / 'transitions.npz'), np.load(tmp_path / 'b' / 'transitions.npz')
    first = db['episode_id'] < 40
    for k in FEATURE_NAMES + OUTPUT_NAMES:
        np.testing.assert_array_equal(da[k], db[k][first])


def test_splits_disjoint_stratified_and_verified(trained):
    ds = load_dataset(trained['tmp'] / 'data')
    eps = [set(ds['arrays']['episode_id'][m].tolist()) for m in ds['masks']]
    assert not (eps[0] & eps[1] or eps[0] & eps[2] or eps[1] & eps[2])
    for dt_counts in trained['report']['episodes_by_dt_and_split'].values():
        assert dt_counts == {'train': 14, 'val': 3, 'test': 3}


def test_generator_rejects_overwrite(trained):
    with pytest.raises(FileExistsError):
        generate_dataset(trained['cfg']['data'], trained['tmp'] / 'data', log=lambda *_: None)


def test_transitions_aligned_with_simulator():
    rows, meta, _ = simulate_episode(3, DEFAULTS['data'] | {'steps_per_episode': 5})
    p = SpacecraftParams(**{k: meta[k] for k in PARAM_KEYS})
    r = rows[2]
    s = SpacecraftState(ring_omega_rad_s=r['ring_omega_rad_s'], hub_omega_rad_s=r['hub_omega_rad_s'],
                        applied_torque_nm=r['applied_torque_nm'])
    nxt, t = step(p, s, r['requested_torque_nm'], r['axial_accel_m_s2'], r['dt_s'])
    assert nxt.ring_omega_rad_s == r['next_ring_omega_rad_s'] and t['crew_gravity_radial_m_s2'] == r['crew_radial_m_s2']
    assert rows[3]['ring_omega_rad_s'] == r['next_ring_omega_rad_s']


def test_normalisation_fitted_on_training_split_only(trained):
    ds = load_dataset(trained['tmp'] / 'data')
    ck = torch.load(trained['run_dir'] / 'checkpoint_best.pt', weights_only=True)
    x = encode_features(ds['arrays'])
    np.testing.assert_allclose(ck['input_mean'], x[ds['masks'][0]].mean(axis=0), rtol=1e-12, atol=1e-15)
    assert not np.allclose(ck['input_mean'], x.mean(axis=0), rtol=1e-6)


# ---- checkpoint, schema, inference ----------------------------------------------------------------
def _example_raw():
    p = SpacecraftParams()
    return {'ring_omega_rad_s': 0.31, 'hub_omega_rad_s': 0.0, 'applied_torque_nm': 1e6, 'requested_torque_nm': 2e6,
            'axial_accel_m_s2': 0.0, 'ring_inertia_kg_m2': p.ring_inertia_kg_m2, 'hub_inertia_kg_m2': p.hub_inertia_kg_m2,
            'bearing_drag_nms': p.bearing_drag_nms, 'radius_m': p.radius_m, 'dt_s': 2.0,
            'motor_torque_limit_nm': p.motor_torque_limit_nm, 'motor_power_limit_w': p.motor_power_limit_w,
            'motor_slew_nm_s': p.motor_slew_nm_s}


def test_checkpoint_reload_reproduces_predictions(trained):
    a = DynamicsPredictor.load(trained['run_dir'])
    b = DynamicsPredictor.load(trained['run_dir'] / 'checkpoint_best.pt')
    raw = _example_raw()
    pa, pb = a.predict(raw, on_out_of_envelope='flag'), b.predict(raw, on_out_of_envelope='flag')
    for k in OUTPUT_NAMES:
        assert np.isfinite(pa[k]).all() and pa[k] == pytest.approx(pb[k], rel=0, abs=0)
    assert not a.model.training


def test_predictor_validates_feature_schema(trained):
    pr = DynamicsPredictor.load(trained['run_dir'])
    raw = _example_raw()
    with pytest.raises(SchemaError):
        pr.predict({k: v for k, v in raw.items() if k != 'radius_m'})
    with pytest.raises(SchemaError):
        pr.predict(raw | {'extra_feature': 1.0})
    with pytest.raises(SchemaError):
        pr.predict(raw | {'ring_omega_rad_s': float('nan')})
    with pytest.raises(SchemaError):
        pr.predict(raw | {'radius_m': -5.0})
    with pytest.raises(OutOfEnvelopeError):
        pr.predict(raw | {'dt_s': 3.0})
    flagged = pr.predict(raw | {'dt_s': 3.0}, on_out_of_envelope='flag')
    assert flagged['outside_training_envelope'].all()


def test_checkpoint_with_wrong_feature_order_rejected(trained):
    ck = torch.load(trained['run_dir'] / 'checkpoint_best.pt', weights_only=True)
    ck['schema']['features'] = list(reversed(ck['schema']['features']))
    with pytest.raises(SchemaError):
        DynamicsPredictor(ck)


def test_archived_checkpoint_still_loads_in_default_architecture():
    from gravity_ai.data.catalog_loader import project_root
    path = project_root() / 'outputs' / 'checkpoints' / 'dynamics_synthetic.pt'
    if not path.is_file():
        pytest.skip('archived checkpoint not present')
    ck = torch.load(path, weights_only=False)
    DynamicsNet(ck['inputs'], ck['outputs']).load_state_dict(ck['model_state'])


# ---- actuator and rollout -------------------------------------------------------------------------
def test_vectorised_actuator_matches_physics_step():
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = SpacecraftParams(motor_torque_limit_nm=rng.uniform(2e6, 2e7), motor_power_limit_w=rng.uniform(1e6, 1e7),
                             motor_slew_nm_s=rng.uniform(2e5, 4e6))
        s = SpacecraftState(ring_omega_rad_s=rng.uniform(0.05, 0.6), hub_omega_rad_s=rng.uniform(-0.3, 0.3),
                            applied_torque_nm=rng.uniform(-1, 1) * p.motor_torque_limit_nm)
        cmd, dt = rng.uniform(-1.5, 1.5) * p.motor_torque_limit_nm, float(rng.choice([0.5, 1, 2, 5]))
        nxt, _ = step(p, s, cmd, 0.0, dt)
        got = actuator_torque(s.applied_torque_nm, cmd, s.encoder_omega_rad_s, dt, p.motor_torque_limit_nm,
                              p.motor_power_limit_w, p.motor_slew_nm_s)
        assert float(got) == nxt.applied_torque_nm


def _physics_step_fn(arrays):
    out = []
    for i in range(len(arrays['ring_omega_rad_s'])):
        p = SpacecraftParams(**{k: float(arrays[k][i]) for k in PARAM_KEYS})
        s = SpacecraftState(ring_omega_rad_s=float(arrays['ring_omega_rad_s'][i]),
                            hub_omega_rad_s=float(arrays['hub_omega_rad_s'][i]),
                            applied_torque_nm=float(arrays['applied_torque_nm'][i]))
        nxt, t = step(p, s, float(arrays['requested_torque_nm'][i]), float(arrays['axial_accel_m_s2'][i]),
                      float(arrays['dt_s'][i]))
        out.append([nxt.ring_omega_rad_s, nxt.hub_omega_rad_s, t['crew_gravity_radial_m_s2'],
                    t['crew_gravity_tangential_m_s2'], t['crew_gravity_axial_m_s2']])
    return np.array(out)


def test_rollout_propagates_complete_state(trained):
    """With the simulator itself as the one-step model, a rollout that receives only the initial
    state and the known commands must reproduce the whole recorded trajectory, including the
    actuator torque that the learned model does not predict."""
    a = load_dataset(trained['tmp'] / 'data')['arrays']
    idx = np.where(a['episode_id'] == 5)[0]
    idx = idx[np.argsort(a['step_index'][idx])]
    first = idx[:1]
    pred, applied = rollout(_physics_step_fn, a['ring_omega_rad_s'][first], a['hub_omega_rad_s'][first],
                            a['applied_torque_nm'][first], a['requested_torque_nm'][idx][None],
                            a['axial_accel_m_s2'][idx][None], {k: a[k][first] for k in PARAM_KEYS}, a['dt_s'][idx[0]])
    np.testing.assert_allclose(pred[0], truth_outputs(a)[idx], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(applied[0], a['next_applied_torque_nm'][idx])


def test_rollout_feeds_back_predictions_not_truth():
    calls = []

    def biased(arrays):
        calls.append(np.array(arrays['ring_omega_rad_s']))
        r = arrays['ring_omega_rad_s']
        return np.stack([r + 0.01, arrays['hub_omega_rad_s'], arrays['radius_m'] * r**2, 0 * r,
                         -arrays['axial_accel_m_s2']], 1)

    params = {k: getattr(SpacecraftParams(), k) for k in PARAM_KEYS}
    rollout(biased, 0.3, 0.0, 0.0, np.zeros((1, 4)), np.zeros((1, 4)), params, 1.0)
    assert [float(c[0]) for c in calls] == pytest.approx([0.30, 0.31, 0.32, 0.33])


# ---- resume, evaluation, API ----------------------------------------------------------------------
def test_resume_restores_state_and_extends_history(tmp_path, trained):
    cfg, sources = small_cfg(tmp_path, dataset_dir=str(trained['tmp'] / 'data'))
    first = train(cfg, sources, tmp_path, run_id='r', log=lambda *_: None)
    assert first['start_mode'] == 'from_scratch' and first['epochs_completed'] == 2
    cfg3, sources3 = small_cfg(tmp_path, dataset_dir=str(trained['tmp'] / 'data'), epochs=3)
    resumed = train(cfg3, sources3, tmp_path, resume=tmp_path / 'runs' / 'r', log=lambda *_: None)
    assert resumed['start_mode'].startswith('resumed') and resumed['epochs_completed'] == 3
    assert [h['epoch'] for h in resumed['history']] == [1, 2, 3]
    assert resumed['history'][:2] == first['history']
    bad, bad_sources = small_cfg(tmp_path, dataset_dir=str(trained['tmp'] / 'data'), epochs=4, learning_rate=0.1)
    with pytest.raises(ConfigError):
        train(bad, bad_sources, tmp_path, resume=tmp_path / 'runs' / 'r', log=lambda *_: None)


def test_fresh_evaluation_from_disk(trained):
    from gravity_ai.evaluate_dynamics import evaluate_run
    r = evaluate_run(trained['run_dir'], horizons=[2, 4], stride=4, log=lambda *_: None)
    assert r['loaded_from_disk'] and r['one_step']['test']['episodes'] == 12
    assert set(r['one_step']['test']['models']) == {'learned_model', 'persistence', 'ridge_baseline'}
    assert r['rollouts'] and (trained['run_dir'] / 'evaluation' / 'evaluation.json').is_file()
    stored = json.loads((trained['run_dir'] / 'evaluation' / 'evaluation.json').read_text())
    assert stored['dataset_fingerprint_sha256'] == trained['report']['dataset_fingerprint_sha256']


def test_api_reports_model_status_and_keeps_controller(trained, monkeypatch):
    import gravity_ai.api.app as api
    import gravity_ai.api.learned_dynamics as ld
    monkeypatch.setattr(ld, 'selected_run_dir', lambda root: (trained['run_dir'], 'test'))
    client = TestClient(api.app)
    status = client.get('/api/models').json()['learned_dynamics_predictor']
    assert status['checkpoint_exists'] and status['checkpoint_loads']
    assert status['used_for_control'] is False and status['used_by_api_simulate'] is False
    r = client.post('/api/dynamics/compare', json={'duration_s': 120, 'dt_s': 2, 'max_points': 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['active_controller'] == 'pid' and body['learned_model']['role'] == 'prediction_only'
    assert body['steps'] == 60 and np.isfinite(body['one_step']['gravity_magnitude_mae_g'])
    sim = client.post('/api/simulate', json={'duration_s': 120, 'dt_s': 2, 'max_points': 50}).json()
    assert sim['data_type'] == 'SYNTHETIC_SPACECRAFT_DYNAMICS'


def test_api_compare_without_trained_model(monkeypatch):
    import gravity_ai.api.app as api
    import gravity_ai.api.learned_dynamics as ld
    monkeypatch.setattr(ld, 'selected_run_dir', lambda root: (None, 'no selected or completed run'))
    client = TestClient(api.app)
    assert client.get('/api/models').json()['learned_dynamics_predictor']['checkpoint_loads'] is False
    assert client.post('/api/dynamics/compare', json={'duration_s': 120, 'dt_s': 2}).status_code == 503
    assert client.post('/api/simulate', json={'duration_s': 120, 'dt_s': 2, 'max_points': 50}).status_code == 200
