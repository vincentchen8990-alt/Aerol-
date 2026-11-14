"""Integration contract between the learned predictor, /api/models, /api/dynamics/compare and the
Orbital Lab viewer: run selection, metadata, one-step vs open-loop forecast semantics, timestamp
alignment, unsupported/out-of-range settings, and failure handling that keeps the simulation usable."""
import json

import pytest
from fastapi.testclient import TestClient

import gravity_ai.api.learned_dynamics as ld
from gravity_ai.api.app import app
from gravity_ai.predict_dynamics import ENV_RUN_DIR, selected_run_dir
from gravity_ai.simulation.generate import generate_dataset
from gravity_ai.train.dynamics import train
from gravity_ai.train.dynamics_config import load_config

client = TestClient(app)
MISSION = {'duration_s': 120, 'dt_s': 2, 'max_points': 900}


@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('viewer')
    cfg, sources = load_config(None, overrides={
        'data.episodes': 80, 'data.steps_per_episode': 12, 'train.dataset_dir': str(tmp / 'data'),
        'train.output_root': str(tmp / 'runs'), 'train.epochs': 2, 'train.batch_size': 64})
    generate_dataset(cfg['data'], tmp / 'data', log=lambda *_: None)
    train(cfg, sources, tmp, run_id='viewer', log=lambda *_: None)
    return tmp / 'runs' / 'viewer'


@pytest.fixture
def serve(monkeypatch, run_dir):
    monkeypatch.setattr(ld, 'selected_run_dir', lambda root: (run_dir, 'test selection'))
    return run_dir


def compare(**kw):
    return client.post('/api/dynamics/compare', json={**MISSION, **kw})


# ---- which checkpoint is served ---------------------------------------------------------------------
def test_selected_run_precedence(tmp_path, monkeypatch):
    runs = tmp_path / 'outputs' / 'runs'
    runs.mkdir(parents=True)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)
    assert selected_run_dir(tmp_path) == (None, 'no selected or completed run')
    (runs / 'latest_dynamics.json').write_text(json.dumps({'run_dir': str(runs / 'newest')}))
    assert selected_run_dir(tmp_path)[0] == runs / 'newest'
    (runs / 'selected_dynamics.json').write_text(json.dumps({'run_dir': 'outputs/runs/chosen'}))
    path, source = selected_run_dir(tmp_path)
    assert path == tmp_path / 'outputs' / 'runs' / 'chosen' and 'selected_dynamics.json' in source
    monkeypatch.setenv(ENV_RUN_DIR, str(tmp_path / 'override'))
    assert selected_run_dir(tmp_path) == (tmp_path / 'override', f'environment variable {ENV_RUN_DIR}')


def test_repository_serves_the_selected_run():
    from gravity_ai.data.catalog_loader import project_root
    pointer = project_root() / 'outputs' / 'runs' / 'selected_dynamics.json'
    if not pointer.is_file():
        pytest.skip('no selected run in this checkout')
    assert json.loads(pointer.read_text())['run_id'] == 'dyn-v2-residual-e200-p20'


# ---- metadata the viewer reads ----------------------------------------------------------------------
def test_models_metadata_for_viewer(serve):
    m = client.get('/api/models').json()['learned_dynamics_predictor']
    assert m['checkpoint_loads'] and m['run_id'] == 'viewer' and m['selection_source'] == 'test selection'
    assert m['supported_dt_s'] == [0.5, 1.0, 2.0, 5.0]
    r = m['training_ranges']['radius_m']
    assert r['unit'] == 'm' and 40 <= r['min'] < r['max'] <= 400
    assert m['used_for_control'] is False and m['used_by_api_simulate'] is False and 'Orbital Lab' in m['used_by_viewer']


# ---- one-step vs open-loop forecast -------------------------------------------------------------------
def test_forecast_window_starts_from_simulated_state_and_is_never_reset(serve):
    d = compare(forecast_start_s=20, forecast_horizon_s=10).json()
    f, one = d['forecast'], {p['time_s']: p for p in d['series']}
    assert (f['start_s'], f['horizon_s'], f['steps'], f['truncated_at_mission_end']) == (20.0, 10.0, 5, False)
    assert [p['time_s'] for p in f['series']] == [22.0, 24.0, 26.0, 28.0, 30.0]
    assert f['start_state']['ring_rpm'] == pytest.approx(one[20.0]['simulator_ring_rpm'], rel=1e-12)
    # Step 1 of the forecast and the one-step prediction share the same (simulated) input state
    # (float32 network, different batch sizes -> agreement to rounding noise) ...
    assert f['series'][0]['forecast_ring_rpm'] == pytest.approx(one[22.0]['one_step_ring_rpm'], abs=1e-6)
    # ... later steps build on the model's own predictions instead of being reset to the simulator.
    assert abs(f['series'][-1]['forecast_ring_rpm'] - one[30.0]['one_step_ring_rpm']) > 1e-6
    end = f['series'][-1]
    assert f['error_at_end']['ring_speed_error_rpm'] == pytest.approx(end['forecast_ring_rpm'] - end['simulator_ring_rpm'])
    assert f['error_at_end']['gravity_error_g'] == pytest.approx(end['forecast_gravity_g'] - end['simulator_gravity_g'])
    assert 'not re-planned' in f['command_source'] and d['active_controller'] == 'pid'


def test_rest_of_mission_forecast_from_zero_matches_whole_mission_rollout(serve):
    d = compare(forecast_start_s=0, forecast_horizon_s=None).json()
    assert d['forecast']['horizon_s'] == 120.0 and d['forecast']['steps'] == 60
    assert d['forecast']['series'][-1]['forecast_ring_rpm'] == pytest.approx(d['series'][-1]['rollout_ring_rpm'], rel=1e-12)


def test_forecast_truncated_at_mission_end_and_invalid_starts(serve):
    f = compare(forecast_start_s=110, forecast_horizon_s=60).json()['forecast']
    assert f['truncated_at_mission_end'] and f['horizon_s'] == 10.0 and f['requested_horizon_s'] == 60.0
    for bad in ({'forecast_start_s': 3}, {'forecast_start_s': 120}, {'forecast_horizon_s': 3}):
        r = compare(**bad)
        assert r.status_code == 422 and r.json()['detail']['code'] == 'invalid_forecast'


# ---- same mission, aligned timestamps -----------------------------------------------------------------
@pytest.mark.parametrize('mission', [MISSION, {'duration_s': 1200, 'dt_s': 0.5, 'max_points': 900}])
def test_compare_uses_the_same_mission_and_timestamps_as_simulate(serve, mission):
    body = {**mission, 'radius_m': 150, 'target_gravity_g': 0.8, 'motor_torque_limit_nm': 6e6}
    d = client.post('/api/dynamics/compare', json=body).json()
    sim = client.post('/api/simulate', json=body).json()['samples']
    assert all(d['mission'][k] == v for k, v in body.items())
    assert [s['time_s'] for s in sim] == [p['time_s'] for p in d['series']]
    for s, p in zip(sim[::50], d['series'][::50]):
        assert p['simulator_ring_rpm'] == pytest.approx(s['ring_rpm'], rel=1e-12)
        assert p['simulator_gravity_g'] == pytest.approx(s['crew_gravity_g'], rel=1e-12)


# ---- supported conditions -----------------------------------------------------------------------------
def test_unsupported_time_step_is_refused_unless_explicitly_allowed(serve):
    r = compare(dt_s=3)
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['code'] == 'unsupported_time_step' and detail['supported_dt_s'] == [0.5, 1.0, 2.0, 5.0]
    ok = compare(dt_s=3, forecast_horizon_s=12, allow_unsupported_time_step=True).json()
    assert ok['learned_model']['time_step_supported'] is False and 'dt_s' in ok['training_envelope']['details']
    assert client.post('/api/simulate', json={**MISSION, 'dt_s': 3}).status_code == 200


def test_out_of_range_settings_are_reported_with_trained_range(serve):
    env = compare(radius_m=450).json()['training_envelope']
    det = env['details']['radius_m']
    assert not env['inside_all_ranges'] and det['value_min'] == 450 and det['trained_max'] < 450 and det['unit'] == 'm'
    assert 'not a guarantee of accuracy' in env['note']


# ---- failures keep the simulation usable ----------------------------------------------------------------
def test_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(ld, 'selected_run_dir', lambda root: (tmp_path / 'absent', 'test selection'))
    m = client.get('/api/models').json()['learned_dynamics_predictor']
    assert not m['checkpoint_exists'] and not m['checkpoint_loads'] and 'not found' in m['load_error']
    r = compare()
    assert r.status_code == 503 and r.json()['detail']['code'] == 'model_unavailable'
    assert client.post('/api/simulate', json=MISSION).status_code == 200


def test_corrupt_checkpoint(monkeypatch, tmp_path):
    bad = tmp_path / 'corrupt'
    bad.mkdir()
    (bad / 'checkpoint_best.pt').write_bytes(b'not a checkpoint')
    monkeypatch.setattr(ld, 'selected_run_dir', lambda root: (bad, 'test selection'))
    m = client.get('/api/models').json()['learned_dynamics_predictor']
    assert m['checkpoint_exists'] and not m['checkpoint_loads'] and m['load_error']
    assert compare().status_code == 503
    assert client.post('/api/simulate', json=MISSION).status_code == 200


# ---- the served viewer ----------------------------------------------------------------------------------
def test_viewer_contains_prediction_panel():
    html = client.get('/').text
    assert '<title>Orbital Lab' in html and 'id="pred-panel"' in html
    for needle in ('/api/dynamics/compare', '/api/models', 'One-step prediction', 'Open-loop forecast',
                   'simulated physics — not measured spacecraft data', 'PREDICTION ONLY',
                   '<option value="10">10 s</option>', '<option value="30">30 s</option>',
                   '<option value="60" selected>60 s</option>', '<option value="rest">Rest of mission</option>',
                   "body:JSON.stringify(missionBody())", '...missionBody()', 'if(seq!==pred.seq)return'):
        assert needle in html, needle
