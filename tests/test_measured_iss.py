"""Archived measured-ISS integration: exported test-vector parity, input shapes, timestamp/gap handling,
g vs micro-g, invalid negative RPM, one-step-only behaviour, honest unavailability, archived labelling
and separation from the unchanged simulated ring."""
import json
import math
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gravity_ai.api.app import app
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.measured import assets
from gravity_ai.measured.inference import (MeasuredInputError, PimsModel, RpmModel, nominal_radial_g,
                                           parse_timestamps, pims_histories, rpm_histories)

client = TestClient(app)
ROOT = assets.measured_root()
pytestmark = pytest.mark.skipif(not (ROOT / assets.MANIFEST).is_file(), reason='measured-ISS handoff not present')

RPM_TS = [f'2017-08-17T17:23:{25 + i:02d}' for i in range(8)]
PIMS_TS = [f'2026-11-10T00:{(7 + 16 * i) // 60:02d}:{(7 + 16 * i) % 60:02d}.070Z' for i in range(8)]


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def rpm_model():
    return RpmModel(load(assets.RPM_MODEL))


@pytest.fixture(scope='module')
def pims_model():
    return PimsModel(load(assets.PIMS_MODEL))


@pytest.fixture(scope='module')
def pims_vectors():
    return load(assets.PIMS_TEST_VECTORS)


def rpm_post(history, timestamps=RPM_TS, **kw):
    return client.post('/api/measured/rpm/predict', json={'history_rpm': history, 'timestamps': timestamps, **kw})


def pims_post(history, timestamps=PIMS_TS, **kw):
    return client.post('/api/measured/pims/predict', json={'history_g': history, 'timestamps_utc': timestamps, **kw})


# ---- exported test-vector parity --------------------------------------------------------------------
def test_rpm_test_vectors_single_and_batched(rpm_model):
    tv = load(assets.RPM_TEST_VECTORS)
    expected = np.asarray(tv['mlp_next_rpm'])
    np.testing.assert_allclose(rpm_model.predict_raw(tv['histories_rpm']), expected, rtol=0, atol=1e-10)
    for history, value in zip(tv['histories_rpm'], expected):
        out = rpm_model.predict_raw(history)
        assert out.shape == (1,) and abs(out[0] - value) < 1e-10
        api = rpm_post(history).json()
        assert abs(api['diagnostics']['raw_model_output_rpm'] - value) < 1e-10 and api['forecast_rpm'] == pytest.approx(value, abs=1e-10)


def test_pims_test_vectors_single_and_batched(pims_model, pims_vectors):
    hist = np.array([e['history_g'] for e in pims_vectors['examples']])
    expected = np.array([e['expected_next_g'] for e in pims_vectors['examples']])
    assert pims_vectors['units'] == 'g' and hist.shape == (8, 8, 3)
    np.testing.assert_allclose(pims_model.predict(hist), expected, rtol=1e-12, atol=1e-20)
    for h, e in zip(hist, expected):
        np.testing.assert_allclose(pims_model.predict(h)[0], e, rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(pims_post(h.tolist()).json()['forecast_g'], e, rtol=1e-12, atol=1e-20)


def test_matches_handoff_reference_implementations(rpm_model, pims_model, pims_vectors):
    import types

    def module(name, rel):  # in-memory load: no __pycache__ written into the handoff package
        m = types.ModuleType(name)
        exec(compile((ROOT / rel).read_text(encoding='utf-8'), str(ROOT / rel), 'exec'), m.__dict__)
        return m
    ref_rpm, ref_pims = module('ref_rpm', 'rpm_predict.py'), module('ref_pims', 'outputs/pims/inference.py')
    replay = load(assets.RPM_REPLAY)
    hist = np.array([[p['measured_rpm'] for p in c['points']][i - 8:i] for c in replay['clips'] for i in range(8, len(c['points']))])
    assert np.max(np.abs(rpm_model.predict_raw(hist) - ref_rpm.predict(load(assets.RPM_MODEL), hist))) < 1e-10
    ph = np.array([e['history_g'] for e in pims_vectors['examples']])
    assert np.max(np.abs(pims_model.predict(ph) - ref_pims.predict_batch(load(assets.PIMS_MODEL), ph))) < 1e-20


# ---- input shapes ------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', [[77.0] * 7, [77.0] * 9, [[77.0]] * 8, [[77.0] * 4] * 2, [[[77.0]] * 8], 77.0])
def test_rpm_rejects_wrong_shapes(bad):
    with pytest.raises(MeasuredInputError) as e:
        rpm_histories(bad)
    assert e.value.code == 'invalid_shape'
    r = rpm_post(bad)
    assert r.status_code == 422 and r.json()['detail']['code'] == 'invalid_shape' and r.json()['detail']['forecast_rpm'] is None


def test_rpm_accepts_flat_and_batched():
    assert rpm_histories([77.0] * 8).shape == (1, 8)
    assert rpm_histories([[77.0] * 8] * 3).shape == (3, 8)
    r = rpm_post([[77.0] * 8] * 2)  # the API takes one history per request
    assert r.status_code == 422 and r.json()['detail']['code'] == 'invalid_shape'


@pytest.mark.parametrize('shape', [(3, 8), (24,), (8, 2), (7, 3), (8, 4), (1, 24)])
def test_pims_rejects_wrong_shapes(shape):
    bad = np.full(shape, 1e-6).tolist()
    with pytest.raises(MeasuredInputError) as e:
        pims_histories(bad)
    assert e.value.code == 'invalid_shape'
    assert pims_post(bad).json()['detail']['code'] == 'invalid_shape'


def test_pims_accepts_8x3_and_batched():
    assert pims_histories(np.full((8, 3), 1e-6)).shape == (1, 8, 3)
    assert pims_histories(np.full((5, 8, 3), 1e-6)).shape == (5, 8, 3)


@pytest.mark.parametrize('bad,code', [([77.0] * 7 + [float('nan')], 'nonfinite_value'), ([77.0] * 7 + [-1.0], 'negative_measured_rpm'),
                                      ([77.0] * 7 + ['77'], 'invalid_value'), ([77.0] * 7 + [True], 'invalid_value')])
def test_rpm_rejects_invalid_values(bad, code):
    with pytest.raises(MeasuredInputError) as e:
        rpm_histories(bad)
    assert e.value.code == code


# ---- timestamps, cadence and gaps ------------------------------------------------------------------------
def test_rpm_timestamps_must_be_consecutive_seconds():
    ok = rpm_post([77.8994] * 8).json()
    assert ok['history_timestamps'] == RPM_TS and ok['forecast_timestamp'] == '2017-08-17T17:23:33'
    assert 'unspecified' in ok['timestamp_timezone'] and ok['cadence_s'] == 1.0 and ok['horizon_s'] == 1.0
    gap = RPM_TS[:4] + [f'2017-08-17T17:23:{30 + i:02d}' for i in range(4)]  # 2 s jump between samples 3 and 4
    r = rpm_post([77.8994] * 8, gap)
    d = r.json()['detail']
    assert r.status_code == 422 and d['code'] == 'invalid_cadence' and d['index'] == 4 and d['interval_s'] == 2.0
    assert d['available'] is False and d['status'] == 'unavailable' and d['forecast_rpm'] is None
    for stamps in (list(reversed(RPM_TS)), RPM_TS[:7] + [RPM_TS[6]], RPM_TS[:7], None,
                   RPM_TS[:7] + ['2017-08-17T17:23:32Z'], RPM_TS[:7] + ['not a time']):
        r = rpm_post([77.8994] * 8, stamps)
        assert r.status_code == 422 and r.json()['detail']['code'] in (
            'invalid_cadence', 'invalid_timestamps', 'timestamps_required', 'timestamp_timezone'), stamps


def test_pims_timestamps_must_be_utc_and_16_s_apart(pims_vectors):
    h = pims_vectors['examples'][0]['history_g']
    ok = pims_post(h).json()
    assert ok['history_timestamps_utc'] == PIMS_TS and ok['forecast_timestamp_utc'] == '2026-11-10T00:02:15.070Z'
    first, last = (datetime.fromisoformat(t.replace('Z', '+00:00')) for t in (PIMS_TS[0], PIMS_TS[-1]))
    assert (last - first).total_seconds() == 112 == ok['history_span_s'] and ok['cadence_s'] == ok['horizon_s'] == 16.0
    naive = [t[:-1] for t in PIMS_TS]
    assert pims_post(h, naive).json()['detail']['code'] == 'timestamp_timezone'
    for wrong_step in (15, 17, 32, 1):
        t, stamps = datetime(2026, 11, 10, 1, 0, 0), []
        for i in range(8):
            t += timedelta(seconds=0 if i == 0 else (wrong_step if i == 5 else 16))
            stamps.append(t.isoformat(timespec='milliseconds') + 'Z')
        r = pims_post(h, stamps)
        assert r.status_code == 422 and r.json()['detail']['code'] == 'invalid_cadence' and r.json()['detail']['forecast_g'] is None


def test_parse_timestamps_tolerates_only_sub_millisecond_jitter():
    parsed = parse_timestamps(['2026-11-10T00:00:00.000Z', '2026-11-10T00:00:16.000Z'], 2, 16, require_utc=True)
    assert (parsed[1] - parsed[0]).total_seconds() == 16
    with pytest.raises(MeasuredInputError) as e:
        parse_timestamps(['2026-11-10T00:00:00.000Z', '2026-11-10T00:00:16.002Z'], 2, 16, require_utc=True)
    assert e.value.code == 'invalid_cadence'


# ---- g versus micro-g ----------------------------------------------------------------------------------
def test_pims_units_are_g_and_micro_g_inputs_are_refused(pims_vectors):
    h = np.array(pims_vectors['examples'][0]['history_g'])
    ok = pims_post(h.tolist()).json()
    assert ok['units'] == 'g' and 'multiply g values by 1e6' in ok['display_units']
    assert max(abs(v) for v in ok['forecast_g']) < 1e-5  # g-scale values, not micro-g
    assert ok['coordinate_frame'] == 'SSAnalysis' and ok['live_telemetry'] is False
    r = pims_post((h * 1e6).tolist())  # micro-g passed where g is required
    assert r.status_code == 422 and r.json()['detail']['code'] == 'units_suspect'


def test_viewer_converts_to_micro_g_only_by_1e6():
    js = client.get('/iss_measured.js').content.decode('utf-8')
    assert 'const MICRO = 1e6;' in js and '9.80665' not in js
    assert '* MICRO' in js and 'µg' in js


# ---- negative RPM, derived g, one-step only ---------------------------------------------------------------
def test_negative_forecast_is_invalid_with_raw_value_kept(rpm_model):
    replay = load(assets.RPM_REPLAY)
    pts = next(c for c in replay['clips'] if c['name'] == 'first_recorded_slowdown')['points']
    rpm = [p['measured_rpm'] for p in pts]
    i = next(i for i, p in enumerate(pts) if p['forecast_status'] == 'invalid_negative_rpm' and i >= 8
             and all(pts[j]['elapsed_seconds'] + 1 == pts[j + 1]['elapsed_seconds'] for j in range(i - 8, i)))
    history, stamps = rpm[i - 8:i], [p['timestamp_as_recorded'] for p in pts[i - 8:i]]
    raw = float(rpm_model.predict_raw(history)[0])
    assert raw < 0
    d = rpm_post(history, stamps).json()
    assert d['available'] is True and d['valid'] is False and d['status'] == 'invalid_negative_rpm'
    assert d['forecast_rpm'] is None and d['derived_nominal_radial_g']['from_forecast_rpm'] is None
    assert d['derived_nominal_radial_g']['suppressed_reason']
    assert d['diagnostics']['raw_model_output_rpm'] == pytest.approx(raw, abs=1e-12) and d['diagnostics']['clipped'] is False
    assert d['forecast_timestamp'] == pts[i]['timestamp_as_recorded']


def test_derived_nominal_radial_g_formula():
    for rpm in (0.0, 13.4235, 77.0, 77.8994):
        assert nominal_radial_g(rpm) == pytest.approx(0.15 * (2 * math.pi * rpm / 60) ** 2 / 9.80665, rel=1e-14)
    assert nominal_radial_g(-0.01) is None and nominal_radial_g(None) is None
    d = rpm_post([77.8994] * 8).json()['derived_nominal_radial_g']
    assert d['label'] == 'derived nominal radial g' and 'not directly measured gravity' in d['note']
    hw = client.get('/api/measured/hardware').json()
    assert hw['derived']['mars_s_nominal_setting_radial_g'] == pytest.approx(0.15 * (2 * math.pi * 77 / 60) ** 2 / 9.80665)


def test_rpm_is_one_step_only():
    r = rpm_post([77.8994] * 8, steps=2)
    assert r.status_code == 422 and r.json()['detail']['code'] == 'one_step_only' and r.json()['detail']['forecast_rpm'] is None
    assert rpm_post([77.8994] * 8, horizon_s=60).status_code == 422  # unknown fields are refused
    assert pims_post(np.full((8, 3), 1e-6).tolist(), steps=15).json()['detail']['code'] == 'one_step_only'
    paths = [p for p in client.get('/openapi.json').json()['paths'] if p.startswith('/api/measured')]
    assert '/api/measured/rpm/predict' in paths and not any('rollout' in p or 'recursive' in p for p in paths)
    status = client.get('/api/measured/status').json()['models']['mars_s_rpm']
    assert status['one_step_only'] is True and status['used_for_control'] is False and status['torque_response_model'] is False
    assert 'FAILED' in status['metrics']['recursive_status']


# ---- replays: real data, gaps preserved, fresh history ----------------------------------------------------
def test_rpm_replay_preserves_gaps_and_resets_history():
    d = client.get('/api/measured/rpm/replay').json()
    src = load(assets.RPM_REPLAY)
    assert d['data_kind'] == 'ARCHIVED_MEASUREMENTS_REPLAY' and d['live_telemetry'] is False and d['mission'] == 'MHU-2'
    assert d['mission_recorded_span']['first_timestamp_as_recorded'].startswith('2017-08-17') and d['cadence_s'] == 1.0
    for clip, orig in zip(d['clips'], src['clips']):
        pts = clip['points']
        assert [p['timestamp_as_recorded'] for p in pts] == [p['timestamp_as_recorded'] for p in orig['points']]  # nothing added
        assert [p['measured_rpm'] for p in pts] == [p['measured_rpm'] for p in orig['points']]
        assert clip['gaps'], 'each exported clip contains a real recorded gap'
        stamps = [datetime.fromisoformat(p['timestamp_as_recorded']) for p in pts]
        for i, p in enumerate(pts):
            if p['missing_samples_before']:
                assert all(pts[j]['forecast_status'] == 'insufficient_contiguous_measured_history' for j in range(i, i + 8))
                assert pts[i]['history_reset_reason'] == 'after_gap'
            if p['forecast_issued_at']:
                # history = the 8 measured samples immediately before, each exactly 1 s apart
                assert (stamps[i] - stamps[i - 8]).total_seconds() == 8 and p['forecast_issued_at'] == pts[i - 1]['timestamp_as_recorded']
                if p['exported_forecast_rpm'] is not None:
                    assert p['raw_model_output_rpm'] == pytest.approx(p['exported_forecast_rpm'], abs=1e-10)
            if p['forecast_status'] == 'invalid_negative_rpm':
                assert p['forecast_rpm'] is None and p['derived_nominal_radial_g_from_forecast'] is None and p['raw_model_output_rpm'] < 0
            if p['forecast_status'] == 'one_step_from_fresh_measured_history':
                assert p['forecast_rpm'] >= 0 and p['derived_nominal_radial_g_from_forecast'] == pytest.approx(nominal_radial_g(p['forecast_rpm']))
    slowdown = next(c for c in d['clips'] if c['name'] == 'first_recorded_slowdown')
    assert slowdown['status_counts']['invalid_negative_rpm'] == 92


def test_pims_replay_is_real_16s_utc_data():
    path = assets.pims_replay_path()
    if not path.is_file():
        pytest.skip('PIMS replay has not been built in this checkout')
    d = client.get('/api/measured/pims/replay').json()
    assert d['live_telemetry'] is False and d['units'] == 'g' and d['coordinate_frame'] == 'SSAnalysis' and d['cadence_s'] == 16.0
    assert d['provenance']['matches_training_dataset_sha256'] is True
    names = {c['name'] for c in d['clips']}
    assert 'held_out_test_start' in names and 'recorded_gap' in names
    for clip in d['clips']:
        pts = clip['points']
        stamps = [datetime.fromisoformat(p['timestamp_utc'].replace('Z', '+00:00')) for p in pts]
        assert all(t.tzinfo is not None and datetime(2026, 11, 1).date() <= t.date() <= datetime(2026, 11, 11).date() for t in stamps)
        steps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        assert all(s == 16 or s > 16 for s in steps)  # never upsampled below the 16-s cadence
        assert all(max(abs(v) for v in p['measured_g']) < 1e-4 for p in pts)  # stored in g
        for i, p in enumerate(pts):
            if p['forecast_g'] is not None:
                assert (stamps[i] - stamps[i - 8]).total_seconds() == 128 and (stamps[i - 1] - stamps[i - 8]).total_seconds() == 112
    gap = next(c for c in d['clips'] if c['name'] == 'recorded_gap')
    assert gap['gaps'] and gap['gaps'][0]['missing_samples'] >= 1
    i = next(i for i, p in enumerate(gap['points']) if p['missing_samples_before'])
    assert all(gap['points'][j]['forecast_g'] is None for j in range(i, i + 8))
    assert gap['points'][i + 8]['forecast_g'] is not None


# ---- honest unavailability -------------------------------------------------------------------------------
@pytest.fixture
def handoff_copy(tmp_path, monkeypatch):
    files = [assets.MANIFEST, assets.HARDWARE, assets.RPM_MODEL, assets.RPM_REPORT, assets.RPM_REPLAY,
             assets.PIMS_MODEL, assets.PIMS_REPORT, assets.PIMS_SOURCE_MANIFEST]
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    monkeypatch.setenv(assets.ENV_ROOT, str(tmp_path))
    return tmp_path


def test_missing_model_is_reported_unavailable(handoff_copy):
    (handoff_copy / assets.RPM_MODEL).unlink()
    r = rpm_post([77.8994] * 8)
    assert r.status_code == 503 and r.json()['detail']['code'] == 'model_unavailable' and r.json()['detail']['forecast_rpm'] is None
    replay = client.get('/api/measured/rpm/replay').json()
    assert replay['model']['available'] is False
    pts = [p for c in replay['clips'] for p in c['points']]
    assert all(p['forecast_rpm'] is None and p['raw_model_output_rpm'] is None for p in pts)
    assert {p['forecast_status'] for p in pts} == {'model_unavailable', 'insufficient_contiguous_measured_history'}
    assert client.get('/api/models').json()['measured_iss']['mars_s_rpm']['available'] is False
    assert client.post('/api/simulate', json={'duration_s': 120, 'dt_s': 2, 'max_points': 50}).status_code == 200


def test_altered_model_fails_the_manifest_hash(handoff_copy):
    path = handoff_copy / assets.PIMS_MODEL
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    status = client.get('/api/measured/status').json()['models']['iss_pims_acceleration']
    assert status['available'] is False and status['load_error_code'] == 'hash_mismatch'
    r = pims_post(np.full((8, 3), 1e-6).tolist())
    assert r.status_code == 503 and r.json()['detail']['forecast_g'] is None


def test_missing_replay_is_unavailable(handoff_copy, monkeypatch, tmp_path):
    (handoff_copy / assets.RPM_REPLAY).unlink()
    assert client.get('/api/measured/rpm/replay').status_code == 503
    monkeypatch.setenv(assets.ENV_PIMS_REPLAY, str(tmp_path / 'absent.json'))
    r = client.get('/api/measured/pims/replay')
    assert r.status_code == 503 and r.json()['detail']['code'] == 'replay_unavailable'


# ---- archived labelling and separation from the simulated ring -------------------------------------------------
def test_status_and_hardware_are_archived_reference_data():
    s = client.get('/api/measured/status').json()
    assert s['mode'] == 'ARCHIVED_MEASUREMENTS' and s['live_telemetry'] is False
    rm, pm = s['models']['mars_s_rpm']['metrics'], s['models']['iss_pims_acceleration']['metrics']
    assert round(rm['one_step_rmse']['mlp_residual'], 6) == 0.245751 and round(rm['one_step_rmse']['persistence'], 6) == 0.474178
    assert round(pm['one_step_rmse_micro_g']['mlp_residual'], 6) == 0.170533 and round(pm['one_step_rmse_micro_g']['persistence'], 6) == 0.225622
    assert pm['one_step_mae_micro_g']['ridge_residual'] < pm['one_step_mae_micro_g']['mlp_residual'] and 'not evidence of neural superiority' in pm['comparison_note']
    assert 'MHU-2' in s['evaluation_note'] and 'new, reserved evaluation dataset' in s['evaluation_note']
    iss = client.get('/api/measured/hardware').json()['profiles']['iss_host_station']
    assert (iss['solar_array_wingspan_m'], iss['solar_array_longitudinal_extent_m'], iss['truss_length_m'],
            iss['pressurized_module_major_axis_m'], iss['reference_mass_kg']) == (109, 73, 94, 67, 419725)


def test_simulated_ring_and_dynamics_selection_unchanged():
    pointer = project_root() / 'outputs' / 'runs' / 'selected_dynamics.json'
    before = pointer.read_bytes()
    for url in ('/api/measured/status', '/api/measured/rpm/replay', '/api/models'):
        assert client.get(url).status_code == 200
    rpm_post([77.8994] * 8)
    assert pointer.read_bytes() == before and json.loads(before)['run_id'] == 'dyn-v2-residual-e200-p20'
    models = client.get('/api/models').json()
    assert models['learned_dynamics_predictor']['data_type'] == 'SIMULATED_SPACECRAFT_DYNAMICS'
    assert models['measured_iss']['mars_s_rpm']['file'] != models['learned_dynamics_predictor']['checkpoint']
    sim = client.post('/api/simulate', json={'duration_s': 120, 'dt_s': 2, 'max_points': 50}).json()
    assert sim['data_type'] == 'SYNTHETIC_SPACECRAFT_DYNAMICS' and sim['parameters']['radius_m'] == 100


def test_viewer_has_labelled_archived_mode_and_keeps_ring():
    html = client.get('/').text
    for needle in ('id="iss-mode"', 'id="ring-mode"', 'data-mode="iss"', 'Hypothetical ring · simulation',
                   'ARCHIVED MEASUREMENTS — NOT LIVE TELEMETRY', 'MHU-2, recorded 17 Aug – 16 Sep 2017',
                   'recorded 1–11 Nov 2026 UTC', 'Derived nominal radial g', 'not a self-running future trajectory',
                   'SCHEMATIC · NOT CAD', 'The 109 m span is not a ring radius', 'µg = g × 10⁶', 'Frame: SSAnalysis',
                   '16 s cadence', 'iss_measured.js', 'id="pred-panel"', 'id="space"'):
        assert needle in html, needle
    js = client.get('/iss_measured.js')
    assert js.status_code == 200
    text = js.content.decode('utf-8')  # served without a charset; browsers use the page's UTF-8
    for needle in ('/api/measured/rpm/replay', '/api/measured/pims/replay', '/api/measured/hardware', 'invalid_negative_rpm',
                   'ARCHIVED MEASUREMENTS · NOT LIVE', 'of the measured RPM for visibility', 'Numbers show the recorded RPM',
                   'solar_array_wingspan_m'):
        assert needle in text, needle
