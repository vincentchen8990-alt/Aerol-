"""Verify the archived measured-ISS integration and print a JSON report (exit code 1 on any failure).

Checks
1. Every file in real_spacecraft_measurements/CORE_TRANSFER_MANIFEST.json matches its SHA256/bytes.
2. The project's NumPy inference reproduces both exported inference_test_vectors.json files and the
   handoff's own reference implementations (rpm_predict.py, outputs/pims/inference.py).
3. If the MHU-2 source archive has been restored to data/raw/mhu/, its SHA256 matches the recorded
   download manifest, every replay-clip measurement equals the archive (keep-last duplicate policy),
   every clip gap is a genuinely missing second in the archive, and the exporter's forecasts at clip
   starts are reproduced from the archive's preceding measured seconds.
4. If NASA PIMS data were reacquired, every file matches the original source manifest and the
   PIMS replay rows are exact rows of the reacquired table.
No training is run; rpm_train.py / pims_train.py are never executed.

    python -m scripts.verify_measured_iss
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
import zipfile
from datetime import datetime, timedelta

import numpy as np

from gravity_ai.data.catalog_loader import project_root
from gravity_ai.measured import assets
from gravity_ai.measured.inference import PimsModel, RpmModel


def _ref_module(name, rel):
    """Load a handoff reference script in memory (no __pycache__ written into the package)."""
    path = assets.measured_root() / rel
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def check_manifest():
    root = assets.measured_root()
    files = json.loads((root / assets.MANIFEST).read_text(encoding='utf-8'))['files']
    bad = []
    for f in files:
        path = root / f['path']
        data = path.read_bytes() if path.is_file() else None
        if data is None or len(data) != f['bytes'] or hashlib.sha256(data).hexdigest() != f['sha256']:
            bad.append(f['path'])
    return {'files_listed': len(files), 'mismatched_or_missing': bad, 'passed': not bad}


def check_parity():
    root = assets.measured_root()
    rpm_doc = assets.load_verified(assets.RPM_MODEL)[0]
    rpm = RpmModel(rpm_doc)
    tv = assets.load_verified(assets.RPM_TEST_VECTORS)[0]
    expected = np.asarray(tv['mlp_next_rpm'])
    single = np.array([rpm.predict_raw(h)[0] for h in tv['histories_rpm']])
    batch = rpm.predict_raw(tv['histories_rpm'])
    ref = _ref_module('handoff_rpm_predict', 'rpm_predict.py')
    replay = assets.load_verified(assets.RPM_REPLAY)[0]
    hist = np.array([[p['measured_rpm'] for p in c['points']][i - 8:i] for c in replay['clips'] for i in range(8, len(c['points']))])
    ref_diff = float(np.max(np.abs(rpm.predict_raw(hist) - ref.predict(rpm_doc, hist))))

    pims_doc = assets.load_verified(assets.PIMS_MODEL)[0]
    pims = PimsModel(pims_doc)
    pv = assets.load_verified(assets.PIMS_TEST_VECTORS)[0]
    ph = np.array([e['history_g'] for e in pv['examples']])
    pe = np.array([e['expected_next_g'] for e in pv['examples']])
    pref = _ref_module('handoff_pims_inference', 'outputs/pims/inference.py')
    out = {
        'rpm_test_vectors': len(expected),
        'rpm_single_max_abs_diff_rpm': float(np.max(np.abs(single - expected))),
        'rpm_batch_max_abs_diff_rpm': float(np.max(np.abs(batch - expected))),
        'rpm_vs_handoff_rpm_predict_on_replay_histories': {'histories': len(hist), 'max_abs_diff_rpm': ref_diff},
        'pims_test_vectors': len(pe),
        'pims_batch_max_abs_diff_g': float(np.max(np.abs(pims.predict(ph) - pe))),
        'pims_single_max_abs_diff_g': float(max(np.max(np.abs(pims.predict(h)[0] - e)) for h, e in zip(ph, pe))),
        'pims_vs_handoff_inference_py_max_abs_diff_g': float(np.max(np.abs(pref.predict_batch(pims_doc, ph) - pims.predict(ph)))),
    }
    out['passed'] = (out['rpm_single_max_abs_diff_rpm'] < 1e-10 and out['rpm_batch_max_abs_diff_rpm'] < 1e-10
                     and ref_diff < 1e-10 and out['pims_batch_max_abs_diff_g'] < 1e-18 and out['pims_single_max_abs_diff_g'] < 1e-18
                     and out['pims_vs_handoff_inference_py_max_abs_diff_g'] < 1e-18)
    return out


def check_mhu2_archive():
    root = assets.measured_root()
    replay = assets.load_verified(assets.RPM_REPLAY)[0]
    archive = root / replay['source_archive']
    if not archive.is_file():
        return {'skipped': f'{replay["source_archive"]} not restored locally', 'passed': None}
    manifest = assets.load_verified(assets.MHU_DOWNLOAD_MANIFEST)[0]
    record = next(f for f in manifest['files'] if f['filename'] == archive.name)
    data = archive.read_bytes()
    sha_ok = hashlib.sha256(data).hexdigest() == record['sha256'] and len(data) == record['bytes']
    import pandas as pd  # only needed for this optional raw-archive check
    with zipfile.ZipFile(archive) as z, z.open('SupplementaryData13.txt') as handle:
        frame = pd.read_csv(handle, sep='\t', usecols=['Date', 'RPM'])
    frame['timestamp'] = pd.to_datetime(frame['Date'], format='mixed')
    frame = frame.drop_duplicates('timestamp', keep='last')
    lookup = dict(zip(frame.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%S'), frame.RPM.astype(float)))
    model = RpmModel(assets.load_verified(assets.RPM_MODEL)[0])
    value_mismatch, gap_not_missing, start_checks, start_max = [], [], 0, 0.0
    for clip in replay['clips']:
        pts = clip['points']
        for p in pts:
            if lookup.get(p['timestamp_as_recorded']) != p['measured_rpm']:
                value_mismatch.append(p['timestamp_as_recorded'])
        for a, b in zip(pts, pts[1:]):
            ta, tb = datetime.fromisoformat(a['timestamp_as_recorded']), datetime.fromisoformat(b['timestamp_as_recorded'])
            for s in range(1, int((tb - ta).total_seconds())):
                if (ta + timedelta(seconds=s)).isoformat() in lookup:
                    gap_not_missing.append((ta + timedelta(seconds=s)).isoformat())
        for p in pts[:8]:
            if p.get('forecast_for_this_timestamp_rpm') is None:
                continue
            t = datetime.fromisoformat(p['timestamp_as_recorded'])
            hist = [lookup.get((t - timedelta(seconds=8 - j)).isoformat()) for j in range(8)]
            if None in hist:
                value_mismatch.append('missing archive history before ' + p['timestamp_as_recorded'])
                continue
            start_max = max(start_max, abs(float(model.predict_raw(hist)[0]) - p['forecast_for_this_timestamp_rpm']))
            start_checks += 1
    return {'archive': replay['source_archive'], 'sha256_matches_download_manifest': sha_ok,
            'archive_unique_timestamps': len(lookup),
            'clip_points_checked': sum(len(c['points']) for c in replay['clips']), 'clip_value_mismatches': value_mismatch,
            'clip_gap_seconds_present_in_archive': gap_not_missing,
            'clip_start_forecasts_reproduced_from_archive_history': start_checks, 'clip_start_forecast_max_abs_diff_rpm': start_max,
            'passed': sha_ok and not value_mismatch and not gap_not_missing and start_max < 1e-10}


def check_pims_reacquisition():
    base = project_root() / 'real_spacecraft_measurements' / 'reacquired' / 'pims'
    new_manifest = base / 'data' / 'raw' / 'pims' / 'source_manifest.json'
    if not new_manifest.is_file():
        return {'skipped': 'no reacquired NASA PIMS data', 'passed': None}
    from scripts.build_pims_replay import SPLIT_MANIFEST, compare_source_files
    original = assets.load_verified(assets.PIMS_SOURCE_MANIFEST)[0]
    comparison = compare_source_files(original, json.loads(new_manifest.read_text(encoding='utf-8')))
    table = base / 'data' / 'processed' / 'iss_mams_acceleration.csv'
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    expected = assets.load_verified(SPLIT_MANIFEST)[0]['dataset_sha256']
    rows = {}
    with table.open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            rows[r['timestamp_utc']] = (float(r['ax_g']), float(r['ay_g']), float(r['az_g']), r['segment_id'])
    replay_path = assets.pims_replay_path()
    replay_mismatch, checked = [], 0
    if replay_path.is_file():
        for clip in json.loads(replay_path.read_text(encoding='utf-8'))['clips']:
            for p in clip['points']:
                checked += 1
                if rows.get(p['timestamp_utc']) != (p['ax_g'], p['ay_g'], p['az_g'], p['segment_id']):
                    replay_mismatch.append(p['timestamp_utc'])
    ok = not (comparison['sha256_mismatches'] or comparison['missing_from_reacquisition'] or comparison['extra_in_reacquisition'])
    return {'source_files_compared': comparison['original_files'], 'source_sha256_mismatches': comparison['sha256_mismatches'],
            'missing': comparison['missing_from_reacquisition'], 'extra': comparison['extra_in_reacquisition'],
            'processed_table_sha256': digest, 'matches_training_dataset_sha256': digest == expected,
            'table_rows': len(rows), 'replay_points_checked': checked, 'replay_points_not_in_table': replay_mismatch,
            'passed': ok and digest == expected and not replay_mismatch and checked > 0}


def main():
    report = {'handoff_manifest': check_manifest(), 'inference_parity': check_parity(),
              'mhu2_source_archive': check_mhu2_archive(), 'pims_reacquisition': check_pims_reacquisition()}
    selected = project_root() / 'outputs' / 'runs' / 'selected_dynamics.json'
    report['simulated_dynamics_selection'] = json.loads(selected.read_text(encoding='utf-8')) if selected.is_file() else None
    print(json.dumps(report, indent=2))
    return 1 if any(v.get('passed') is False for v in report.values() if isinstance(v, dict)) else 0


if __name__ == '__main__':
    sys.exit(main())
