"""Build the small NASA PIMS replay used by the archived-measurement viewer.

Reads the processed table written by real_spacecraft_measurements/pims_acquire.py when it was run
with its output root redirected (so the originally recorded manifests stay unchanged), checks that:

* every reacquired binary/header file has the SHA256 recorded in the ORIGINAL source manifest, and
* the processed table is byte-identical to the dataset the model was trained and tested on
  (dataset_sha256 in the original split manifest),

then writes two short clips of real 16-second samples to outputs/measured_iss/pims_replay_nov2026.json.
No value is interpolated, resampled, filtered or generated.

    python -m scripts.build_pims_replay [--reacquired real_spacecraft_measurements/reacquired/pims]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gravity_ai.data.catalog_loader import project_root
from gravity_ai.measured import assets

TEST_CLIP_SAMPLES = 150          # 40 minutes of the held-out test split
GAP_CLIP_BEFORE, GAP_CLIP_AFTER = 40, 60
SPLIT_MANIFEST = 'data/processed/iss_mams_split_manifest.json'


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def compare_source_files(original: dict, reacquired: dict) -> dict:
    # pims_acquire.py records OS-native separators (backslashes when run on Windows).
    key = lambda f: f['relative_path'].replace('\\', '/')  # noqa: E731
    old = {key(f): f for pair in original['files'] for f in pair['files']}
    new = {key(f): f for pair in reacquired['files'] for f in pair['files']}
    mismatched = sorted(p for p in old.keys() & new.keys()
                        if (old[p]['sha256'], old[p]['bytes']) != (new[p]['sha256'], new[p]['bytes']))
    return {'original_files': len(old), 'reacquired_files': len(new), 'missing_from_reacquisition': sorted(old.keys() - new.keys()),
            'extra_in_reacquisition': sorted(new.keys() - old.keys()), 'sha256_mismatches': mismatched}


def read_table(path: Path, split: dict) -> list[dict]:
    bounds = [(name, _utc(split[name]['start_inclusive']), _utc(split[name]['end_exclusive']))
              for name in ('train', 'validation', 'test')]
    rows = []
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            stamp = _utc(row['timestamp_utc'])
            name = next((n for n, lo, hi in bounds if lo <= stamp < hi), None)
            if name is None or row['quality'] != 'Valid':
                raise ValueError(f'row outside the recorded splits or not Valid: {row}')
            rows.append({'timestamp_utc': row['timestamp_utc'], 'ax_g': float(row['ax_g']), 'ay_g': float(row['ay_g']),
                         'az_g': float(row['az_g']), 'segment_id': row['segment_id'], 'split': name,
                         'source_file': row['source_file']})
    return rows


def select_clips(rows: list[dict]) -> list[dict]:
    test = [r for r in rows if r['split'] == 'test']
    held_out = test[:TEST_CLIP_SAMPLES]
    if len({r['segment_id'] for r in held_out}) != 1:
        raise ValueError('expected the first held-out test samples to form one contiguous segment')
    clips = [{'name': 'held_out_test_start', 'label': 'Held-out test split', 'split': 'test',
              'description': f'First {len(held_out)} samples of the held-out test days (10-11 Nov 2026 UTC).',
              'points': held_out}]
    # A real recorded gap (segment boundary) with enough samples on both sides inside one split.
    # Prefer the validation split (not used for fitting) over the training split.
    candidates = []
    for i in range(1, len(rows)):
        if rows[i]['segment_id'] == rows[i - 1]['segment_id']:
            continue
        window = rows[i - GAP_CLIP_BEFORE:i + GAP_CLIP_AFTER]
        if i >= GAP_CLIP_BEFORE and len(window) == GAP_CLIP_BEFORE + GAP_CLIP_AFTER and len({r['split'] for r in window}) == 1 \
                and len({r['segment_id'] for r in window}) == 2:
            candidates.append((i, window))
    if candidates:
        i, window = next((c for c in candidates if c[1][0]['split'] == 'validation'), candidates[0])
        missing = round((_utc(rows[i]['timestamp_utc']) - _utc(rows[i - 1]['timestamp_utc'])).total_seconds() / 16) - 1
        clips.append({'name': 'recorded_gap', 'label': f'Recorded gap ({window[0]["split"]} split)', 'split': window[0]['split'],
                      'description': f'Real segment boundary at {rows[i]["timestamp_utc"]}: {missing} missing 16-s samples '
                                     'in the NASA record; history restarts after it.',
                      'points': window})
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reacquired', type=Path, default=project_root() / 'real_spacecraft_measurements' / 'reacquired' / 'pims')
    parser.add_argument('--output', type=Path, default=project_root() / assets.PIMS_REPLAY_DEFAULT)
    args = parser.parse_args()

    split, _ = assets.load_verified(SPLIT_MANIFEST)
    original, original_info = assets.load_verified(assets.PIMS_SOURCE_MANIFEST)
    new_manifest_path = args.reacquired / 'data' / 'raw' / 'pims' / 'source_manifest.json'
    table = args.reacquired / 'data' / 'processed' / 'iss_mams_acceleration.csv'
    reacquired = json.loads(new_manifest_path.read_text(encoding='utf-8'))
    comparison = compare_source_files(original, reacquired)
    if comparison['sha256_mismatches'] or comparison['missing_from_reacquisition'] or comparison['extra_in_reacquisition']:
        raise SystemExit(f'Reacquired NASA files differ from the original source manifest: {json.dumps(comparison)[:2000]}')
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    if digest != split['dataset_sha256']:
        raise SystemExit(f'Reacquired table sha256 {digest} differs from the training dataset {split["dataset_sha256"]}; '
                         'refusing to build a replay from different data.')
    rows = read_table(table, split)
    clips = select_clips(rows)
    first = original['files'][0]
    names = sorted({r['source_file'] for c in clips for r in c['points']})
    by_name = {p['name']: p for p in original['files']}
    out = {
        'description': 'Real NASA PIMS MAMS OSS BTMF 16-second samples for the archived-measurement viewer',
        'data_kind': 'ARCHIVED_MEASUREMENTS',
        'instrument': 'NASA MAMS OSS bias-compensated trimmed-mean-filtered acceleration (sensor ossbtmf)',
        'recorded_dates_utc': original['requested_utc_dates'],
        'coordinate_frame': original['validation']['postparse']['data_coordinate_system']['name'],
        'coordinate_frame_comment': original['validation']['postparse']['data_coordinate_system'].get('comment'),
        'sensor_location_comment': first['sensor_coordinate_system'].get('comment'),
        'units': 'g', 'cadence_s': 16.0,
        'display_rule': ('A timestamp step larger than 16 s is a recorded gap: never interpolated, and history restarts '
                         'after it. Display micro-g only as g x 1e6. No values exist between 16-s samples.'),
        'provenance': {
            'archive': original['sources']['archive'], 'timing_rules': original['sources']['timing'],
            'processed_table_sha256': digest, 'matches_training_dataset_sha256': True,
            'original_source_manifest': f'real_spacecraft_measurements/{assets.PIMS_SOURCE_MANIFEST} (unchanged; sha256 {original_info["sha256"]})',
            'reacquired_source_manifest': assets.display_path(new_manifest_path),
            'reacquired_at_utc': reacquired.get('acquired_at_utc'),
            'reacquired_files_matching_original_sha256': comparison['original_files'],
            'built_by': 'scripts/build_pims_replay.py', 'built_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'source_files_in_clips': {n: {'binary_sha256': by_name[n]['files'][0]['sha256'],
                                          'source_url': by_name[n]['files'][0]['source_url']} for n in names},
        },
        'clips': [{**c, 'points': [{k: r[k] for k in ('timestamp_utc', 'ax_g', 'ay_g', 'az_g', 'segment_id', 'split', 'source_file')}
                                   for r in c['points']]} for c in clips],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': assets.display_path(args.output), 'rows_in_table': len(rows), 'table_sha256': digest,
                      'clips': [{'name': c['name'], 'points': len(c['points']), 'first': c['points'][0]['timestamp_utc'],
                                 'last': c['points'][-1]['timestamp_utc']} for c in clips],
                      'source_comparison': {k: (len(v) if isinstance(v, list) else v) for k, v in comparison.items()}}, indent=2))


if __name__ == '__main__':
    main()
