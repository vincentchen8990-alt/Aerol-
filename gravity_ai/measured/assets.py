"""Locate and integrity-check the measured-ISS handoff files.

Every model, report and replay file served from the handoff directory is checked against its
SHA256 and byte count in CORE_TRANSFER_MANIFEST.json before use. A missing or altered file
raises AssetError; callers report it as "unavailable" and never substitute simulated values.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from gravity_ai.data.catalog_loader import project_root

ENV_ROOT = 'GRAVITY_MEASURED_ROOT'
ENV_PIMS_REPLAY = 'GRAVITY_PIMS_REPLAY'

MANIFEST = 'CORE_TRANSFER_MANIFEST.json'
HARDWARE = 'hardware_profiles.json'
RPM_MODEL = 'outputs/rpm/selected_model_with_usage_notes.json'
RPM_REPORT = 'outputs/rpm/training_report.json'
RPM_REPLAY = 'outputs/rpm/replay_mhu2.json'
RPM_TEST_VECTORS = 'outputs/rpm/inference_test_vectors.json'
PIMS_MODEL = 'outputs/pims/selected_model.json'
PIMS_REPORT = 'outputs/pims/training_report.json'
PIMS_TEST_VECTORS = 'outputs/pims/inference_test_vectors.json'
PIMS_SOURCE_MANIFEST = 'data/raw/pims/source_manifest.json'
MHU_DOWNLOAD_MANIFEST = 'data/raw/mhu/download_manifest.json'
# Derived by scripts/build_pims_replay.py from the reacquired NASA table (not part of the handoff).
PIMS_REPLAY_DEFAULT = Path('outputs') / 'measured_iss' / 'pims_replay_nov2026.json'


class AssetError(Exception):
    """A required measured-data file is missing, unlisted, altered or unreadable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def detail(self) -> dict:
        return {'code': self.code, 'message': str(self)}


def measured_root() -> Path:
    value = os.environ.get(ENV_ROOT)
    return Path(value) if value else project_root() / 'real_spacecraft_measurements'


def pims_replay_path() -> Path:
    value = os.environ.get(ENV_PIMS_REPLAY)
    return Path(value) if value else project_root() / PIMS_REPLAY_DEFAULT


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_cache: dict = {}


def _stat_key(path: Path):
    st = path.stat()
    return str(path.resolve()), st.st_mtime_ns, st.st_size


def manifest_entries(root: Path | None = None) -> dict:
    root = measured_root() if root is None else root
    path = root / MANIFEST
    if not path.is_file():
        raise AssetError('manifest_missing', f'{MANIFEST} not found in {display_path(root)}')
    key = ('manifest',) + _stat_key(path)
    if key not in _cache:
        try:
            files = json.loads(path.read_text(encoding='utf-8'))['files']
            _cache[key] = {f['path']: f for f in files}
        except (ValueError, KeyError, TypeError) as e:
            raise AssetError('manifest_invalid', f'{MANIFEST} could not be read: {e}') from e
    return _cache[key]


def load_verified(rel: str, root: Path | None = None) -> tuple[dict, dict]:
    """Return (parsed JSON, provenance) for a handoff file whose SHA256 matches the manifest."""
    root = measured_root() if root is None else root
    path = root / rel
    if not path.is_file():
        raise AssetError('file_missing', f'{rel} not found in {display_path(root)}')
    manifest_path = root / MANIFEST
    key = ('verified',) + _stat_key(path) + (_stat_key(manifest_path) if manifest_path.is_file() else None,)
    if key in _cache:
        return _cache[key]
    entry = manifest_entries(root).get(rel)
    if entry is None:
        raise AssetError('not_in_manifest', f'{rel} is not listed in {MANIFEST}')
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != entry['sha256'] or len(data) != entry['bytes']:
        raise AssetError('hash_mismatch', f'{rel} does not match {MANIFEST} (sha256 {digest[:12]}..., '
                                          f'{len(data)} bytes; expected {entry["sha256"][:12]}..., {entry["bytes"]} bytes)')
    try:
        doc = json.loads(data.decode('utf-8'))
    except ValueError as e:
        raise AssetError('invalid_json', f'{rel} is not valid JSON: {e}') from e
    info = {'file': rel, 'root': display_path(root), 'sha256': digest, 'bytes': len(data),
            'verified_against': MANIFEST}
    _cache[key] = (doc, info)
    return doc, info


def file_sha256_cached(path: Path) -> str | None:
    """SHA256 of an optional large file (e.g. a restored raw archive), cached by mtime/size."""
    if not path.is_file():
        return None
    key = ('sha',) + _stat_key(path)
    if key not in _cache:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                h.update(block)
        _cache[key] = h.hexdigest()
    return _cache[key]
