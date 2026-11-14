"""Acquire real ISS MAMS OSS-bias/TMF accelerations and parse NASA PAD files.

This is measured ISS disturbance acceleration, NOT rotating-habitat command
response. NASA's PADprimer timing rules are applied: ignore the embedded time
column and filename end time; use the first filename start and header cadence;
append '+' files continuously; start a new, grid-aligned segment at '-' files.
No missing acceleration values are invented and no window may cross a segment
or a date-based split. All transfers are from the public NASA GRC archive.
"""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
BASE = "https://gipoc.grc.nasa.gov/pims/pub/pad"
SENSOR = "mams_accel_ossbtmf"
STAMP = r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.\d{3}"
FILE_RE = re.compile(rf"(?P<start>{STAMP})(?P<marker>[+-])(?P<end>{STAMP})\.ossbtmf")
SOURCES = {
    "archive": "https://gipoc.grc.nasa.gov/wp/pims/acceleration-archives/",
    "schema": "https://gipoc.grc.nasa.gov/pims/plots/user/mohan/how2read_pad_binary_files.html",
    "timing": "https://gipoc.grc.nasa.gov/pims/pub/plots/user/mohan/PADprimer.pdf",
    "instrument": "https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/acceleration-measurements/",
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def get_bytes(url):
    error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "ISS-MAMS-research-data-audit/1.0"})
            with urlopen(request, timeout=35) as response:
                data = response.read(5_000_001)
                if len(data) > 5_000_000:
                    raise ValueError(f"Unexpectedly large response: {url}")
                return data
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to retrieve {url}: {error}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def day_url(day):
    return f"{BASE}/year{day.year}/month{day.month:02d}/day{day.day:02d}/{SENSOR}/"


def list_day(day):
    url = day_url(day)
    document = get_bytes(url)
    parser = LinkParser()
    parser.feed(document.decode("utf-8", errors="replace"))
    names = sorted({
        unquote(urlsplit(link).path).rsplit("/", 1)[-1]
        for link in parser.links
        if FILE_RE.fullmatch(unquote(urlsplit(link).path).rsplit("/", 1)[-1])
    })
    if not names:
        raise RuntimeError(f"No actual .ossbtmf data files listed for {day}: {url}")
    return {
        "date": day.isoformat(), "directory_url": url,
        "listing_sha256": digest(document), "files": names,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def fetch_pair(task):
    listing, name, raw_root = task
    folder = raw_root / listing["date"]
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for suffix, kind in [("", "binary"), (".header", "header")]:
        filename = name + suffix
        url = listing["directory_url"] + quote(filename, safe="")
        path = folder / filename
        if not path.is_file() or not path.stat().st_size:
            data = get_bytes(url)
            temporary = path.with_name(path.name + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        else:
            data = path.read_bytes()
        if data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError(f"HTML returned instead of {kind}: {url}")
        entries.append({
            "kind": kind, "filename": filename,
            "relative_path": str(path.relative_to(ROOT)),
            "source_url": url, "bytes": len(data), "sha256": digest(data),
            "retrieved_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        })
    return {"name": name, "source_date": listing["date"], "files": entries}


def parse_stamp(text):
    return datetime.strptime(text, "%Y_%m_%d_%H_%M_%S.%f").replace(tzinfo=timezone.utc)


def parse_pair(pair):
    binary_entry, header_entry = pair["files"]
    raw = (ROOT / binary_entry["relative_path"]).read_bytes()
    header = ET.fromstring((ROOT / header_entry["relative_path"]).read_bytes())
    if header.tag != SENSOR or header.findtext("SensorID") != "ossbtmf":
        raise ValueError(f"Wrong sensor in {pair['name']}")
    quality = (header.findtext("DataQualityMeasure") or "").strip()
    if quality != "Valid":
        raise ValueError(f"Non-valid source quality {quality!r}: {pair['name']}")
    gdata = header.find("GData")
    if gdata is None or gdata.attrib.get("file") != pair["name"]:
        raise ValueError(f"Header/binary identity mismatch: {pair['name']}")
    if gdata.attrib.get("format") != "binary 32 bit IEEE float little endian":
        raise ValueError(f"Unsupported source binary format: {pair['name']}")
    if len(raw) == 0 or len(raw) % 16:
        raise ValueError(f"Binary length is not a positive multiple of 4 float32s: {pair['name']}")
    rows = list(struct.iter_unpack("<4f", raw))
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"Nonfinite data require explicit handling: {pair['name']}")

    # Independent float decoder checks exact first-20-record interpretation.
    other = array("f")
    other.frombytes(raw[:20 * 16])
    if sys.byteorder != "little":
        other.byteswap()
    expected = [value for row in rows[:20] for value in row]
    if list(other) != expected:
        raise AssertionError("Independent float decoders disagree")

    fs = float(header.findtext("SampleRate"))
    if fs != 0.0625:
        raise ValueError(f"This acquisition expects 0.0625 Hz, observed {fs}")
    match = FILE_RE.fullmatch(pair["name"])
    if not match:
        raise ValueError("Unexpected filename format")
    pair.update({
        "records": len(rows), "sample_rate_hz": fs,
        "cutoff_frequency_hz": float(header.findtext("CutoffFreq")),
        "source_quality": quality,
        "sensor_coordinate_system": header.find("SensorCoordinateSystem").attrib,
        "data_coordinate_system": header.find("DataCoordinateSystem").attrib,
        "header_time_zero_ignored": header.findtext("TimeZero"),
        "filename_end_ignored": match.group("end"),
        "raw_relative_time_ignored_first_last": [rows[0][0], rows[-1][0]],
        "first_20_records_independent_decode_match": True,
    })
    return rows, parse_stamp(match.group("start")), match.group("marker"), quality


def build_csv(pairs, start_day, end_day, output):
    dt = 16.0
    begin = datetime.combine(start_day, datetime.min.time(), timezone.utc)
    end = datetime.combine(end_day + timedelta(days=1), datetime.min.time(), timezone.utc)
    grid_origin = None
    previous_end = None
    segment_number = -1
    segments = []
    daily_counts = {}
    row_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax_g", "ay_g", "az_g", "segment_id", "source_file", "quality"])
        for pair in sorted(pairs, key=lambda x: x["name"]):
            values, filename_start, marker, quality = parse_pair(pair)
            new_segment = previous_end is None or marker == "-"
            if new_segment:
                segment_number += 1
                if grid_origin is None:
                    grid_origin = filename_start
                    actual_start = filename_start
                else:
                    count = round((filename_start - grid_origin).total_seconds() / dt)
                    actual_start = grid_origin + timedelta(seconds=dt * count)
                    if abs((actual_start - filename_start).total_seconds()) > dt / 2 + 1e-9:
                        raise AssertionError("Grid adjustment exceeded NASA rule")
                if previous_end is not None and actual_start <= previous_end:
                    raise ValueError("New series overlaps a previous reconstructed series")
                segments.append({
                    "segment_id": f"segment_{segment_number:03d}",
                    "filename_start_utc": filename_start.isoformat(),
                    "reconstructed_start_utc": actual_start.isoformat(),
                    "grid_adjustment_s": (actual_start - filename_start).total_seconds(),
                    "missing_samples_before": None if previous_end is None else max(0, round((actual_start - previous_end).total_seconds() / dt) - 1),
                    "source_marker": marker,
                })
            else:
                actual_start = previous_end + timedelta(seconds=dt)
            pair["reconstructed_start_utc"] = actual_start.isoformat()
            pair["start_offset_from_filename_s"] = (actual_start - filename_start).total_seconds()
            pair["segment_id"] = f"segment_{segment_number:03d}"
            for i, (_, ax, ay, az) in enumerate(values):
                stamp = actual_start + timedelta(seconds=dt * i)
                if begin <= stamp < end:
                    writer.writerow([
                        stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                        format(ax, ".17g"), format(ay, ".17g"), format(az, ".17g"),
                        pair["segment_id"], pair["name"], quality,
                    ])
                    day = stamp.date().isoformat()
                    daily_counts[day] = daily_counts.get(day, 0) + 1
                    row_count += 1
            previous_end = actual_start + timedelta(seconds=dt * (len(values) - 1))
            pair["reconstructed_end_utc"] = previous_end.isoformat()
    return {"rows": row_count, "rows_by_utc_day": daily_counts, "segments": segments,
            "sha256": digest(output.read_bytes()), "bytes": output.stat().st_size}


def validate_csv(output, pairs):
    previous = None
    count = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    lows = [float("inf")] * 3
    highs = [float("-inf")] * 3
    with output.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
            if row["quality"] != "Valid":
                raise ValueError("Non-valid row in output")
            if previous is not None:
                if stamp <= previous[0]:
                    raise ValueError("Duplicate or unordered timestamp")
                if row["segment_id"] == previous[1] and (stamp - previous[0]).total_seconds() != 16.0:
                    raise ValueError("Non-contiguous samples within one output segment")
            previous = (stamp, row["segment_id"])
            values = [float(row[k]) for k in ("ax_g", "ay_g", "az_g")]
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Nonfinite output acceleration")
            for i, value in enumerate(values):
                lows[i] = min(lows[i], value)
                highs[i] = max(highs[i], value)
            if date(2026, 11, 1) <= stamp.date() <= date(2026, 11, 11):
                split_counts["train" if stamp.day < 8 else "validation" if stamp.day < 10 else "test"] += 1
            count += 1
    coordinate_count = len({json.dumps(p["data_coordinate_system"], sort_keys=True) for p in pairs})
    if coordinate_count != 1:
        raise ValueError("Data coordinate system changes require separate treatment")
    for pair in pairs:
        for item in pair["files"]:
            if digest((ROOT / item["relative_path"]).read_bytes()) != item["sha256"]:
                raise ValueError("Source SHA256 changed after decoding")
    return {
        "validated_rows": count, "fixed_split_rows_before_windowing": split_counts,
        "strictly_increasing_unique_timestamps": True,
        "within_segment_dt_s": 16.0, "all_accelerations_finite": True,
        "unique_data_coordinate_systems": coordinate_count,
        "data_coordinate_system": pairs[0]["data_coordinate_system"],
        "all_source_sha256_reverified": True,
        "source_binary_header_pairs": len(pairs),
        "source_total_bytes": sum(item["bytes"] for pair in pairs for item in pair["files"]),
        "source_total_records_before_utc_filter": sum(pair["records"] for pair in pairs),
        "axis_min_g": lows, "axis_max_g": highs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 11, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 11))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.end < args.start or args.end > datetime.now(timezone.utc).date():
        raise ValueError("Invalid date range or future observations requested")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be from 1 to 8")
    raw_root = ROOT / "data" / "raw" / "pims"
    raw_root.mkdir(parents=True, exist_ok=True)
    days = [args.start + timedelta(days=n) for n in range((args.end - args.start).days + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        listings = list(pool.map(list_day, days))
    tasks = [(listing, name, raw_root) for listing in listings for name in listing["files"]]
    print(f"Found {len(tasks)} binary/header pairs across {len(days)} days", flush=True)
    pairs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, pair in enumerate(pool.map(fetch_pair, tasks), 1):
            pairs.append(pair)
            if i % 20 == 0 or i == len(tasks):
                print(f"Downloaded and hashed {i}/{len(tasks)} pairs", flush=True)
    output = ROOT / "data" / "processed" / "iss_mams_acceleration.csv"
    summary = build_csv(pairs, args.start, args.end, output)
    manifest = {
        "dataset": "NASA ISS MAMS OSS bias-compensated trimmed-mean-filtered acceleration",
        "data_type": "MEASURED_PROCESSED_ISS_ACCELERATION_NOT_RING_COMMAND_RESPONSE",
        "sources": SOURCES, "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_utc_dates": [args.start.isoformat(), args.end.isoformat()],
        "sample_rate_hz": 0.0625, "dt_s": 16.0, "acceleration_units": "standard gravity g",
        "directory_listings": listings, "files": pairs, "processed": summary,
        "parsing_assumptions": [
            "Rows are 4 little-endian IEEE float32 values: relative time, ax_g, ay_g, az_g.",
            "NASA PADprimer explicitly requires ignoring embedded relative times and filename end times.",
            "A first series uses its first filename UTC start; subsequent plus-marked files continue at 16-second cadence.",
            "Minus-marked files start new segments, snapped to the nearest original cadence grid (at most 8 s adjustment).",
            "Gaps are represented by timestamp jumps and segment IDs, not invented or imputed acceleration values.",
            "OSS BTMF is already bias-compensated and trimmed-mean-filtered; BiasCoeff is NOT subtracted again.",
            "Only header quality Valid is accepted. Data-coordinate-system metadata is retained; no additional frame transformation applied.",
            "Rows outside requested UTC dates are excluded, including final files extending into the next day.",
            "The first requested directory may start after midnight. No preceding-day data were invented to fill that edge.",
            "No motor command, motor torque, rotating-ring speed, ring inertia, or ring gravity labels occur in this dataset.",
        ],
        "validation": {
            "official_four_column_example": "Old-host example inaccessible; new-host Examples directory returned 404. No example-based validation is claimed.",
            "equivalent_checks": "Every header/binary identity, declared format, byte divisibility, Valid quality and finite record checked. First 20 records of every binary decoded independently with struct and array, with exact agreement.",
            "postparse": validate_csv(output, pairs),
        },
    }
    manifest_path = raw_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Fixed chronological split for this requested acquisition. Keep all windows
    # within both their segment and their assigned split; fit preprocessing on train.
    if args.start == date(2026, 11, 1) and args.end == date(2026, 11, 11):
        split = {
            "dataset_sha256": summary["sha256"], "basis": "reconstructed UTC date",
            "train": {"start_inclusive": "2026-11-01T00:00:00Z", "end_exclusive": "2026-11-08T00:00:00Z"},
            "validation": {"start_inclusive": "2026-11-08T00:00:00Z", "end_exclusive": "2026-11-10T00:00:00Z"},
            "test": {"start_inclusive": "2026-11-10T00:00:00Z", "end_exclusive": "2026-11-12T00:00:00Z"},
            "requirements": ["Fit scaling only on training data", "Never create windows across segment IDs", "Require contiguous 16-second timestamps", "Never create windows spanning split boundaries", "Use validation only for model/settings selection"],
        }
        (output.parent / "iss_mams_split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
