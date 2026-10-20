#!/usr/bin/env python3
"""Download and validate the CDC ILINet snapshot used by BIOSTAT 707 ILI demo.

The underlying data originate from CDC ILINet. The demo obtained them through
CMU Delphi's public legacy FluView V3 endpoint, and this script intentionally
uses that *same* published endpoint so its schema remains compatible with
``src/data.py``. No API key or third-party Python packages are required.

Usage, from the repository root (with this file under scripts/):
    python scripts/download_ili_data.py
    python scripts/download_ili_data.py --verify-only
    python scripts/download_ili_data.py --force
    python scripts/download_ili_data.py --require-demo-sha

The endpoint may revise historical values. Without --force, an existing and
valid snapshot is never overwritten. --require-demo-sha refuses an online
response that does not exactly match the original v0.1 demo's raw bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
FIRST_WEEK = 201040
LAST_WEEK = 201939
RAW_RELATIVE = Path("data/raw/fluview_201040_201939.json")
MANIFEST_RELATIVE = Path("data/source_manifest.json")
SOURCE_URL = (
    "https://api.delphi.cmu.edu/epidata/fluview/"
    "?regions=" + ",".join(REGIONS) + "&epiweeks=201040-201939"
)
LICENSE_URL = "https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html"
DEMO_SHA256 = "f194ed2906f5ce5a3e404d180a574178d1f9285819f5ed4289f47f26aba50cfe"
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
REQUIRED_FIELDS = {"region", "epiweek", "wili", "release_date", "issue"}


def mmwr_week_one_start(year: int) -> date:
    """First Sunday of CDC/MMWR year, the Sunday in the week containing Jan 4."""
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=(jan4.weekday() + 1) % 7)


def expected_epiweeks() -> list[int]:
    """CDC/MMWR epiweeks including week 53 when present, not ISO weeks."""
    weeks = []
    for year in range(FIRST_WEEK // 100, LAST_WEEK // 100 + 1):
        n_weeks = (mmwr_week_one_start(year + 1) - mmwr_week_one_start(year)).days // 7
        for week in range(1, n_weeks + 1):
            code = year * 100 + week
            if FIRST_WEEK <= code <= LAST_WEEK:
                weeks.append(code)
    return weeks


def validate_response(raw_bytes: bytes) -> tuple[dict, list[dict], dict]:
    """Fail closed on API error, duplicates, coverage gaps, or invalid targets."""
    try:
        obj = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Source response is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON response is not an object")
    if obj.get("result") != 1 or obj.get("message") != "success":
        raise ValueError(f"Source API reported an error: result={obj.get('result')!r}, "
                         f"message={obj.get('message')!r}")
    records = obj.get("epidata")
    if not isinstance(records, list) or not records:
        raise ValueError("Response contains no epidata list")
    expected = expected_epiweeks()
    expected_pairs = {(r, w) for r in REGIONS for w in expected}
    if len(records) != len(expected_pairs):
        raise ValueError(f"Expected {len(expected_pairs)} region-week rows, received {len(records)}")
    pairs = set()
    by_region = Counter()
    for i, row in enumerate(records):
        if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
            raise ValueError(f"Row {i}: missing fields {sorted(REQUIRED_FIELDS - set(row)) if isinstance(row, dict) else 'all'}")
        region, week, wili = row["region"], row["epiweek"], row["wili"]
        if region not in REGIONS or type(week) is not int:
            raise ValueError(f"Row {i}: unexpected region or epiweek")
        pair = (region, week)
        if pair in pairs:
            raise ValueError(f"Duplicate region-week: {pair}")
        pairs.add(pair)
        by_region[region] += 1
        if isinstance(wili, bool) or not isinstance(wili, (int, float)):
            raise ValueError(f"Row {i}: nonnumeric wili")
        if not math.isfinite(wili) or not (0 <= wili <= 100):
            raise ValueError(f"Row {i}: invalid wili={wili}")
    if pairs != expected_pairs:
        missing = sorted(expected_pairs - pairs)[:5]
        extra = sorted(pairs - expected_pairs)[:5]
        raise ValueError(f"Wrong weekly/region coverage; missing examples={missing}; extra examples={extra}")
    summary = {
        "row_count": len(records),
        "region_counts": dict(sorted(by_region.items())),
        "first_week": expected[0],
        "last_week": expected[-1],
        "contains_2014w53": 201453 in expected,
        "min_wili": min(row["wili"] for row in records),
        "max_wili": max(row["wili"] for row in records),
    }
    return obj, records, summary


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory temporary file prevents partially downloaded JSON files.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def fetch_source() -> tuple[bytes, int]:
    request = Request(SOURCE_URL, headers={"User-Agent": "BIOSTAT707-ILI-Demo/0.1 (educational research)",
                                           "Accept": "application/json"})
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=90) as response:
                status = response.status
                if status != 200:
                    raise ValueError(f"Source returned HTTP {status}, expected 200")
                content = response.read(MAX_RESPONSE_BYTES + 1)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds allowed size (20 MiB)")
                return content, status
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(attempt)
    raise RuntimeError("Unreachable retry state")


def verify_existing(raw_path: Path, manifest_path: Path, require_demo_sha: bool) -> dict:
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Raw JSON and manifest must both exist; rerun with --force to redownload")
    raw_bytes = raw_path.read_bytes()
    obj, records, summary = validate_response(raw_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(raw_bytes).hexdigest()
    if sha != manifest.get("sha256"):
        raise ValueError("Existing raw file does not match source_manifest.json SHA-256")
    if manifest.get("url") != SOURCE_URL or manifest.get("http_status") != 200:
        raise ValueError("Existing manifest URL or HTTP status does not match original source")
    if manifest.get("rows") != len(records):
        raise ValueError("Existing manifest row count does not match raw JSON")
    if manifest.get("bytes") != len(raw_bytes):
        raise ValueError("Existing manifest byte count does not match raw JSON")
    if require_demo_sha and sha != DEMO_SHA256:
        raise ValueError(f"Existing snapshot SHA {sha} differs from demo SHA {DEMO_SHA256}")
    print(f"PASS (cached, no network): {summary['row_count']} rows, "
          f"{len(REGIONS)} regions, {len(expected_epiweeks())} weeks/region")
    print(f"SHA-256: {sha}")
    if sha != DEMO_SHA256:
        print("WARNING: Snapshot differs from demo v0.1; do not assume identical model results.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="Repository root; default is parent of scripts/ (where this file belongs)")
    parser.add_argument("--force", action="store_true", help="Redownload and replace existing raw JSON and manifest")
    parser.add_argument("--verify-only", action="store_true", help="Verify cached data and manifest without accessing the network")
    parser.add_argument("--require-demo-sha", action="store_true", help="Require exactly the original demo v0.1 snapshot SHA-256")
    args = parser.parse_args(argv)
    if args.force and args.verify_only:
        parser.error("--force and --verify-only cannot be used together")
    root = args.project_root.resolve()
    raw_path = root / RAW_RELATIVE
    manifest_path = root / MANIFEST_RELATIVE
    try:
        if args.verify_only or (raw_path.exists() and not args.force):
            verify_existing(raw_path, manifest_path, args.require_demo_sha)
            return 0
        print(f"Downloading CDC ILINet via CMU Delphi: {SOURCE_URL}")
        content, status = fetch_source()
        obj, records, summary = validate_response(content)
        sha = hashlib.sha256(content).hexdigest()
        if args.require_demo_sha and sha != DEMO_SHA256:
            raise ValueError(f"Downloaded snapshot SHA {sha} differs from demo SHA {DEMO_SHA256}; "
                             "old data left unchanged")
        manifest = {
            "url": SOURCE_URL,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "bytes": len(content),
            "sha256": sha,
            "fields": sorted(set().union(*(set(r) for r in records))),
            "rows": len(records),
            "api_result": obj["result"],
            "api_message": obj["message"],
            "source": "CDC ILINet via CMU Delphi",
            "license": "Public Domain US Government",
            "license_reference": LICENSE_URL,
            "snapshot_path": RAW_RELATIVE.as_posix(),
            "matches_demo_v0_1_sha256": sha == DEMO_SHA256,
            "validation": summary,
        }
        # Both writes are atomic individually; SHA validation detects an
        # interruption between them instead of silently using mixed versions.
        atomic_write(raw_path, content)
        atomic_write(manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(f"PASS (downloaded): {summary['row_count']} rows, 10 HHS regions, "
              f"{len(expected_epiweeks())} weeks/region")
        print(f"Raw data: {raw_path}")
        print(f"Manifest: {manifest_path}")
        print(f"SHA-256: {sha}")
        if sha != DEMO_SHA256:
            print("WARNING: Historical source values differ from demo v0.1. "
                  "Keep this download as a new snapshot; results may differ.")
        return 0
    except (ValueError, OSError, URLError, HTTPError, TimeoutError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
