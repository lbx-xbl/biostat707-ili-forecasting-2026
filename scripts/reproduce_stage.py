#!/usr/bin/env python3
"""Rerun one analysis stage in this repository checkout.

The original run_rolling.py, run_diagnostics.py, run_probabilistic.py and
run_vintage.py first check that earlier development files are unchanged. Some
of those files (companion ZIP archives one directory up, setup logs, progress
notes) were never part of this repository, so in a fresh clone those scripts
stop before any analysis. The original scripts are frozen and hash-checked by
scripts/verify_submission.py, so they are not edited.

This wrapper runs the same stage unchanged, except for that preservation check:
every baseline file that exists in this repository is still checked by
SHA-256, except README.md and .gitignore, which were rewritten for this
repository and are not analysis inputs. Files absent from the repository and
those two files are reported as skipped.

Example (from the repository root):
    python scripts/reproduce_stage.py rolling --output-dir outputs/rolling_reproduction
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "rolling": ("run_rolling", "preservation", "docs/pilot_preservation_baseline.json"),
    "diagnostics": ("run_diagnostics", "preservation", "docs/diagnostics_preservation_baseline.json"),
    "probabilistic": ("run_probabilistic", "preservation", "docs/probabilistic_preservation_baseline.json"),
    "vintage": ("run_vintage", "preserve", "docs/vintage_preservation_baseline.json"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Rewritten for this repository on purpose; not analysis inputs.
REWRITTEN = {"README.md", ".gitignore"}


def in_repository(path):
    resolved = (ROOT / path).resolve()
    return ROOT in resolved.parents and resolved.is_file()


def make_check(baseline_path):
    def check():
        baseline = json.loads((ROOT / baseline_path).read_text())
        present = {p: h for p, h in baseline.items() if in_repository(p) and p not in REWRITTEN}
        changed = [p for p, h in present.items() if sha(ROOT / p) != h]
        if changed:
            raise ValueError(f"Frozen repository files changed: {changed}")
        return {
            "status": "PASS (repository files only)",
            "files_checked": len(present),
            "skipped_not_in_repository": sorted(set(baseline) - set(present)),
        }
    return check


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: python scripts/reproduce_stage.py {{{'|'.join(STAGES)}}} [--output-dir DIR] [--config FILE]")
    stage = sys.argv[1]
    module_name, check_name, baseline = STAGES[stage]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module(module_name)
    setattr(module, check_name, make_check(baseline))
    sys.argv = [f"{module_name}.py"] + sys.argv[2:]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
