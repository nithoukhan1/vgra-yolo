"""Run folders across Kaggle versions (STEP-03).

A Kaggle "Save Version" run stops after 12 hours. A long training therefore runs as
several versions of the same notebook:

1. Version 1 trains until its time budget is nearly used, stops cleanly at the end of an
   epoch, and keeps `weights/resume.pt` (a full checkpoint with the optimizer).
2. Version 2 has the output of version 1 attached as input. `find_previous_run` finds the
   run folder there, `restore_run` copies it into the working folder, and training
   continues from `resume.pt`. The logs (results.csv) continue in the same folder.
3. When the run has finished, `metrics.json` says so and later versions do nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

SKIP_DIRS = {"images", "labels", ".git", "__pycache__", "weights"}  # never hold a run folder


def results_rows(run_dir: str | Path) -> int:
    """Number of finished epochs logged in results.csv (0 if there is none)."""
    f = Path(run_dir) / "results.csv"
    if not f.is_file():
        return 0
    return max(len(f.read_text(encoding="utf-8").strip().splitlines()) - 1, 0)


def read_metrics(run_dir: str | Path) -> dict:
    f = Path(run_dir) / "metrics.json"
    if not f.is_file():
        return {}
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def is_finished(run_dir: str | Path) -> bool:
    """True when the run has trained to its end (not stopped for time)."""
    return bool(read_metrics(run_dir).get("finished"))


def find_previous_run(run_name: str, search_root: str | Path) -> Path | None:
    """Find a folder called `run_name` that holds a training run, anywhere under `search_root`.

    A folder counts if it has results.csv, weights/resume.pt or metrics.json. If several
    are found (for example the outputs of two earlier versions), the one with the most
    logged epochs wins. Image and label folders are skipped, so large datasets are not walked.
    """
    root = Path(search_root)
    if not root.is_dir():
        return None
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if Path(dirpath).name != run_name:
            continue
        p = Path(dirpath)
        if (p / "results.csv").is_file() or (p / "weights" / "resume.pt").is_file() or (p / "metrics.json").is_file():
            found.append(p)
            dirnames[:] = []  # do not look inside a run folder
    if not found:
        return None
    return max(found, key=lambda p: (is_finished(p), results_rows(p), str(p)))


def restore_run(src: str | Path, dst: str | Path) -> Path:
    """Copy a previous run folder into the working folder (the input copy is read-only)."""
    src, dst = Path(src), Path(dst)
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


class TimeBudget:
    """Decides when to stop so that the run ends cleanly before the session limit.

    Stop after an epoch if one more epoch (with a 25% margin) plus the reserve for the
    final evaluation would pass the budget, counted from the start of the script.
    """

    def __init__(self, hours: float, reserve_minutes: float = 20.0, start: float | None = None, margin: float = 1.25):
        self.budget = float(hours) * 3600.0
        self.reserve = float(reserve_minutes) * 60.0
        self.start = time.time() if start is None else float(start)
        self.margin = float(margin)

    def elapsed(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.start

    def should_stop(self, last_epoch_seconds: float, now: float | None = None) -> bool:
        return self.elapsed(now) + self.margin * last_epoch_seconds + self.reserve > self.budget
