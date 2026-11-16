"""Small helpers used by every script: config loading, seeding, run info."""

from __future__ import annotations

import copy
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in (nested dicts are merged)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, base_path: str | Path | None = None) -> dict:
    """Load an experiment config on top of base.yaml.

    `path` is e.g. configs/V1.yaml. The base file defaults to base.yaml in the
    same folder. `exp_id` is always the file name without extension, so the
    file name and the run name cannot disagree.
    """
    path = Path(path)
    base_path = Path(base_path) if base_path else path.with_name("base.yaml")

    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path.resolve() != base_path.resolve():
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})

    cfg["exp_id"] = path.stem
    return cfg


def run_name(cfg: dict, seed: int) -> str:
    """Run name rule: <expID>_s<seed>, e.g. V1_s0."""
    return f"{cfg['exp_id']}_s{int(seed)}"


def git_commit(repo_root: str | Path = REPO_ROOT) -> str:
    """Short commit hash of the repo, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty(repo_root: str | Path = REPO_ROOT) -> bool | None:
    """True if there are uncommitted changes; None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    """Installed version of a package, or None if it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def library_versions() -> dict:
    """Versions that matter for reproducing a run."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vgra": package_version("vgra"),
        "ultralytics": package_version("ultralytics"),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
    }


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (the last two only if installed)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_run_info(run_dir: str | Path, cfg: dict, seed: int) -> Path:
    """Write run_info.json into the run folder. A run without it does not count.

    A run that continues in a later session keeps the first session's record and gets one
    more entry under "sessions". A warning is printed if the code commit has changed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "run_info.json"
    session = {
        "commit": git_commit(),
        "uncommitted_changes": git_is_dirty(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": library_versions(),
    }
    if out.is_file():
        with open(out, encoding="utf-8") as f:
            info = json.load(f)
        info.setdefault("sessions", []).append(session)
        if session["commit"] != info.get("commit"):
            print(f"WARNING: this session runs commit {session['commit']}, "
                  f"but the run started with commit {info.get('commit')}")
    else:
        info = {
            "run_name": run_name(cfg, seed),
            "exp_id": cfg["exp_id"],
            "seed": int(seed),
            "commit": session["commit"],
            "uncommitted_changes": session["uncommitted_changes"],
            "started_utc": session["started_utc"],
            "versions": session["versions"],
            "config": cfg,
            "sessions": [session],
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out
