"""STEP-00 pass check: is the environment ready?

Run:  python scripts/00_check_setup.py
Ends with 'SETUP CHECK: PASS' or 'SETUP CHECK: FAIL' (exit code 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def pinned_version(package: str) -> str | None:
    """Read '<package>==<version>' from requirements.txt."""
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith(package + "=="):
            return line.split("==", 1)[1].strip()
    return None


def main() -> int:
    ok = True

    def report(label: str, value, good: bool = True) -> None:
        nonlocal ok
        ok = ok and good
        print(f"[{'ok' if good else 'FAIL'}] {label}: {value}")

    # 1. our package
    try:
        import vgra
        from vgra import utils

        report("vgra import", f"version {vgra.__version__}")
    except Exception as e:  # noqa: BLE001
        report("vgra import", repr(e), good=False)
        print("SETUP CHECK: FAIL")
        return 1

    report("repo commit", utils.git_commit())

    # 2. pinned Ultralytics
    want = pinned_version("ultralytics")
    have = utils.package_version("ultralytics")
    report("ultralytics", f"installed {have}, pinned {want}", good=(have is not None and have == want))

    # 3. torch and GPU (GPU is informative only: EDA steps run on CPU)
    try:
        import torch

        report("torch", torch.__version__)
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[info] GPUs: {n} {names}")
    except ImportError:
        report("torch", "not installed", good=False)

    # 4. base config loads and has the expected sections
    try:
        cfg = utils.load_config(REPO_ROOT / "configs" / "base.yaml")
        missing = [k for k in ("data", "model", "loss", "train", "out_dir") if k not in cfg]
        report("configs/base.yaml", f"{len(cfg['data']['classes'])} classes", good=not missing)
    except Exception as e:  # noqa: BLE001
        report("configs/base.yaml", repr(e), good=False)

    print("SETUP CHECK: PASS" if ok else "SETUP CHECK: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
