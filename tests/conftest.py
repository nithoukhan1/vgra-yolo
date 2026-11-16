"""Lets tests import helpers that live in this folder (tiny_data.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
