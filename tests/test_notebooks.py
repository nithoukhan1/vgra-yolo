"""The evaluation notebook stops on any failure and never runs a step without the run check."""

import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_eval.ipynb"


def cells():
    return ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def test_settings_define_everything_the_steps_use():
    s = cells()[0]
    tree = ast.parse(s)
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {"REPO", "REF", "OPEN_TEST", "RUNS", "RUN_ARGS", "READY"} <= names
    assert "OPEN_TEST = False" in s  # the notebook in the repo is always the safe dry run


def test_steps_stop_on_failure():
    code = cells()
    check, steps = code[2], code[3:]
    assert "SystemExit" not in check and "raise RuntimeError" in check and "READY = True" in check
    for s in [check] + steps:
        ast.parse(s)  # plain Python: no shell magics whose failures would be ignored
        assert "!python" not in s
    for s in steps[:-1]:
        assert 'assert READY' in s and "sh(" in s


def test_unfinished_runs_block_only_the_test_session():
    check = cells()[2]
    assert "(problems if OPEN_TEST else warnings)" in check
