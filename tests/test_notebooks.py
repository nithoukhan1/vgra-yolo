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


def test_frozen_weights_manifest_matches_the_test_session_runs():
    import re

    frozen = json.loads((NB.parents[1] / "docs" / "final_weights.json").read_text(encoding="utf-8"))
    runs = {k: v for k, v in frozen.items() if not k.startswith("_")}
    settings = cells()[0]
    planned = re.search(r"test session: (\[.*?\])", settings).group(1)
    assert set(json.loads(planned)) == set(runs)
    for name, v in runs.items():
        assert re.fullmatch(r"[0-9a-f]{64}", v["sha256"]), name
        assert v["epochs"] == 100 and v["detector"] == name.rsplit("_s", 1)[0]


def test_hash_check_blocks_only_the_test_session():
    check = cells()[2]
    assert "final_weights.json" in check and "sha256(" in check
    assert check.count("(problems if OPEN_TEST else warnings)") == 2  # unfinished run, wrong weights
