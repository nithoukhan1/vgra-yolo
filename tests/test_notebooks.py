"""The evaluation notebook stops on any failure and never runs a step without the run check; the training notebook
starts the right commands (smoke or auto-resume, the protocol batch fallback), downloads every pretrained file first
and reports each run; both stop before anything runs when the cloned code is not at REF."""

import ast
import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_eval.ipynb"
NB_TRAIN = NB.parent / "kaggle_train.ipynb"


def cells(nb=NB):
    return ["".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


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


def test_frozen_weights_manifest_matches_the_test_sessions():
    import re

    from vgra import utils

    frozen = json.loads((NB.parents[1] / "docs" / "final_weights.json").read_text(encoding="utf-8"))
    runs = {k: v for k, v in frozen.items() if not k.startswith("_")}
    settings = cells()[0]
    part_a = json.loads(re.search(r"Part A \(done 2026-11-21, dataset vgra-final-detectors\): (\[.*?\])", settings).group(1))
    ext = json.loads(re.search(r"extension detectors \(protocol 11\.3, dataset vgra-rd-fr-s0\): (\[.*?\])", settings).group(1))
    assert len(part_a) == 8 and ext == ["RD_s0", "FR_s0"] and not set(part_a) & set(ext)
    assert set(part_a) | set(ext) == set(runs)
    for name, v in runs.items():
        det, seed = name.rsplit("_s", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", v["sha256"]), name
        assert v["detector"] == det and v["seed"] == int(seed)
        assert v["epochs"] == utils.load_config(NB.parents[1] / "configs" / f"{det}.yaml")["train"]["epochs"], name
        assert 1 <= v["best_epoch"] <= v["epochs"], name


@pytest.mark.parametrize("open_test", [False, True])
@pytest.mark.parametrize("runs,from_pred", [(["RD_s0", "FR_s0"], True), (["B0_s0", "B1_s0"], False)])
def test_i3_reads_the_i1_tables_for_rd_and_fr(runs, from_pred, open_test):
    """Protocol section 10 (2026-11-23): for RD and FR, I3 reads the I1 tables instead of running the detector again;
    the YOLO runs keep the route of the Part A session."""
    check = cells()[2]
    kinds = check[check.index("from vgra import utils as vgra_utils"):check.index('print("detector kinds')]
    ns = {"RUNS": runs}
    exec(kinds.replace("/kaggle/working/vgra-yolo", NB.parents[1].as_posix()), ns)
    assert ns["I3_FROM_PRED"] is from_pred
    called = []
    ns.update(READY=True, OPEN_TEST=open_test, RUN_ARGS=" ".join(runs), sh=called.append)
    exec(cells()[4], ns)
    assert len(called) == 1 and "scripts/04_premise_test.py" in called[0]
    assert ("--from-pred /kaggle/working/runs/predictions" in called[0]) is from_pred
    assert ("--open-test" in called[0]) is open_test and ("--split test" in called[0]) is open_test


def test_hash_check_blocks_only_the_test_session():
    check = cells()[2]
    assert "final_weights.json" in check and "sha256(" in check
    assert check.count("(problems if OPEN_TEST else warnings)") == 2  # unfinished run, wrong weights


# ---------- both notebooks: the code must be at REF -------------------------------------------

@pytest.mark.parametrize("nb", [NB, NB_TRAIN], ids=["eval", "train"])
def test_setup_stops_when_the_code_is_not_at_ref(tmp_path, monkeypatch, nb):
    """2026-11-23: notebook D was started before the tag was on GitHub; `git checkout` failed, the notebook went on
    with the old main and crashed later. The setup cell now stops right after the checkout."""
    import subprocess

    setup = cells(nb)[1]
    assert setup.index("git checkout") < setup.index("rev-parse") < setup.index("pip install")
    guard = "\n".join(line for line in setup.splitlines() if not line.lstrip().startswith(("!", "%", "#")))
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *a], cwd=repo, check=True,
                       capture_output=True)

    git("init", "-q")
    (repo / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-q", "-m", "one")
    git("tag", "detectors-v4")
    (repo / "a.txt").write_text("2")
    git("commit", "-q", "-am", "two")
    monkeypatch.chdir(repo)
    with pytest.raises(RuntimeError, match="not at detectors-v4"):  # checkout failed: still on a later commit
        exec(guard, {"REF": "detectors-v4"})
    with pytest.raises(RuntimeError, match="not at detectors-v9"):  # the tag does not exist
        exec(guard, {"REF": "detectors-v9"})
    git("checkout", "-q", "detectors-v4")
    exec(guard, {"REF": "detectors-v4"})  # at the tag: goes on


# ---------- training notebook -----------------------------------------------------------------

def train_settings(**change):
    ns = {}
    exec(cells(NB_TRAIN)[0], ns)
    ns.update(change)
    return ns


def test_training_settings_are_notebook_d():
    from vgra import utils

    ns = train_settings()
    assert ns["REF"] == "detectors-v5" and ns["SMOKE"] is True and ns["BATCH"] == {}
    assert [(c, s) for c, s, _ in ns["RUNS"]] == [("configs/RD.yaml", 0), ("configs/FR.yaml", 0)]
    assert sorted(g for _, _, g in ns["RUNS"]) == [0, 1]  # one run per GPU
    summary = cells(NB_TRAIN)[4]
    fallback = ast.literal_eval(summary.split("FALLBACK = ")[1].split("  #")[0])
    for exp, b in fallback.items():  # protocol 11.3: half the configured batch
        assert b * 2 == utils.load_config(NB.parents[1] / "configs" / f"{exp}.yaml")["train"]["batch"]


def test_weights_cell_downloads_every_pretrained_file():
    from vgra import utils

    weights = cells(NB_TRAIN)[2]
    assert "yolo26n.pt" in weights  # Ultralytics AMP check
    for f in sorted((NB.parents[1] / "configs").glob("*.yaml")):
        if f.name == "base.yaml":
            continue
        m = utils.load_config(f)["model"]
        assert (m["weights"] if m.get("kind", "yolo") != "frcnn" else "COCO_V1.get_state_dict") in weights, f.name


@pytest.mark.parametrize("smoke,batch", [(True, {}), (False, {}), (False, {"FR": 2}), (True, {"RD": 4})])
def test_training_commands(tmp_path, monkeypatch, smoke, batch):
    import subprocess
    import time

    calls = []

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            calls.append(cmd)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    ns = train_settings(SMOKE=smoke, BATCH=batch)
    exec(cells(NB_TRAIN)[3].replace("/kaggle/working", tmp_path.as_posix()), ns)
    assert len(calls) == 2
    for cmd, (cfg, seed, gpu) in zip(calls, ns["RUNS"]):
        exp = Path(cfg).stem
        assert cmd[1:8] == ["scripts/03_train_detector.py", "--config", cfg, "--seed", str(seed), "--device", str(gpu)]
        assert ("--smoke" in cmd) == smoke and ("--auto-resume" in cmd) == (not smoke)
        assert (cmd[-2:] == ["--batch", str(batch[exp])]) if exp in batch else "--batch" not in cmd
    logs = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert all(("_smoke_" in n) == smoke for n in logs) and len(logs) == 2
    assert ns["EXIT"] == {"RD_s0": 0, "FR_s0": 0}  # the summary cell reads the exit codes


def _summary(tmp_path, capsys, **change):
    ns = train_settings(**change)
    ns["os"] = __import__("os")
    exec(cells(NB_TRAIN)[4].replace("/kaggle/working", tmp_path.as_posix()), ns)
    return capsys.readouterr().out


def _run(tmp_path, folder, **metrics):
    (tmp_path / "runs" / folder).mkdir(parents=True)
    (tmp_path / "runs" / folder / "metrics.json").write_text(json.dumps(metrics))


def test_training_summary_smoke(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "RD_s0_smoke_20261124-0900.log").write_text(
        "RUN STATUS: FINISHED after 1 epochs\n\n=== timing (smoke run) ===\nestimated full epoch on 14000 images: "
        "21.0 min training + 2.0 min validation\npeak GPU memory: 9.1 GB\n")
    _run(tmp_path, "RD_s0_smoke640", batch=8, epochs_total=1, finished=True, valid={"map50": 0.1, "map50_95": 0.05})
    (tmp_path / "logs" / "FR_s0_smoke_20261124-0901.log").write_text(
        "Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB\n")
    out = _summary(tmp_path, capsys)
    assert "RD_s0: SMOKE RUN DONE | batch 8" in out and "peak GPU memory: 9.1 GB" in out
    assert "FR_s0: OUT OF MEMORY with the configured batch" in out and '"FR": 2 to BATCH' in out
    out = _summary(tmp_path, capsys, BATCH={"FR": 2})
    assert "FR_s0: OUT OF MEMORY at batch 2. No further fallback" in out


def test_training_summary_full_runs(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    for n in ("RD_s0", "FR_s0"):
        (tmp_path / "logs" / f"{n}_20261124-0900.log").write_text("line\n" * 30 + "last line\n")
    _run(tmp_path, "RD_s0", batch=8, epochs_total=72, finished=True, valid={"map50": 0.6, "map50_95": 0.4})
    out = _summary(tmp_path, capsys, SMOKE=False)
    assert "RD_s0: FINISHED | batch 8 | epochs 72" in out
    assert "FR_s0: no metrics.json" in out and "last line" in out
    _run(tmp_path, "FR_s0", batch=4, epochs_total=9, finished=False, valid={"map50": 0.5, "map50_95": 0.3})
    out = _summary(tmp_path, capsys, SMOKE=False)
    assert "FR_s0: STOPPED FOR TIME (attach this output, run the next version) | batch 4 | epochs 9" in out


def test_training_summary_reports_a_crash(tmp_path, capsys):
    """2026-11-24, notebook D version 2: FR_s0 crashed at resume, but the summary read the metrics.json restored from
    version 1 and printed STOPPED FOR TIME. The exit code now decides first."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "RD_s0_20261124-0440.log").write_text("RUN STATUS: STOPPED FOR TIME after 64 epochs\n")
    _run(tmp_path, "RD_s0", batch=8, epochs_total=64, finished=False, valid={"map50": 0.6372, "map50_95": 0.408})
    (tmp_path / "logs" / "FR_s0_20261124-0440.log").write_text(
        "auto-resume: found ... (15 epochs logged)\nTraceback (most recent call last):\nValueError: loaded state dict "
        "contains a parameter group that doesn't match the size of optimizer's group\n")
    _run(tmp_path, "FR_s0", batch=4, epochs_total=15, finished=False, valid={"map50": 0.5868, "map50_95": 0.3449})
    out = _summary(tmp_path, capsys, SMOKE=False, EXIT={"RD_s0": 0, "FR_s0": 1})
    assert "RD_s0: STOPPED FOR TIME" in out and "batch 8 | epochs 64" in out
    assert "FR_s0: CRASHED (exit code 1)" in out and "ValueError: loaded state dict" in out
    assert "FR_s0: STOPPED FOR TIME" not in out
    (tmp_path / "logs" / "FR_s0_20261124-0440.log").write_text("torch.OutOfMemoryError: CUDA out of memory.\n")
    out = _summary(tmp_path, capsys, SMOKE=False, EXIT={"RD_s0": 0, "FR_s0": 1})
    assert "FR_s0: OUT OF MEMORY" in out and "FR_s0: STOPPED FOR TIME" not in out and "CRASHED" not in out
