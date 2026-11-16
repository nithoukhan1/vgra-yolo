"""STEP-03 tests: finding and continuing runs across Kaggle versions."""

import json

from vgra import runs, utils


def make_run(folder, epochs, finished=None, resume=True):
    folder.mkdir(parents=True)
    (folder / "results.csv").write_text("epoch,loss\n" + "".join(f"{i},0.1\n" for i in range(1, epochs + 1)))
    if resume:
        (folder / "weights").mkdir()
        (folder / "weights" / "resume.pt").write_bytes(b"ckpt")
    if finished is not None:
        (folder / "metrics.json").write_text(json.dumps({"finished": finished, "valid": {"map50": 0.6}}))
    return folder


def test_results_rows_and_finished(tmp_path):
    run = make_run(tmp_path / "B1_s0", 7, finished=False)
    assert runs.results_rows(run) == 7 and not runs.is_finished(run)
    assert runs.results_rows(tmp_path / "missing") == 0 and not runs.is_finished(tmp_path / "missing")


def test_find_previous_run_takes_the_longest(tmp_path):
    root = tmp_path / "input"
    make_run(root / "version1" / "runs" / "B1_s0", 30)
    longest = make_run(root / "version2" / "runs" / "B1_s0", 61)
    make_run(root / "version2" / "runs" / "B0_s0", 90)  # other run: ignored
    make_run(root / "data" / "images" / "B1_s0", 99)  # inside an image folder: never searched
    assert runs.find_previous_run("B1_s0", root) == longest
    assert runs.find_previous_run("B1_s1", root) is None
    assert runs.find_previous_run("B1_s0", tmp_path / "nowhere") is None


def test_find_previous_run_prefers_finished(tmp_path):
    root = tmp_path / "input"
    make_run(root / "a" / "runs" / "B1_s0", 40, finished=False)
    done = make_run(root / "b" / "runs" / "B1_s0", 40, finished=True, resume=False)
    assert runs.find_previous_run("B1_s0", root) == done


def test_restore_run_copies(tmp_path):
    src = make_run(tmp_path / "input" / "runs" / "B1_s0", 3)
    dst = runs.restore_run(src, tmp_path / "working" / "runs" / "B1_s0")
    assert (dst / "weights" / "resume.pt").read_bytes() == b"ckpt" and runs.results_rows(dst) == 3
    assert runs.restore_run(dst, dst) == dst  # copying onto itself does nothing


def test_time_budget():
    budget = runs.TimeBudget(hours=11.0, reserve_minutes=20, start=0.0)
    epoch = 20 * 60  # a 20-minute epoch
    assert not budget.should_stop(epoch, now=9.0 * 3600)
    # 10.3 h used + 25 min (1.25 epochs) + 20 min reserve > 11 h: stop now
    assert budget.should_stop(epoch, now=10.3 * 3600)
    assert budget.elapsed(now=3600.0) == 3600.0


def test_run_info_keeps_sessions(tmp_path):
    cfg = {"exp_id": "B1"}
    path = utils.write_run_info(tmp_path, cfg, seed=0)
    utils.write_run_info(tmp_path, cfg, seed=0)
    info = json.loads(path.read_text())
    assert info["run_name"] == "B1_s0" and len(info["sessions"]) == 2
    assert info["sessions"][0]["started_utc"] == info["started_utc"]
