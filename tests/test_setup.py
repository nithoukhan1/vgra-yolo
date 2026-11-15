"""STEP-00 tests: the package imports and the helpers behave."""

import json
from pathlib import Path

import yaml

import vgra
from vgra import utils

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import():
    assert isinstance(vgra.__version__, str)


def test_deep_update_merges_nested_and_keeps_base():
    base = {"model": {"scc": True, "rank": 16}, "out_dir": "a"}
    new = utils.deep_update(base, {"model": {"rank": 8}})
    assert new == {"model": {"scc": True, "rank": 8}, "out_dir": "a"}
    assert base["model"]["rank"] == 16  # base is not changed


def test_load_config_overrides_base(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"exp_id": "base", "model": {"scc": True, "vgra": "off"}})
    )
    (tmp_path / "V1.yaml").write_text(yaml.safe_dump({"model": {"vgra": "learned"}}))

    cfg = utils.load_config(tmp_path / "V1.yaml")
    assert cfg["exp_id"] == "V1"  # taken from the file name
    assert cfg["model"] == {"scc": True, "vgra": "learned"}


def test_run_name():
    assert utils.run_name({"exp_id": "V1"}, 0) == "V1_s0"


def test_write_run_info(tmp_path):
    cfg = {"exp_id": "B1", "model": {"scc": True}}
    path = utils.write_run_info(tmp_path / "B1_s2", cfg, seed=2)
    info = json.loads(path.read_text())
    assert info["run_name"] == "B1_s2"
    assert info["seed"] == 2
    for key in ("commit", "started_utc", "versions", "config"):
        assert key in info


def test_repo_base_config():
    cfg = utils.load_config(REPO_ROOT / "configs" / "base.yaml")
    assert cfg["exp_id"] == "base"
    assert len(cfg["data"]["classes"]) == 9
    assert set(cfg["data"]["gateable"]) <= set(cfg["data"]["classes"])
    assert cfg["model"]["vgra"] in ("off", "oracle", "learned")
