from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import soarm
from soarm.spec import RobotSpec, SpecError


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_so101_loads() -> None:
    spec = soarm.so101()
    assert spec.name == "so101"
    assert spec.joints == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
    )
    assert spec.arm_joints == spec.joints[:5]
    assert spec.backend == "soarm.feetech:FeetechArm"
    assert spec.hardware.gripper_closed_ticks == 1986
    assert spec.hardware.gripper_open_ticks == 2670
    assert spec.mjcf_path.is_file()


def test_so101_from_another_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert soarm.so101().mjcf_path.is_file()


def test_every_robot_yaml_loads(any_spec: RobotSpec) -> None:
    assert any_spec.status in ("ready", "draft")


def test_home_inside_limits(ready_spec: RobotSpec) -> None:
    for joint, value in ready_spec.home.items():
        lo, hi = ready_spec.limits[joint]
        assert lo <= value <= hi, f"{joint}: home {value} outside {lo}..{hi}"


def test_wave_ranges_inside_limits(ready_spec: RobotSpec) -> None:
    for joint, (wlo, whi) in ready_spec.wave_ranges.items():
        lo, hi = ready_spec.limits[joint]
        assert lo <= wlo < whi <= hi, f"{joint}: wave {wlo}..{whi} outside {lo}..{hi}"


def test_load_accepts_file_or_folder() -> None:
    folder = soarm.robot_dir("so101")
    assert soarm.load(folder) == soarm.load(folder / "robot.yaml")


def test_unknown_key_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    so101_data["colour"] = "orange"
    with pytest.raises(SpecError, match="unknown key.*colour"):
        soarm.load(write(tmp_path, so101_data))


def test_unknown_nested_key_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    so101_data["hardware"]["voltage"] = 12
    with pytest.raises(SpecError, match=r"hardware: unknown key.*voltage"):
        soarm.load(write(tmp_path, so101_data))


def test_missing_key_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    del so101_data["limits"]
    with pytest.raises(SpecError, match="missing key.*limits"):
        soarm.load(write(tmp_path, so101_data))


@pytest.mark.parametrize("section", ["limits", "home", "max_vel", "wave_ranges"])
def test_joint_outside_joints_is_an_error(tmp_path: Path, so101_data: dict, section: str) -> None:
    so101_data[section]["elbow_twist"] = so101_data[section]["elbow_flex"]
    with pytest.raises(SpecError, match=f"{section}: elbow_twist not in joints"):
        soarm.load(write(tmp_path, so101_data))


def test_joint_missing_from_limits_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    del so101_data["limits"]["wrist_roll"]
    with pytest.raises(SpecError, match="limits: missing joint.*wrist_roll"):
        soarm.load(write(tmp_path, so101_data))


def test_wrong_type_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    so101_data["hardware"]["baudrate"] = "fast"
    with pytest.raises(SpecError, match="hardware.baudrate: expected an integer"):
        soarm.load(write(tmp_path, so101_data))


def test_reversed_limit_is_an_error(tmp_path: Path, so101_data: dict) -> None:
    so101_data["limits"]["elbow_flex"] = [92.0, -92.0]
    with pytest.raises(SpecError, match="limits.elbow_flex"):
        soarm.load(write(tmp_path, so101_data))
