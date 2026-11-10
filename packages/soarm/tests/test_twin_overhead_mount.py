"""The overhead camera mount in the twin: beside the SO-101, the arm cannot drive into it, and the camera
rides on top of the built pole."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

import soarm

pytestmark = pytest.mark.twin
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as sc  # noqa: E402
from machinekind_twin.collision import CollisionChecker  # noqa: E402
from machinekind_twin.robots import get_spec  # noqa: E402
from machinekind_twin.workspace import CameraRecord, FixtureRecord, Workspace  # noqa: E402

FOLDER = soarm.ROBOT_DIR / "fixtures" / "overhead_cam_32"
TWO, THREE = ["bottom", "top"], ["bottom", "middle", "top"]


@pytest.fixture(scope="module")
def fixture() -> dict:
    return yaml.safe_load((FOLDER / "fixture.yaml").read_text(encoding="utf-8"))


def offsets(fixture: dict, parts: list[str]) -> list[float]:
    """Height of each part's frame in the stack."""
    z, out = 0.0, []
    for p in parts:
        out.append(z)
        z += fixture["parts"][p].get("top", [0.0, 0.0, 0.0])[2]
    return out


def station(placement: str = "left", parts: list[str] | None = None):
    ws = Workspace(robot="so101")
    ws.fixtures.append(FixtureRecord("slup", "so101/overhead_cam_32", placement=placement,
                                     parts=parts or TWO))
    ws.add_camera(CameraRecord("kamera3", source="1", width=1920, height=1080, fixture="slup"))
    return ws


def turned(home: dict, pan: float) -> dict:
    """The arm stretched out flat, turned left (pan < 0) or right."""
    return {**home, "shoulder_pan": pan, "shoulder_lift": 0.0, "elbow_flex": 0.0}


@pytest.mark.parametrize("clearance", [0.0, 0.02])
@pytest.mark.parametrize(("placement", "into", "away"), [("left", -90.0, 90.0), ("right", 90.0, -90.0)])
def test_the_arm_cannot_drive_into_the_pole(clearance, placement, into, away) -> None:
    home = get_spec("so101").home
    checker = CollisionChecker(sc.build(station(placement).scene_config()), clearance=clearance)
    assert checker.config_clear(home) is None
    assert (checker.config_clear(turned(home, into)) or "").startswith("kolizja: lower_arm z slup/")
    assert checker.config_clear(turned(home, away)) is None
    assert checker.config_clear(turned(home, into / 2)) is None
    assert "elementu stanowiska" in (checker.closer(home, turned(home, into)) or "")


@pytest.mark.parametrize("parts", [TWO, THREE])
def test_the_camera_rides_on_top_of_the_built_pole(fixture: dict, parts: list[str]) -> None:
    ws = station(parts=parts)
    scene = sc.build(ws.scene_config())
    foot = ws.fixture_pose(ws.fixtures[0])
    assert foot[:3, 3] == pytest.approx(fixture["placements"]["left"]["xyz"])
    top = offsets(fixture, parts)[-1]
    head = scene.data.xpos[scene.model.body("slup/top").id] - scene.T_base2world[:3, 3]
    assert head == pytest.approx(foot[:3, 3] + [0.0, 0.0, top], abs=1e-6)
    camera = ws.camera("kamera3").true_pose()
    site = np.array(fixture["parts"]["top"]["cameras"]["camera"])
    assert camera is not None and camera[:3, 3] == pytest.approx(foot[:3, 3] + site[:3, 3] + [0.0, 0.0, top])
    assert scene.camera_pose("kamera3") == pytest.approx(camera, abs=1e-6)
