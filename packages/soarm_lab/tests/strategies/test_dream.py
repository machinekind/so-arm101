"""The Dream strategy on a fake Dream client: no network and no Dream library.

`FakeClient` stands in for `dream.DreamClient`: it keeps the policy config and every tick it is handed, says a
chunk is due once the buffer is empty, and answers the chunk it was given. A fake encoder stands in for the
JPEG encoder, except in the test of the encoder itself. The tests play the arm themselves: the measured
joints stay where they are unless a test moves them.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, Observation, RunInfo

import soarm
from soarm_lab.robot import load_robot
from soarm_lab.strategies.dream import (
    ARM_JOINTS,
    FRAME_SIZE,
    IK_ARM,
    LEFT_REST_RAD,
    DreamSettings,
    DreamStrategy,
    encode_frame,
)

SPEC = load_robot()
HOME = dict(SPEC.home)
SCENE_IMAGE = np.zeros((480, 640, 3), np.uint8)
WRIST_IMAGE = np.zeros((360, 640, 3), np.uint8)


class FakeClient:
    """`dream.DreamClient` as the strategy uses it."""

    mode = "async"

    def __init__(self, chunk: np.ndarray | None = None, error: Exception | None = None):
        self.chunk = chunk if chunk is not None else np.zeros((48, 6))
        self.error = error
        self.config: dict[str, Any] | None = None
        self.ticks: list[tuple[dict[str, str], dict[str, Any]]] = []
        self.commanded: list[Any] = []
        self.resets = 0
        self.closed = False

    def policy(self, config: dict[str, Any]) -> None:
        self.config = config

    def set_observation(self, frames, observation, *, timestamp=None) -> None:
        self.ticks.append((dict(frames), dict(observation)))

    def refill_due(self, executed: int, remaining: int) -> bool:
        return remaining <= 0

    def get_action(self, *, commanded=None) -> dict[str, Any]:
        self.commanded.append(commanded)
        if self.error is not None:
            raise self.error
        return {"q_r_chunk": self.chunk.tolist(), "q_l_chunk": np.zeros_like(self.chunk).tolist()}

    def reset_episode(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class DreamServerError(RuntimeError):
    """The Dream client's refusal, with the attributes the strategy reads."""

    def __init__(self, message: str, retryable: bool, retry_after_s: float | None = None, reason: str = ""):
        super().__init__(message)
        self.retryable, self.retry_after_s, self.reason = retryable, retry_after_s, reason


def fake_encode(rgb: np.ndarray) -> str:
    return f"jpeg {rgb.shape[1]}x{rgb.shape[0]}"


def run_info() -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits=dict(SPEC.limits),
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)


def obs(joints: dict[str, float], step: int = 0, frames: dict[str, Frame] | None = None) -> Observation:
    frames = {"top": Frame(SCENE_IMAGE, 0.0)} if frames is None else frames
    return Observation(t=step / 20, step=step, joints=joints, command={}, frames=frames)


def started(chunk: np.ndarray | None = None, error: Exception | None = None,
            **settings) -> tuple[DreamStrategy, FakeClient]:
    client = FakeClient(chunk, error)
    s = DreamStrategy(DreamSettings(**settings), client_factory=lambda: client, encode=fake_encode)
    s.start(run_info(), obs(HOME))
    return s, client


def home_rad() -> np.ndarray:
    return np.radians([HOME[j] for j in ARM_JOINTS])


def first_request(s: DreamStrategy, joints: dict[str, float] = HOME) -> Action:
    """The first tick: it hands the client its tick and asks for a chunk; the next tick collects it."""
    action = s.act(obs(joints, 1))
    s.wait_request(5.0)
    return action


def arm(action: Action) -> list[float]:
    return [action.target[j] for j in ARM_JOINTS]


def test_the_policy_runs_the_so101_through_dreams_ik() -> None:
    _, client = started()
    config = client.config
    assert config is not None
    assert config["checkpoint"] == "dreamlabs/dream-1-beta"
    assert config["ik"] == {"arm": IK_ARM, "head": "eef"}
    assert (config["action_hz"], config["execution_horizon"]) == (20.0, 30)
    body = config["embodiment"]
    assert set(body["video"]) == {"scene", "wrist_l", "wrist_r"}
    # Dream's spec of the arm is six wide (the dummy sixth joint), and names the drive the client makes.
    assert {part: end - start for part, (start, end) in body["state"].items()
            if not part.split("_", 1)[1].startswith("eef")} == {
        "left_arm": 6, "right_arm": 6, "right_grip": 1, "left_grip": 1, "right_gripper": 1, "left_gripper": 1}


def test_a_tick_sends_this_arm_on_the_right_and_a_dummy_left_arm() -> None:
    s, client = started()
    joints = {**HOME, "shoulder_pan": 10.0, "gripper": 40.0}
    s.act(obs(joints, 1))
    frames, observation = client.ticks[-1]
    assert frames == {f"video.{role}": "jpeg 640x480" for role in ("scene", "wrist_l", "wrist_r")}
    # Never the tool poses or the gripper drives: the client computes those.
    assert set(observation) == {"prompt", "state.right_arm", "state.right_grip", "state.left_arm",
                                "state.left_grip"}
    assert observation["prompt"] == "pick up the cube"
    right = observation["state.right_arm"]
    assert right.shape == (5,) and right.dtype == np.float32
    np.testing.assert_allclose(right, np.radians([joints[j] for j in ARM_JOINTS]), rtol=1e-6)
    assert observation["state.right_grip"].shape == (1,)
    assert observation["state.right_grip"][0] == pytest.approx(0.4)
    np.testing.assert_allclose(observation["state.left_arm"], LEFT_REST_RAD, rtol=1e-6)
    assert observation["state.left_grip"].tolist() == [1.0]


def test_a_wrist_camera_fills_the_right_wrist_and_black_the_left() -> None:
    s, client = started(scene_camera="top", wrist_camera="wrist", fill_frame="black")
    s.act(obs(HOME, 1, {"top": Frame(SCENE_IMAGE, 0.0), "wrist": Frame(WRIST_IMAGE, 0.0)}))
    frames, _ = client.ticks[-1]
    assert frames == {"video.scene": "jpeg 640x480", "video.wrist_l": f"jpeg {FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
                      "video.wrist_r": "jpeg 640x360"}


def test_a_stale_frame_holds_the_arm() -> None:
    s, client = started()
    action = s.act(obs(HOME, 1, {"top": Frame(SCENE_IMAGE, 2.5)}))
    assert not action.done and "2.5 s old" in action.note
    assert arm(action) == pytest.approx([HOME[j] for j in ARM_JOINTS])
    assert client.ticks == [] and client.commanded == []


def jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's start-of-frame marker."""
    i = 2
    while i + 9 < len(data):
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    raise AssertionError("no start-of-frame marker")


def test_a_frame_goes_out_as_a_320x176_base64_jpeg() -> None:
    try:
        encoded = encode_frame(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    except RuntimeError as exc:
        pytest.skip(str(exc))                                   # neither OpenCV nor Pillow
    data = base64.b64decode(encoded, validate=True)
    assert data[:2] == b"\xff\xd8"
    assert jpeg_size(data) == FRAME_SIZE == (320, 176)


def test_a_chunk_plays_row_by_row_in_degrees_and_gripper_units() -> None:
    rows = 48
    chunk = np.zeros((rows, 6))
    chunk[:, :5] = home_rad() + np.linspace(0.0, 0.05, rows)[:, None]
    chunk[:, 5] = np.linspace(0.30, 0.40, rows)
    s, client = started(chunk)
    assert arm(first_request(s)) == pytest.approx([HOME[j] for j in ARM_JOINTS])   # no chunk yet: holds
    np.testing.assert_allclose(client.commanded[0]["q_r_chunk"], home_rad())       # anchored where it is
    for k in range(3):
        action = s.act(obs(HOME, 2 + k))
        assert arm(action) == pytest.approx(np.degrees(chunk[k, :5]).tolist())
        assert action.target["gripper"] == pytest.approx(100.0 * chunk[k, 5])
        assert set(action.target) == set(SPEC.joints) and not action.done
    assert (s.status.chunks, s.status.rows_left, s.status.skipped) == (1, rows - 3, 0)
    assert len(client.commanded) == 1                                              # not due yet


def test_the_speed_cap_ramps_toward_a_far_row() -> None:
    chunk = np.tile(np.r_[home_rad() + np.radians([30.0, 0, 0, 0, 0]), 0.35], (48, 1))
    s, _ = started(chunk, max_step_deg=2.0)
    first_request(s)
    pan = [s.act(obs(HOME, 2 + k)).target["shoulder_pan"] for k in range(4)]
    assert np.diff([HOME["shoulder_pan"], *pan]) == pytest.approx([2.0] * 4)
    assert s.status.capped


def test_a_new_prompt_starts_a_fresh_episode() -> None:
    s, client = started()
    first_request(s)
    s.act(obs(HOME, 2))
    assert s.status.rows_left == 47
    s.set_prompt("put the cube on the left")
    s.act(obs(HOME, 3))
    assert client.resets == 1 and s.status.rows_left == 0
    assert client.ticks[-1][1]["prompt"] == "put the cube on the left"
    assert len(client.commanded) == 2                                   # the new episode asks at once


def test_a_busy_model_is_asked_again_and_a_refusal_ends_the_run() -> None:
    busy = DreamServerError("starting", retryable=True, retry_after_s=5.0, reason="pool_warming")
    s, client = started(error=busy)
    first_request(s)
    action = s.act(obs(HOME, 2))
    assert not action.done and "pool_warming" in action.note
    assert len(client.commanded) == 1                                              # waits the 5 s it was told

    s, client = started(error=DreamServerError("key refused", retryable=False))
    first_request(s)
    action = s.act(obs(HOME, 2))
    assert action.done and "key refused" in action.note


def test_the_strategy_follows_the_strategy_protocol() -> None:
    """As tests/test_entry_points.py checks the offline strategies; `stop` closes the client."""
    s, client = started()
    assert isinstance(s.hz, float) and s.hz == 20.0
    action = s.act(obs(HOME, 1))
    assert isinstance(action, Action)
    assert set(action.target) == set(SPEC.joints) and not action.done
    s.stop("zatrzymana")
    assert client.closed and not s.started
