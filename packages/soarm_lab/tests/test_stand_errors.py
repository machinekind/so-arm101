"""Simulated stand errors: the hidden play of `Backlash`, its size at the hand, and a fixed camera whose
believed pose is shifted (every point moves by the same offset) or turned (a seeded field, the set distance
at the workspace's centre)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Observation

from soarm_lab.stand_errors import PLAY_JOINTS, Backlash, CameraPoseError, play_for, tilted, uniform

K = np.array([[920.0, 0.0, 639.5], [0.0, 920.0, 359.5], [0.0, 0.0, 1.0]])
SIZE = (1280, 720)


def look_at(eye, target) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, np.cross(z, x), z]), eye
    return T


T_TRUE = look_at([0.62, -0.30, 0.38], [0.2, 0.0, 0.0])
#: Points over the workspace at the heights of the table and of two and three stacked cubes [m].
POINTS = np.array([[x, y, z] for x in (0.16, 0.2, 0.25) for y in (-0.08, 0.0, 0.08)
                   for z in (0.0, 0.029, 0.058)])


# --------------------------------------------------------------------------- the hidden play
def test_backlash_hides_the_play_from_the_strategy():
    """The strategy sees its own targets reached; the arm lags by half the play after a turn."""

    class Echo:
        hz = 10.0

        def __init__(self) -> None:
            self.seen: list[dict[str, float]] = []
            self.targets = iter([{"elbow_flex": 10.0}, {"elbow_flex": 12.0}, {"elbow_flex": 11.0},
                                 {"elbow_flex": 9.0}])

        def start(self, run, obs) -> None:
            pass

        def act(self, obs) -> Action:
            self.seen.append(dict(obs.joints))
            return Action(target=next(self.targets))

    echo = Echo()
    play = Backlash(echo, {"elbow_flex": 2.0})
    obs = Observation(t=0.0, step=0, joints={"elbow_flex": 10.0}, command={"elbow_flex": 10.0})
    play.start(None, obs)
    sent = []
    for _ in range(4):
        action = play.act(obs)
        sent.append(action.target["elbow_flex"])
        # The arm reaches what it was sent.
        obs = replace(obs, joints={"elbow_flex": sent[-1]}, command={"elbow_flex": sent[-1]})
    # Up 2 deg moves the arm 1 deg less; back down 1 deg stays inside the play; down 2 more moves 1 deg.
    assert sent == [10.0, 11.0, 11.0, 10.0]
    assert [s["elbow_flex"] for s in echo.seen] == [10.0, 10.0, 12.0, 11.0]


def test_the_play_is_sized_at_the_hand():
    play = play_for(0.006)
    assert set(play) == set(PLAY_JOINTS) and all(0.5 < deg < 10.0 for deg in play.values())


# --------------------------------------------------------------------------- the fixed camera
def test_without_an_error_the_field_is_the_identity():
    camera = CameraPoseError(K, SIZE, T_TRUE, T_TRUE.copy())
    assert camera.field(POINTS) == pytest.approx(POINTS, abs=1e-12)


def test_a_believed_pose_shifted_sideways_moves_every_point_by_that_offset():
    camera = uniform(K, SIZE, T_TRUE, (0.003, -0.004))
    offset = camera.offset(POINTS)
    assert offset[:, :2] == pytest.approx(np.tile([0.003, -0.004], (len(POINTS), 1)), abs=1e-12)
    assert offset[:, 2] == pytest.approx(0.0)


def test_a_turned_believed_pose_puts_the_centre_the_set_distance_off_in_a_seeded_direction():
    a = tilted(K, SIZE, T_TRUE, 0.010, np.random.default_rng(4))
    b = tilted(K, SIZE, T_TRUE, 0.010, np.random.default_rng(5))
    centre = np.array([[0.2, 0.0, 0.0]])
    for camera in (a, b):
        assert np.linalg.norm(camera.offset(centre)[0, :2]) == pytest.approx(0.010, abs=1e-6)
        assert camera.T_believed[:3, 3] == pytest.approx(T_TRUE[:3, 3])
    da, db = a.offset(centre)[0, :2], b.offset(centre)[0, :2]
    assert math.degrees(math.acos(float(da @ db) / 1e-4)) > 5.0
    # A field, not an offset: it changes across the workspace and with the height.
    spread = np.ptp(np.linalg.norm(a.offset(POINTS)[:, :2], axis=1))
    assert 0.0005 < spread < 0.006


def test_the_true_camera_sees_points_where_it_projects_them():
    camera = uniform(K, SIZE, T_TRUE, (0.0, 0.0))
    uv = camera.pixels(np.array([[0.2, 0.0, 0.0]]))
    assert uv[0] == pytest.approx([639.5, 359.5], abs=1e-6)
