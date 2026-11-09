"""Simulated stand errors: hidden play in the arm's joints and a fixed camera whose believed pose is wrong.

`Backlash` wraps a strategy: the strategy sees its own targets reached, the simulated arm gets the link
side, which follows a target only once it has taken up each joint's play. `play_for` sizes the play per
joint from a distance at the hand.

`CameraPoseError` is a fixed camera calibrated to a wrong pose. A point the camera sees lies on the ray
from its TRUE pose; the perception meets the ray through the same pixel from the BELIEVED pose with the
point's height. `field(P)` gives where the perception puts points `P`, so a board corner and a block go
through one error. `uniform` shifts the believed pose sideways, which shifts every point by the same
offset; `tilted` turns it about its optical centre.

numpy and the twin's API only; no twin.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation

#: The arm joints that place the hand, and so carry play.
PLAY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


class Backlash:
    """Hidden play of each joint between the strategy's targets and the simulated arm [deg].

    The strategy sees the motor side: its targets reached, and measured joints that include the play's
    offset. The arm gets the link side, which follows a target only once the target has taken up the play.
    """

    def __init__(self, strategy: Any, play_deg: Mapping[str, float]):
        self.strategy = strategy
        self.play = dict(play_deg)
        self.hz = strategy.hz
        self.time_limit_s = getattr(strategy, "time_limit_s", None)
        self.motor: dict[str, float] = {}
        self.link: dict[str, float] = {}

    def _seen(self, obs: Observation) -> Observation:
        def shifted(values: Mapping[str, float]) -> dict[str, float]:
            return {k: float(v) + self.motor[k] - self.link[k] if k in self.link else float(v)
                    for k, v in values.items()}

        return replace(obs, joints=shifted(obs.joints), command=shifted(obs.command))

    def start(self, run: Any, obs: Observation) -> None:
        self.motor = {k: float(obs.joints[k]) for k in self.play if k in obs.joints}
        self.link = dict(self.motor)
        self.strategy.start(run, obs)

    def act(self, obs: Observation) -> Action:
        action = self.strategy.act(self._seen(obs))
        target = dict(action.target)
        for k, play in self.play.items():
            if k not in target or k not in self.link:
                continue
            x, y = float(target[k]), self.link[k]
            if x > y + play / 2:
                y = x - play / 2
            elif x < y - play / 2:
                y = x + play / 2
            self.motor[k], self.link[k] = x, y
            target[k] = y
        return replace(action, target=target)

    def stop(self, reason: str) -> None:
        self.strategy.stop(reason)


@functools.lru_cache(maxsize=1)
def _kinematics() -> Any:
    from .kinematics import RobotKinematics
    from .robot import load_robot

    return RobotKinematics(load_robot())


def play_for(metres: float, xy: tuple[float, float] = (0.2, 0.0), kin: Any = None) -> dict[str, float]:
    """Play per joint [deg] that moves the hand `metres` across its whole width, at a grasp over `xy`."""
    from .strategies.grasp import top_down

    kin = kin if kin is not None else _kinematics()
    home = dict(kin.spec.home, gripper=0.0)
    sol = kin.ik(np.array([*xy, 0.02]), top_down(0.0), seed=home)
    assert sol.ok
    joints = dict(sol.joints)
    out = {}
    for name in PLAY_JOINTS:
        step = 0.1
        ahead, back = dict(joints), dict(joints)
        ahead[name] += step
        back[name] -= step
        moved = np.linalg.norm(kin.tcp(ahead)[:3, 3] - kin.tcp(back)[:3, 3]) / (2 * step)
        out[name] = metres / float(moved)
    return out


# ------------------------------------------------------------------ a fixed camera's wrong pose
def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@dataclass(frozen=True)
class CameraPoseError:
    """A pinhole camera at `T_true` that the perception believes stands at `T_believed` (camera-to-base,
    OpenCV axes)."""

    K: np.ndarray
    size: tuple[int, int]
    T_true: np.ndarray
    T_believed: np.ndarray

    def pixels(self, P: np.ndarray) -> np.ndarray:
        """Where the camera really sees base-frame points (n, 3) [px]."""
        P = np.atleast_2d(np.asarray(P, float))
        R, t = self.T_true[:3, :3], self.T_true[:3, 3]
        c = (P - t) @ R
        uv = c[:, :2] / c[:, 2:3]
        return uv * np.diag(self.K)[:2] + self.K[:2, 2]

    def field(self, P: np.ndarray) -> np.ndarray:
        """Where the perception puts base-frame points (n, 3): the believed ray through each one's pixel,
        met with the plane at its height."""
        P = np.atleast_2d(np.asarray(P, float))
        uv = self.pixels(P)
        d_cam = np.column_stack([(uv - self.K[:2, 2]) / np.diag(self.K)[:2], np.ones(len(P))])
        R, o = self.T_believed[:3, :3], self.T_believed[:3, 3]
        d = d_cam @ R.T
        s = (P[:, 2] - o[2]) / d[:, 2]
        out = o + s[:, None] * d
        out[:, 2] = P[:, 2]
        return out

    def offset(self, P: np.ndarray) -> np.ndarray:
        """`field(P) - P` (n, 3) [m]."""
        P = np.atleast_2d(np.asarray(P, float))
        return self.field(P) - P


def uniform(K: np.ndarray, size: tuple[int, int], T_true: np.ndarray,
            offset_xy: np.ndarray | tuple[float, float]) -> CameraPoseError:
    """A believed pose shifted sideways by `offset_xy` [m]: every point moves by exactly that offset."""
    T = np.array(T_true, float, copy=True)
    T[:2, 3] += np.asarray(offset_xy, float)[:2]
    return CameraPoseError(np.asarray(K, float), size, np.asarray(T_true, float), T)


def tilted(K: np.ndarray, size: tuple[int, int], T_true: np.ndarray, metres: float, rng: np.random.Generator,
           at: tuple[float, float, float] = (0.2, 0.0, 0.0)) -> CameraPoseError:
    """A believed pose turned about the camera's optical centre, by an axis across the line of sight to
    `at` in a seeded direction, so far that the perception puts `at` `metres` off."""
    T_true = np.asarray(T_true, float)
    o = T_true[:3, 3]
    sight = np.asarray(at, float) - o
    sight /= np.linalg.norm(sight)
    a = np.cross(sight, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(sight, a)
    phi = float(rng.uniform(0.0, 2 * math.pi))
    axis = math.cos(phi) * a + math.sin(phi) * b

    def believed(angle: float) -> CameraPoseError:
        T = T_true.copy()
        T[:3, :3] = _rotation(axis, angle) @ T_true[:3, :3]
        return CameraPoseError(np.asarray(K, float), size, T_true, T)

    def error(angle: float) -> float:
        return float(np.linalg.norm(believed(angle).offset(np.array([at]))[0, :2]))

    lo, hi = 0.0, 1e-3
    while error(hi) < metres and hi < 0.5:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if error(mid) < metres else (lo, mid)
    return believed(0.5 * (lo + hi))
