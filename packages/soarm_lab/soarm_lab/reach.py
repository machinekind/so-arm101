"""Where the arm can place a block, with the hand straight down or tilted.

`can_place` answers for one spot: can the arm put a block of a catalog kind with its bottom at a given
height at (x, y)? `reach_map` answers over a grid on the table, for a planner choosing where to build.

Both follow `pick_place`'s plan in outline, with its default settings: the release point with the
block's bottom just over the surface, the open hand rising until the jaw clears the placed block, and a
carry above the surface, each reachable with the hand down, straight or tilted about its closing axis by
up to `max_tilt_deg`. A tilted hand closes square to the arm and leans away from the base; with a yaw
given, the block must line up with it within `MAX_ROT_ERR_DEG` of block tilt. The held block is taken as
gripped from the table. `pick_place` may also leave with the jaw fully open, depending on where the arm
started, so it fits a few spots these call out of reach. It still plans the exact moves, and its pick
must reach the same tilt.

The arm turns about its base, so with the heading square to the arm the reach depends only on the
distance from the base's turning axis: `reach_map` checks one spot per distance and spreads it around
within the base's turning range.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .blocks import block
from .kinematics import RobotKinematics
from .robot import load_robot
from .strategies.grasp import MAX_ROT_ERR_DEG, MIN_GRASP_Z, TILT_STEP_DEG, hand_down, ik_restarts, wrap
from .strategies.pick_place import (
    FULL_OPEN,
    GRIP_SLIP_M,
    JAW_CLEAR_M,
    MIN_CARRY_CLEAR_M,
    TABLE_TOP_M,
    PickPlaceSettings,
    rot2,
)

#: Settings the plan follows.
_SETTINGS = PickPlaceSettings()


@dataclass(frozen=True)
class PlaceReach:
    """Whether a place fits, and with which hand tilt and closing heading."""

    ok: bool
    #: The least hand tilt from straight down that fits [deg]; NaN when none fits.
    tilt_deg: float = math.nan
    #: The closing heading at the release [deg, base frame]; NaN when none fits.
    heading_deg: float = math.nan


@dataclass(frozen=True)
class ReachMap:
    """The least hand tilt [deg] that places a block at each grid point; NaN where none does."""

    #: Grid coordinates [m, base frame]: `tilt_deg[i, j]` is at (`x_m[j]`, `y_m[i]`).
    x_m: np.ndarray
    y_m: np.ndarray
    tilt_deg: np.ndarray = field(repr=False)


class Reach:
    """Place checks on one arm's kinematics."""

    def __init__(self, kin: RobotKinematics | None = None, *, max_tilt_deg: float = _SETTINGS.max_tilt_deg):
        self.kin = kin if kin is not None else RobotKinematics(load_robot())
        spec = self.kin.spec
        self.max_tilt_deg = float(max_tilt_deg)
        self.seed = dict(spec.home)
        self.pan = spec.arm_joints[0]
        #: Where the base's turning axis crosses the table [m, base frame].
        self.pan_xy = self.kin.joint_anchor(self.pan, spec.home)[:2].copy()
        a = self.kin.to_q({name: spec.limits[name][0] for name in spec.joints})
        b = self.kin.to_q({name: spec.limits[name][1] for name in spec.joints})
        self.lo = np.maximum(np.minimum(a, b), self.kin.lo)
        self.hi = np.minimum(np.maximum(a, b), self.kin.hi)

    def tilts(self) -> list[float]:
        """Hand tilts to try [rad], as `grasp` tries them."""
        most = self.max_tilt_deg
        steps = [k * TILT_STEP_DEG for k in range(1, math.ceil(most / TILT_STEP_DEG - 1e-9))]
        return [0.0] + [math.radians(t) for t in [*steps, most] if most > 0]

    def place(self, kind: str, x: float, y: float, bottom_z: float, yaw_deg: float | None = None
              ) -> PlaceReach:
        """Whether a `kind` block fits with its bottom at `bottom_z` [m above the table] and its centre at
        (`x`, `y`) [m, base frame], lined up with `yaw_deg` [deg] when given."""
        spec = block(kind)
        xy = np.array([x, y], float)
        bearing = self.bearing(xy)
        for size in self.tilts():
            for psi in self._headings(spec.yaw_period, bearing, size, yaw_deg):
                tilt = math.copysign(size, math.sin(bearing - psi)) if size else 0.0
                if self._fits(kind, xy, bottom_z, psi, tilt):
                    return PlaceReach(True, math.degrees(size), math.degrees(wrap(psi)))
        return PlaceReach(False)

    def map(self, kind: str, bottom_z: float, *, x_m: tuple[float, float] = (0.0, 0.4),
            y_m: tuple[float, float] = (-0.3, 0.3), step_m: float = 0.02) -> ReachMap:
        """The least tilt [deg] over a grid of `step_m` on the table, for a block placed square to the arm."""
        xs = np.arange(x_m[0], x_m[1] + 1e-9, step_m)
        ys = np.arange(y_m[0], y_m[1] + 1e-9, step_m)
        k = self.kin.spec.joints.index(self.pan)
        lo, hi = float(self.lo[k]), float(self.hi[k])
        rings: dict[float, float] = {}
        out = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                d = np.array([x, y]) - self.pan_xy
                if not lo <= math.atan2(d[1], d[0]) <= hi:
                    continue
                r = round(float(np.linalg.norm(d)) / step_m) * step_m
                if r not in rings:
                    got = self.place(kind, *(self.pan_xy + np.array([r, 0.0])), bottom_z)
                    rings[r] = got.tilt_deg
                out[i, j] = rings[r]
        return ReachMap(xs, ys, out)

    def bearing(self, xy: np.ndarray) -> float:
        """The direction of `xy` from the base's turning axis [rad, base frame]."""
        d = np.asarray(xy, float)[:2] - self.pan_xy
        return math.atan2(float(d[1]), float(d[0]))

    def _headings(self, period: float, bearing: float, tilt: float, yaw_deg: float | None) -> list[float]:
        """Closing headings to try [rad]: square to the arm first, or across the block lined up with
        `yaw_deg`, within the block-tilt limit of square when tilted."""
        square = [bearing - math.pi / 2, bearing + math.pi / 2]
        if yaw_deg is None:
            return square if tilt else [*square, bearing]
        across = math.radians(yaw_deg) + math.pi / 2
        turns = round(2 * math.pi / period)
        lined = [wrap(across + k * period) for k in range(max(1, turns))]
        lined += [wrap(psi + math.pi) for psi in lined]
        if not tilt:
            return lined
        limit = math.sin(math.radians(MAX_ROT_ERR_DEG))
        return [psi for psi in lined if math.sin(tilt) * abs(math.cos(bearing - psi)) <= limit]

    def _fits(self, kind: str, xy: np.ndarray, bottom_z: float, psi: float, tilt: float) -> bool:
        kin, cfg, spec = self.kin, _SETTINGS, block(kind)
        width, height = 2 * spec.half_size[1], 2 * spec.half_size[2]
        try:
            cmd_open = kin.gripper_for_gap(width + 2 * cfg.open_margin_m)
            cmd_block = kin.gripper_for_gap(width)
            cmd_release = kin.gripper_for_gap(min(width + 2 * cfg.release_margin_m, kin.jaw_gap(FULL_OPEN)))
        except ValueError:
            return False
        open_c, block_c = kin.jaw_centre(cmd_open), kin.jaw_centre(cmd_block)
        # The TCP over the block's bottom and the block's centre in the hand, as a pick from the table
        # leaves them.
        hang = max(MIN_GRASP_Z, height * (1 - cfg.grasp_depth) + float(open_c[0]) * math.cos(tilt))
        hand = np.array([float(block_c[2]), float(open_c[0]) * math.sin(tilt)])
        tcp = xy - rot2(psi) @ hand
        release_z = bottom_z + cfg.place_gap_m + GRIP_SLIP_M + hang
        placed_top = bottom_z + cfg.place_gap_m + height
        carry_z = max(release_z, max(bottom_z, TABLE_TOP_M) + MIN_CARRY_CLEAR_M + hang)
        at_release = self._reaches(np.array([*tcp, release_z]), psi, tilt, self.seed)
        if at_release is None or self._reaches(np.array([*tcp, carry_z]), psi, tilt, at_release) is None:
            return False
        eased = tcp - cfg.release_margin_m * np.array([math.cos(psi), math.sin(psi)])
        clear_z = max(release_z, placed_top + JAW_CLEAR_M + kin.jaw_reach(cmd_release, tilt=tilt))
        return self._reaches(np.array([*eased, clear_z]), psi, tilt, at_release) is not None

    def _reaches(self, point: np.ndarray, psi: float, tilt: float,
                 seed: Mapping[str, float]) -> dict[str, float] | None:
        kin = self.kin
        sol = kin.ik(point, hand_down(psi, tilt), seed=seed, accept=self._clear, restarts=ik_restarts(tilt))
        if sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG):
            return sol.joints
        return None

    def _clear(self, joints: Mapping[str, float]) -> bool:
        q, arm = self.kin.to_q(joints), self.kin.arm
        within = np.all(q[arm] >= self.lo[arm] - 1e-9) and np.all(q[arm] <= self.hi[arm] + 1e-9)
        return bool(within) and not self.kin.self_contacts(joints)


_DEFAULT: Reach | None = None


def _reach(kin: RobotKinematics | None, max_tilt_deg: float) -> Reach:
    global _DEFAULT
    if kin is not None:
        return Reach(kin, max_tilt_deg=max_tilt_deg)
    if _DEFAULT is None:
        _DEFAULT = Reach()
    _DEFAULT.max_tilt_deg = float(max_tilt_deg)
    return _DEFAULT


def can_place(kind: str, x: float, y: float, bottom_z: float, *, yaw_deg: float | None = None,
              max_tilt_deg: float = _SETTINGS.max_tilt_deg, kin: RobotKinematics | None = None) -> PlaceReach:
    """Whether the arm can place a `kind` block with its bottom at `bottom_z` [m above the table] and its
    centre at (`x`, `y`) [m, base frame], straight down or tilted. See `Reach.place`."""
    return _reach(kin, max_tilt_deg).place(kind, x, y, bottom_z, yaw_deg)


def reach_map(kind: str, bottom_z: float, *, max_tilt_deg: float = _SETTINGS.max_tilt_deg,
              x_m: tuple[float, float] = (0.0, 0.4), y_m: tuple[float, float] = (-0.3, 0.3),
              step_m: float = 0.02, kin: RobotKinematics | None = None) -> ReachMap:
    """The least hand tilt that places a `kind` block with its bottom at `bottom_z` over a table grid.
    See `Reach.map`."""
    return _reach(kin, max_tilt_deg).map(kind, bottom_z, x_m=x_m, y_m=y_m, step_m=step_m)
