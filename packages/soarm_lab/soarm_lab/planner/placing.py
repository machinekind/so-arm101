"""Where the arm can set a block down, from the arm model.

A `PlaceCheck` answers whether the arm can put a block with its bottom at height z over (x, y). The
default, `HandDownReach`, runs IK with the hand pointing straight down and the jaw open around the
block, at the release point and where the hand comes in over the place. It takes the better of two
hand headings: the jaw closing along or across the arm. Another check, such as one that lets the hand
tilt, can replace it wherever a `PlaceCheck` is taken.

The first joint turns the whole arm about a vertical axis, so reach depends only on the distance from
that axis and the height. Results are cached on a grid of those two, and the first joint's limits are
checked per query. A query passes only when the grid distances on both sides of it pass with the
heights rounded up, so between grid points the check errs toward less reach.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

import numpy as np

from ..blocks import BlockSpec
from ..kinematics import RobotKinematics
from ..robot import RobotSpec, load_robot
from ..strategies.grasp import MAX_ROT_ERR_DEG, MIN_GRASP_Z, GraspSettings, top_down

#: Whether the arm can set `block` down with its bottom `z` [m] above the table at (`x`, `y`) [m, base frame].
PlaceCheck = Callable[[float, float, float, BlockSpec], bool]

#: Grid steps of the distance from the first joint's axis and of the TCP height [m].
DISTANCE_STEP_M = 0.01
HEIGHT_STEP_M = 0.001
#: Jaw headings tried, relative to the direction from the first joint's axis to the place [rad].
HEADINGS = (0.0, math.pi / 2)
#: Gap left under the block at the release [m].
PLACE_GAP_M = 0.002
#: The block's bottom passes this far over the place's surface as the hand comes in [m].
CARRY_CLEAR_M = 0.01
#: IK iterations per grid point, from the rest pose.
IK_ITERS = 100


class HandDownReach:
    """`PlaceCheck` from IK with the hand pointing down, cached per grid point."""

    def __init__(self, spec: RobotSpec):
        if spec.gripper is None:
            raise ValueError(f"{spec.name}: no gripper in the robot description")
        self.kin = kin = RobotKinematics(spec)
        self.spec = spec
        a = kin.to_q({name: spec.limits.get(name, (-math.inf, math.inf))[0] for name in spec.joints})
        b = kin.to_q({name: spec.limits.get(name, (-math.inf, math.inf))[1] for name in spec.joints})
        self.lo = np.maximum(np.minimum(a, b), kin.lo)
        self.hi = np.minimum(np.maximum(a, b), kin.hi)
        m, first = kin.model, spec.joints[0]
        joint = m.joint(kin.prefix + first)
        body = m.body(int(joint.bodyid[0])).name.removeprefix(kin.prefix)
        T = kin.body(body, spec.home)
        axis = T[:3, :3] @ np.asarray(joint.axis, float)
        if abs(axis[2]) < 0.999:
            raise ValueError(f"{spec.name}: the first joint '{first}' does not turn about a vertical axis")
        #: The first joint's axis in the table plane [m, base frame], and its turn per bearing (+1 or -1).
        self.axis_xy = (T[:3, :3] @ np.asarray(joint.pos, float) + T[:3, 3])[:2]
        self.turn_sign = 1.0 if axis[2] > 0 else -1.0
        #: Grid point -> the first joint's angle there at bearing 0 [rad], or None when unreachable.
        self._cells: dict[tuple[int, int, int, float], float | None] = {}
        self._grips: dict[str, tuple[float, float]] = {}

    def __call__(self, x: float, y: float, z: float, block: BlockSpec) -> bool:
        dx, dy = x - float(self.axis_xy[0]), y - float(self.axis_xy[1])
        distance, bearing = math.hypot(dx, dy), math.atan2(dy, dx)
        cmd, hang = self._grip(block)
        k = round(distance / DISTANCE_STEP_M, 6)
        rings = {math.floor(k), math.ceil(k)}
        heights = {math.ceil(round((z + up + hang) / HEIGHT_STEP_M, 6))
                   for up in (PLACE_GAP_M, CARRY_CLEAR_M)}
        for h in range(len(HEADINGS)):
            pans = (self._cell(h, ring, level, cmd) for ring in rings for level in heights)
            if all(p is not None and self.lo[0] <= p + self.turn_sign * bearing <= self.hi[0] for p in pans):
                return True
        return False

    def _grip(self, block: BlockSpec) -> tuple[float, float]:
        """The gripper command that opens the jaw around the block, and the TCP height above the block's
        bottom while it is held [m], both as `grasp` takes a block from the table."""
        if block.name not in self._grips:
            kin, cfg = self.kin, GraspSettings
            cmd = kin.gripper_for_gap(2 * min(block.half_size[:2]) + 2 * cfg.open_margin_m)
            height = 2 * block.half_size[2]
            hang = max(MIN_GRASP_Z, height * (1 - cfg.grasp_depth) + float(kin.jaw_centre(cmd)[0]))
            self._grips[block.name] = (cmd, hang)
        return self._grips[block.name]

    def _cell(self, heading: int, ring: int, level: int, cmd: float) -> float | None:
        """The first joint's angle at grid distance `ring` and TCP height `level`, in grid steps."""
        key = (heading, ring, level, round(cmd, 1))
        if key not in self._cells:
            self._cells[key] = self._solve(HEADINGS[heading], ring * DISTANCE_STEP_M,
                                           level * HEIGHT_STEP_M, cmd)
        return self._cells[key]

    def _solve(self, heading: float, distance: float, tcp_z: float, cmd: float) -> float | None:
        """The first joint's angle with the TCP `distance` in front of its axis, hand down, or None."""
        kin, spec = self.kin, self.spec
        assert spec.gripper is not None
        point = np.array([self.axis_xy[0] + distance, self.axis_xy[1], tcp_z])
        seed = {**spec.home, spec.gripper: cmd}
        sol = kin.ik_step(point, top_down(heading), seed, iters=IK_ITERS)
        if not (sol.ok and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)):
            return None
        q = kin.to_q(sol.joints)
        if np.any(q[kin.arm] < self.lo[kin.arm] - 1e-9) or np.any(q[kin.arm] > self.hi[kin.arm] + 1e-9):
            return None
        if kin.self_contacts(sol.joints):
            return None
        return float(q[0])


@cache
def arm_reach() -> HandDownReach:
    """The hand-down check for the bundled SO-101, shared so its cache fills once per process."""
    return HandDownReach(load_robot())
