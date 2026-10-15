"""The arm's kinematics, computed by MuJoCo on the description from `robot`.

There is no kinematic chain of its own to maintain: forward kinematics is `mj_kinematics` on the same
MJCF that simulates and renders the scene, so kinematics, physics and renders cannot disagree on geometry.

Inverse kinematics is damped least squares on the Jacobian of the TCP site, as in `a1x_control.Arm.ik`
from galaxeo-manipulators, with two changes:

* **Position takes priority over orientation.** The A1X has six arm joints, the SO-101 five: it cannot
  reach an arbitrary full gripper orientation. Weighting both errors in one sum of squares does not work
  here: radians and metres do not compare, and with a goal rotated by 90 degrees the rotation pulls the
  arm 20 cm off the requested point. Instead, orientation is corrected only in the null space of the
  position Jacobian, that is only by motions that do not change the position (task priority). The point
  is hit whenever it is reachable, and the rotation is as close as it can be at that point.
* **Several starts.** A single DLS start gets stuck in a local minimum: galaxeo measured 30 cm of error
  over most of the A1X workspace. IK starts from the given pose, from the rest pose and from a few random
  ones, and keeps the best.

All poses are in the arm's BASE frame, not the scene's world frame: a camera calibrated relative to the
base stays correct wherever the arm stands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

from .robot import RobotSpec


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A 4x4 matrix from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    """The inverse of a rigid transform, without a general matrix inversion."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass
class IKSolution:
    """An inverse kinematics result in app units."""

    joints: dict[str, float]
    #: TCP position error [m] and orientation error [rad] at the reached pose.
    pos_err: float
    rot_err: float
    #: The position is hit within the tolerance. Orientation is soft: see the module docstring.
    ok: bool


class RobotKinematics:
    """The kinematics of one arm on a compiled MuJoCo model.

    `model` may be the arm alone or a whole scene the arm was inserted into under a name prefix
    (`prefix`). The computations run on a separate `MjData` either way, so they do not touch the
    simulation state.
    """

    def __init__(self, spec: RobotSpec, model: mujoco.MjModel | None = None, prefix: str = ""):
        self.spec = spec
        grip = spec.gripper_range_q
        self._grip_lin = None if grip is None else (float(grip[0]), float(grip[1]))
        self.prefix = prefix
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        ids = [m.joint(prefix + name).id for name in spec.joints]
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([m.jnt_dofadr[i] for i in ids])
        self.lo = m.jnt_range[ids, 0].copy()
        self.hi = m.jnt_range[ids, 1].copy()
        #: Mask of the arm joints (gripper excluded): IK moves only these.
        self.arm = np.array([name != spec.gripper for name in spec.joints])
        self.site_id = m.site(prefix + spec.tcp_site).id
        self.base_id = m.body(prefix + spec.base_body).id
        self._approach = np.asarray(spec.tcp_approach, float)
        self._closing = np.asarray(spec.tcp_closing, float)
        #: Coupled joints (`spec.mirror_joints`): (index of the description's joint, qpos address, ratio).
        #: `write_qpos` writes them on every pose write, not only here.
        self.mirror: list[tuple[int, int, float]] = [
            (spec.joints.index(src), int(m.jnt_qposadr[m.joint(prefix + slaved).id]), float(gain))
            for slaved, (src, gain) in spec.mirror_joints.items()
        ]
        #: qvel addresses of the coupled joints: `stop_qvel` zeroes them with the description's joints.
        self.mirror_dadr = np.array(
            [m.jnt_dofadr[m.joint(prefix + slaved).id] for slaved in spec.mirror_joints], dtype=int)

    # ------------------------------------------------------------ units
    def to_q(self, joints: Mapping[str, float]) -> np.ndarray:
        """App units -> the vector of MJCF joint angles [rad], in the description's order.

        Arm joints are degrees of the same calibration, so only the unit changes. The gripper's
        0..100 maps linearly onto `RobotSpec.gripper_range_q` (closed, open), and past it as far as
        the joint's MJCF range: the jaw travel the real servo makes, not the whole MJCF range. On the
        SO-101 the whole range is -10..100 degrees and the servo covers only -5..55 on 0..100,
        so mapping onto the whole range would show the jaw 1.8 times wider open than on the arm.
        Missing joints take their value from the rest pose.
        """
        q = np.empty(len(self.spec.joints))
        for k, name in enumerate(self.spec.joints):
            value = float(joints.get(name, self.spec.home.get(name, 0.0)))
            if name == self.spec.gripper:
                q[k] = self._grip_to_q(k, value)
            else:
                q[k] = np.radians(value)
        return q

    def _grip_to_q(self, k: int, value: float) -> float:
        closed, opened = self._grip_range()
        q = closed + value / 100.0 * (opened - closed)
        return float(min(max(q, self.lo[k]), self.hi[k]))

    def _grip_from_q(self, k: int, q: float) -> float:
        closed, opened = self._grip_range()
        return float((q - closed) * 100.0 / (opened - closed)) if opened != closed else 0.0

    def _grip_range(self) -> tuple[float, float]:
        if self._grip_lin is None:
            raise ValueError(f"{self.spec.name}: chwytak bez `gripper_range_q` w opisie ramienia")
        return self._grip_lin

    def from_q(self, q: np.ndarray) -> dict[str, float]:
        """The MJCF vector [rad] -> app units."""
        out: dict[str, float] = {}
        for k, name in enumerate(self.spec.joints):
            if name == self.spec.gripper:
                out[name] = self._grip_from_q(k, float(q[k]))
            else:
                out[name] = float(np.degrees(q[k]))
        return out

    def clip(self, joints: Mapping[str, float]) -> dict[str, float]:
        """The same joints, clipped to the MJCF ranges."""
        return self.from_q(np.clip(self.to_q(joints), self.lo, self.hi))

    # ------------------------------------------------------------ kinematics
    def write_qpos(self, qpos: np.ndarray, q: np.ndarray) -> None:
        """Writes pose `q` (description order, MJCF units) into the model's `qpos`, with the coupled joints.

        Every place that sets the arm's pose directly (without dynamics) goes through here: otherwise a
        joint coupled by an equality (the A1X's second jaw) would stay where it was, and the solver would
        yank it into line only on the next step.
        """
        qpos[self.qadr] = q
        for k, adr, g in self.mirror:
            qpos[adr] = g * q[k]

    def stop_qvel(self, qvel: np.ndarray) -> None:
        """Zeroes the velocities of the arm joints and the coupled joints: the counterpart of `write_qpos`.

        A pose written without dynamics should stand still: a coupled joint with a velocity left from the
        previous step would drift off the written pose and, through the equality, drag the gripper along.
        """
        qvel[self.dadr] = 0.0
        qvel[self.mirror_dadr] = 0.0

    def _apply(self, q: np.ndarray) -> None:
        self.write_qpos(self.data.qpos, q)
        mujoco.mj_kinematics(self.model, self.data)

    def _base_inv(self) -> np.ndarray:
        d = self.data
        return inverse(pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id]))

    def tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        """The TCP pose in the base frame (4x4)."""
        self._apply(self.to_q(joints))
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[self.site_id].reshape(3, 3), d.site_xpos[self.site_id])

    def body(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """The pose of any body of the arm in the base frame (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.body(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.xmat[i].reshape(3, 3), d.xpos[i])

    def site(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """The pose of any site of the arm in the base frame (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.site(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[i].reshape(3, 3), d.site_xpos[i])

    def tool_axes(self, joints: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(approach direction, jaw closing axis) in the base frame."""
        R = self.tcp(joints)[:3, :3]
        return R @ self._approach, R @ self._closing

    # -------------------------------------------------------------------- IK
    def ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed: Mapping[str, float] | None = None,
        *,
        rot_gain: float = 0.5,
        tol: float = 1e-3,
        iters: int = 300,
        damping: float = 0.02,
        restarts: int = 6,
        rng: np.random.Generator | None = None,
        accept: Callable[[dict[str, float]], bool] | None = None,
    ) -> IKSolution:
        """Joints that put the TCP at `position`, rotated as close to `rotation` as possible.

        Both in the base frame. IK does not move the gripper: the result keeps `seed`'s gripper. Among the
        starts that hit the point, the smallest rotation error wins (when a rotation is asked for), then
        the pose nearest `seed`; if none hit, the one that came closest.
        """
        seed = dict(seed) if seed is not None else dict(self.spec.home)
        rng = rng if rng is not None else np.random.default_rng(0)
        q_seed = self.to_q(seed)

        # Every start carries the seed's gripper, so whichever wins, the gripper comes back unchanged.
        q_home = self.to_q(self.spec.home)
        q_home[~self.arm] = q_seed[~self.arm]
        starts = [q_seed, q_home]
        for _ in range(restarts):
            q = q_seed.copy()
            q[self.arm] = rng.uniform(self.lo[self.arm], self.hi[self.arm])
            starts.append(q)

        best: tuple[tuple[bool, float], np.ndarray, float, float] | None = None
        target = np.asarray(position, float)
        for q0 in starts:
            q, e_pos, e_rot = self._solve(q0, target, rotation, rot_gain, iters, damping, tol)
            hit = e_pos < tol
            # A hit the caller refuses (a collision) counts as a miss, so another start can win.
            if hit and accept is not None and not accept(self.from_q(q)):
                hit = False
            # Hits first; among hits the rotation error (when one was asked for), then the pose
            # nearest the seed, so a reachable point does not come back on the other elbow branch.
            far = float(np.max(np.abs(q - q_seed)))
            key = (not hit, round(e_rot, 2) if (hit and rotation is not None) else 0.0, far if hit else e_pos)
            if best is None or key < best[0]:
                best = (key, q, e_pos, e_rot)

        assert best is not None
        _, q, e_pos, e_rot = best
        return IKSolution(self.from_q(q), float(e_pos), float(e_rot), bool(e_pos < tol))

    def ik_step(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed: Mapping[str, float] | None = None,
        *,
        iters: int = 30,
        rot_gain: float = 0.5,
        tol: float = 1e-3,
        damping: float = 0.02,
    ) -> IKSolution:
        """One start from `seed` (no restarts, no `accept`): for tracking a point on every loop tick.

        The goal moves by millimetres between ticks, so the previous tick's solution is a good start:
        `_solve` converges in a few iterations (about 0.8 ms, measured together with a collision check),
        while the full `ik` with eight starts takes tens of ms. The result stays near `seed`, without the
        jump to the other elbow branch a random start could pick.
        """
        seed = dict(seed) if seed is not None else dict(self.spec.home)
        q, e_pos, e_rot = self._solve(self.to_q(seed), np.asarray(position, float), rotation, rot_gain, iters,
                                      damping, tol)
        return IKSolution(self.from_q(q), float(e_pos), float(e_rot), bool(e_pos < tol))

    def _solve(
        self,
        q: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray | None,
        rot_gain: float,
        iters: int,
        damping: float,
        tol: float,
    ) -> tuple[np.ndarray, float, float]:
        """One start: position by the pseudo-inverse, rotation by a gradient in the null space."""
        m, d = self.model, self.data
        q = np.clip(q.copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.dadr[self.arm]
        n = int(self.arm.sum())
        want_quat = np.zeros(4)
        cur_quat = np.zeros(4)
        err_quat = np.zeros(4)
        w = np.zeros(3)
        e_pos = e_rot = float("inf")

        self._apply(q)
        # The base does not move, so the target goes to the world frame once.
        base = pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id])
        target = base[:3, :3] @ position + base[:3, 3]
        if rotation is not None:
            mujoco.mju_mat2Quat(want_quat, (base[:3, :3] @ np.asarray(rotation, float)).ravel())

        for _ in range(iters):
            self._apply(q)
            e_p = target - d.site_xpos[self.site_id]
            e_pos = float(np.linalg.norm(e_p))
            if rotation is not None:
                mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site_id])
                mujoco.mju_negQuat(cur_quat, cur_quat)
                mujoco.mju_mulQuat(err_quat, want_quat, cur_quat)
                mujoco.mju_quat2Vel(w, err_quat, 1.0)
                e_rot = float(np.linalg.norm(w))
            else:
                e_rot = 0.0

            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            Jp = jacp[:, cols]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping**2 * np.eye(3))
            dq = Jp_pinv @ e_p
            if rotation is not None:
                # The secondary task projected onto the position's null space: a motion that
                # improves the rotation without moving the TCP point.
                null = np.eye(n) - Jp_pinv @ Jp
                dq_rot = null @ (rot_gain * jacr[:, cols].T @ w)
                dq = dq + dq_rot
                if e_pos < tol * 0.5 and float(np.linalg.norm(dq_rot)) < 1e-4:
                    break  # the point is hit; the rotation could improve only at the point's cost
            elif e_pos < tol * 0.5:
                break

            # Step limit as in galaxeo: a large jump under poor conditioning throws the solution
            # onto another branch, and the iteration does not come back from it.
            step = min(1.0, 0.3 / (float(np.linalg.norm(dq)) + 1e-9))
            q[self.arm] = np.clip(q[self.arm] + step * dq, self.lo[self.arm], self.hi[self.arm])

        self._apply(q)
        e_pos = float(np.linalg.norm(target - d.site_xpos[self.site_id]))
        return q, e_pos, e_rot
