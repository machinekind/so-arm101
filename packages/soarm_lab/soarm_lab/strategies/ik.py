"""Inverse kinematics as a strategy for the twin: `ik` under the `machinekind_twin.strategies` entry points.

The twin finds `IKStrategy` through the entry point, builds `IKSettings` from its config file's `ik:`
section or the panel's form, and runs the strategy through its `StrategyRunner`, as it runs `policy`.
Nothing here imports the twin: the contract types come from `machinekind_twin_api.strategy`.

    # configs/local.yaml of the twin: prefills the panel's form
    ik:
      goal_x: 0.22                 # the goal [m, base frame] when the twin gives none
      goal_y: 0.0
      goal_z: 0.12
      speed_mps: 0.10              # highest TCP speed along the path

No policy, no training: `kinematics.RobotKinematics` alone drives the TCP to the goal. Every tick a
point (the carrot) moves from the TCP of the last target straight toward the goal by at most
`speed_mps / hz`, and `RobotKinematics.ik_step`, started from the last target, turns it into joints.
The TCP follows a straight line and consecutive targets stay near each other. A carrot the arm cannot
reach keeps the target at the edge of the reach, and the arm follows again when the goal comes back.

Guards:

* **reach check**: `start` refuses a goal the full multi-start `RobotKinematics.ik` cannot reach within
  the limits on the arm. A goal dragged during the run is not checked again (the full IK takes tens of
  ms); the note says when the arm cannot follow it;
* **stop on lag**: as in `policy`, the run ends on the measured pose when an arm joint falls behind its
  target by more than `max_tracking_deg`;
* **end on reach**: with `end_on_reach` the run ends as a success once the last target puts the TCP on
  the goal and the measured TCP is within `tol_m` of it. Without it the run follows the goal until the
  twin stops it;
* **the gripper never moves**: it keeps its start angle, or the command it squeezes something with (as
  `policy` keeps it).

The goal is `obs.goal`, read every tick; without one the settings' `goal_x`, `goal_y`, `goal_z` hold it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo, Strategy

from ..kinematics import RobotKinematics
from ..robot import load_robot

logger = logging.getLogger(__name__)


@dataclass
class IKSettings:
    #: The goal when the observation has none [m, base frame].
    goal_x: float = 0.22
    goal_y: float = 0.0
    goal_z: float = 0.12
    #: Ticks per second: the twin's runner ticks at this rate.
    hz: float = 20.0
    #: Highest TCP speed along the path [m/s].
    speed_mps: float = 0.10
    #: The measured TCP is at the goal within this distance [m].
    tol_m: float = 0.005
    #: End the run as a success once the TCP is at the goal; False keeps following a goal the panel drags.
    end_on_reach: bool = True
    #: The run ends, holding the measured pose, when an arm joint lags its target by more than this [deg].
    max_tracking_deg: float = 25.0


@dataclass
class IKStatus:
    """What the strategy knows about its run, for logs and tests (the twin keeps its own runner status)."""

    #: Ticks done; the tick that stops the run on lag does not count.
    step: int = 0
    #: The measured TCP's distance from the goal [m].
    distance: float = float("nan")
    #: The last target puts the TCP on the goal and the measured TCP is within `tol_m` of it.
    success: bool = False
    stopped_because: str = ""
    #: The last target misses its carrot: the arm cannot follow the goal.
    out_of_reach: bool = False


class IKStrategy(Strategy[IKSettings]):
    """Inverse kinematics driving the TCP to a goal point along a straight line, no policy."""

    settings = IKSettings

    def __init__(self, settings: IKSettings | None = None):
        self.config = settings if settings is not None else IKSettings()
        for name in ("hz", "speed_mps", "tol_m", "max_tracking_deg"):
            value = getattr(self.config, name)
            if not value > 0:
                raise ValueError(f"ik: {name} must be positive, is {value}")
        self.hz = float(self.config.hz)
        #: A run with `end_on_reach` ends itself; one without runs until the twin stops it.
        self.time_limit_s = None
        self.status = IKStatus()
        self.max_tracking = np.radians(self.config.max_tracking_deg)
        #: The longest carrot step [m]: `speed_mps` over one tick.
        self.step_m = self.config.speed_mps / self.hz
        #: How close IK puts the TCP to the carrot [m]: at most 1 mm, and a quarter of the step or of
        #: `tol_m` when those are smaller, so a short step still moves the arm.
        self.ik_tol = min(1e-3, self.step_m / 4, self.config.tol_m / 4)
        self._default_goal = np.array([self.config.goal_x, self.config.goal_y, self.config.goal_z], float)
        self.goal = self._default_goal.copy()
        # Known from `start`: the arm's kinematics and the limits that apply on it [rad].
        self._kin: RobotKinematics | None = None
        self._lo = self._hi = np.zeros(0)
        #: The last target [rad, the description's order]; its gripper never changes during a run.
        self.q_cmd = np.zeros(0)
        #: The TCP of the last target and the point the last tick asked IK for [m, base frame].
        self.tcp_cmd = np.zeros(3)
        self.carrot = np.zeros(3)

    @property
    def started(self) -> bool:
        """Whether `start` ran: the arm's kinematics and limits are known."""
        return self._kin is not None

    @property
    def kin(self) -> RobotKinematics:
        """The arm's kinematics, known from `start`."""
        if self._kin is None:
            raise RuntimeError("ik: start the run first")
        return self._kin

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of the targets [rad], known from `start` (`_hardware_limits`)."""
        if self._kin is None:
            raise RuntimeError("ik: start the run first")
        return self._lo, self._hi

    # ------------------------------------------------------------------- start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Starts from where the arm is. Raises to refuse the run: a goal out of the arm's reach."""
        spec = load_robot(run.folder)
        self._kin = RobotKinematics(spec)
        self._lo, self._hi = self._hardware_limits(run.limits)

        measured = dict(obs.joints)
        self.q_cmd = np.clip(self.kin.to_q(measured), self._lo, self._hi)
        grip = self._squeezing_grip_cmd(measured, obs.command)
        if grip is not None:
            self.q_cmd[spec.joints.index(spec.gripper)] = grip
        self.status = IKStatus()
        self._take_goal(obs)

        sol = self.kin.ik(self.goal, seed=self.kin.from_q(self.q_cmd), accept=self._within_limits)
        if not (sol.ok and self._within_limits(sol.joints)):
            x, y, z = self.goal
            raise RuntimeError(f"cel ({x:.3f}, {y:.3f}, {z:.3f}) m poza zasiegiem ramienia")
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()

    def _hardware_limits(self, hw: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """The description's limits and the MJCF ranges, intersected with the ones on the arm (`run.limits`).

        The same limits `policy` clips its targets to. An empty intersection keeps the description's limits.
        """
        kin, spec = self.kin, self.kin.spec
        a = kin.to_q({name: spec.limits[name][0] for name in spec.joints})
        b = kin.to_q({name: spec.limits[name][1] for name in spec.joints})
        base_lo, base_hi = np.maximum(np.minimum(a, b), kin.lo), np.minimum(np.maximum(a, b), kin.hi)
        lo, hi = base_lo.copy(), base_hi.copy()
        for k, name in enumerate(spec.joints):
            if name in hw:
                a, b = kin.to_q({name: hw[name][0]})[k], kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = base_lo[bad], base_hi[bad]
        return lo, hi

    def _within_limits(self, joints: Mapping[str, float]) -> bool:
        """Whether the arm joints of an IK solution lie within the limits that apply on the arm."""
        q, arm = self.kin.to_q(joints), self.kin.arm
        return bool(np.all(q[arm] >= self._lo[arm] - 1e-9) and np.all(q[arm] <= self._hi[arm] + 1e-9))

    def _squeezing_grip_cmd(self, measured: Mapping[str, float],
                            command: Mapping[str, float]) -> float | None:
        """The gripper command angle [rad] from `obs.command` when the gripper SQUEEZES, else None.

        It squeezes when the command is tighter than the measured angle by more than the jaw's `margin`
        (`spec.jaw`), as `policy` decides it. Commanding the measured angle of a blocked jaw is zero
        force: a cube carried to the goal would fall out.
        """
        name = self.kin.spec.gripper
        if name is None or name not in measured:
            return None
        cmd = command.get(name)
        if cmd is None:
            return None
        g = self.kin.spec.joints.index(name)
        q_cmd = float(self.kin.to_q({**measured, name: float(cmd)})[g])
        if float(self.kin.to_q(measured)[g]) - q_cmd <= self.kin.spec.jaw.margin:
            return None
        return float(np.clip(q_cmd, self._lo[g], self._hi[g]))

    def _take_goal(self, obs: Observation) -> None:
        self.goal = self._default_goal.copy() if obs.goal is None else np.asarray(obs.goal, float).copy()

    # --------------------------------------------------------------------- tick

    def act(self, obs: Observation) -> Action:
        """One tick: the next joint targets, or the end of the run."""
        if self._kin is None:
            raise RuntimeError("ik: act() przed start()")
        st = self.status
        kin, arm = self.kin, self.kin.arm
        measured = dict(obs.joints)
        self._take_goal(obs)
        q = kin.to_q(measured)
        lag = float(np.abs(q[arm] - self.q_cmd[arm]).max())
        if lag > self.max_tracking:
            # The arm stops where it IS: the last target may lie past an obstacle.
            return self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                              target=self._hold_pose(measured))

        # The twin's runner drops the target sent with `done`, so success needs the last target on the
        # goal, not only the measured TCP near it.
        to_goal = self.goal - self.tcp_cmd
        d = float(np.linalg.norm(to_goal))
        on_goal = d <= self.ik_tol
        self.carrot = self.goal.copy() if d <= self.step_m else self.tcp_cmd + to_goal * (self.step_m / d)
        sol = kin.ik_step(self.carrot, seed=kin.from_q(self.q_cmd), tol=self.ik_tol)
        q_new = kin.to_q(sol.joints)
        q_new[~arm] = self.q_cmd[~arm]                     # the gripper keeps its target
        self.q_cmd = np.clip(q_new, self._lo, self._hi)
        target = kin.from_q(self.q_cmd)
        self.tcp_cmd = kin.tcp(target)[:3, 3].copy()

        st.step += 1
        st.stopped_because = ""
        st.out_of_reach = bool(np.linalg.norm(self.tcp_cmd - self.carrot) > self.ik_tol)
        st.distance = float(np.linalg.norm(self.goal - kin.tcp(measured)[:3, 3]))
        st.success = on_goal and st.distance <= self.config.tol_m
        if st.success and self.config.end_on_reach:
            return self._halt("cel osiagniety", target=target, success=True)
        return Action(target=target, success=st.success, error=st.distance, note=self._note())

    def _hold_pose(self, measured: Mapping[str, float]) -> dict[str, float]:
        """The last target of a run stopped on lag: the arm joints where they are, the gripper as it was."""
        spec = self.kin.spec
        pose = {name: float(measured[name]) for name in spec.arm_joints if name in measured}
        if spec.gripper is not None:
            pose[spec.gripper] = self.kin.from_q(self.q_cmd)[spec.gripper]
        return pose

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, target: Mapping[str, float], success: bool = False) -> Action:
        """The run is over: the last target and the reason for the twin's panel."""
        self.status.stopped_because = reason
        logger.info("IK zatrzymane: %s", reason)
        return Action(target=dict(target), done=True, success=success, error=self._error(), note=reason)

    def stop(self, reason: str) -> None:
        """The twin ended the run (Stop, STOP, Home, the arm taken, or our own `done`)."""
        # After our own `done` the twin reports "zadanie wykonane" or "strategia skonczyla"; ours says more.
        if reason in ("zadanie wykonane", "strategia skonczyla") and self.status.stopped_because:
            return
        self.status.stopped_because = reason or "zatrzymana"

    def _error(self) -> float | None:
        d = self.status.distance
        return float(d) if np.isfinite(d) else None

    def _note(self) -> str:
        st = self.status
        txt = f"do celu {st.distance * 1000:.0f} mm" + (" (w celu)" if st.success else "")
        return txt + (" - cel poza zasiegiem ramienia" if st.out_of_reach else "")
