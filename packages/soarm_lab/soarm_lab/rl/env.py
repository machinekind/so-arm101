"""The Gymnasium environment on the CPU: `SoArmLab/Reach-v0`, `SoArmLab/Lift-v0`, `SoArmLab/PickPlace-v0`.

    import gymnasium as gym, soarm_lab.rl
    env = gym.make("SoArmLab/Reach-v0")
    obs, info = env.reset(seed=0)

This is the "one environment, plain MuJoCo" version: for checking policies, debugging the reward,
`check_env` and SB3-style libraries. Training on thousands of worlds at once uses `batch.BatchEnv`
on MuJoCo Warp; both compute the observation and the reward with the same functions from `task`.

The scene is built ONCE (293 ms on a laptop, 111 ms on an RTX A4500); a reset changes only the state
and fields of the compiled model. The station (table, cameras, base pose) comes from the twin's export
(`station.Station`, `lerobot-twin export`); by default it is the `sim` station bundled with the package.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .. import sim as sc
from ..kinematics import inverse, pose
from ..station import Station
from . import task as tk
from .randomize import Randomization

#: The preview camera, added when the station has no cameras (`scene_config(view=True)`).
VIEW = "widok"
#: The pickplace target pad: half sides [m] and colour (blue; the cube is red).
TARGET_SIZE = (0.035, 0.035, 0.002)
TARGET_RGBA = (0.2, 0.4, 0.9, 1.0)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


def scene_config(task: tk.TaskConfig, station: Station | None = None, view: bool = False) -> sc.SceneConfig:
    """The task's scene: the station (table, cameras) plus the task's objects."""
    ws = station or Station.load()
    cfg = sc.SceneConfig(ws)
    if task.cube:
        h = task.cube_half
        cfg.objects = [*cfg.objects, sc.Box(task.cube, (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                            mass=task.cube_mass)]
        cfg.grasp_sensors = [*cfg.grasp_sensors, task.cube]
    if task.target:
        # A mocap body without collisions: the cube lies on the table top, so `height` means the same
        # as in lift.
        cfg.objects = [*cfg.objects, sc.Box(task.target, TARGET_SIZE, (0.3, 0.1), rgba=TARGET_RGBA, free=False,
                                            mocap=True)]
    if view and not cfg.cameras:
        # Scaled to the arm's desk (SO-101: 0.56 m, a factor of exactly 1).
        f = ws.spec().desk.map_side / 0.56
        cfg.cameras = [sc.CameraView.from_fov(VIEW, 480, 360, 50.0, look_at(np.array([0.62, -0.42, 0.42]) * f,
                                                                           np.array([0.18, 0.0, 0.05]) * f))]
    return cfg


class ModelFields:
    """The nominal model fields that randomization multiplies, stored once."""

    def __init__(self, scene: sc.Scene, cube: str | None):
        m = scene.model
        self.act = scene.act_ids
        self.dof = scene.kin.dadr
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.bias = m.actuator_biasprm[self.act, 1:3].copy()
        self.damping = m.dof_damping[self.dof].copy()
        self.armature = m.dof_armature[self.dof].copy()
        self.frictionloss = m.dof_frictionloss[self.dof].copy()
        self.cube_body = m.body(cube).id if cube else None
        self.cube_geom = m.geom(cube).id if cube else None
        self.cube_mass = m.body_mass[self.cube_body] if cube else 0.0
        self.cube_inertia = m.body_inertia[self.cube_body].copy() if cube else None
        self.cube_friction = m.geom_friction[self.cube_geom].copy() if cube else None

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData, s: dict[str, float]) -> None:
        """Applies the multipliers `s` to the model fields, always from the nominal (they never pile up)."""
        m.actuator_gainprm[self.act, 0] = self.kp * s["kp"]
        m.actuator_biasprm[self.act, 1] = self.bias[:, 0] * s["kp"]
        m.dof_damping[self.dof] = self.damping * s["damping"]
        m.dof_armature[self.dof] = self.armature * s["armature"]
        m.dof_frictionloss[self.dof] = self.frictionloss * s["frictionloss"]
        if self.cube_body is not None:
            # The mass together with the inertia: it is the same object, only heavier.
            m.body_mass[self.cube_body] = self.cube_mass * s["cube_mass"]
            m.body_inertia[self.cube_body] = self.cube_inertia * s["cube_mass"]
            m.geom_friction[self.cube_geom] = self.cube_friction * np.array([s["cube_friction"], 1.0, 1.0])
        # Mass and armature enter constants computed at compile time (body_invweight0,
        # dof_invweight0), and the softness of contacts depends on them. Without recomputing them a
        # cube with a changed mass gets the contacts of another cube, and when it jams between the
        # jaw and the table it flies off at 10-180 m/s.
        mujoco.mj_setConst(m, d)


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task: str | tk.TaskConfig = "reach", *, station: Station | None = None,
                 randomization: Randomization | None = None, render_mode: str | None = None,
                 render_camera: str | None = None):
        station = station or Station.load()
        self.task = task if isinstance(task, tk.TaskConfig) else tk.make_task(task, spec=station.spec())
        self.render_mode = render_mode
        self.scene = sc.build(scene_config(self.task, station, view=render_mode is not None))
        self.render_camera = render_camera or (self.scene.cfg.cameras[0].name if self.scene.cfg.cameras else None)
        self.kin = self.scene.kin
        self.limits = tk.Limits.of(self.kin)
        self.rand = randomization if randomization is not None else Randomization()
        self.fields = ModelFields(self.scene, self.task.cube)
        m = self.scene.model
        m.opt.timestep = self.task.timestep
        self.T_world2base = inverse(self.scene.T_base2world)
        self.site = self.kin.site_id
        if self.task.cube:
            self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.fields.cube_body]]
            self.cube_dadr = m.jnt_dofadr[m.body_jntadr[self.fields.cube_body]]
            self.jaw_adr = [m.sensor(f"{self.task.cube}_jaw{k}").adr[0] for k in range(2)]
        if self.task.target:
            self.target_mocap = int(m.body_mocapid[m.body(self.task.target).id])

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.task.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.task.act_dim,), np.float32)
        self.goal = np.zeros(3)
        #: pickplace: the place target on the table top (z = 0), in the base frame.
        self.target = np.zeros(3)
        #: pickplace: the "cube was carried" latch (`task.carry_latch`), cleared in `reset`.
        self._carried = False
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(self.task.act_dim)
        self.t = 0
        self._streak = 0                     # success ticks in a row (`task.end_on_success`)
        self._delay: list[np.ndarray] = []
        self._noise = 0.0
        self._cube_delay, self._cube_period = 0, 1
        self._cube_hist: list[tuple[np.ndarray, np.ndarray]] = []
        self._cube_seen = (np.zeros(3), np.eye(3))

    # ------------------------------------------------------------ state
    def state(self) -> dict[str, np.ndarray]:
        """Everything the observation and the reward are computed from, in the base frame."""
        d = self.scene.data
        R, t = self.T_world2base[:3, :3], self.T_world2base[:3, 3]
        out = {"q": d.qpos[self.kin.qadr].copy(), "tcp": R @ d.site_xpos[self.site] + t}
        if self.task.cube:
            b = self.fields.cube_body
            out["cube_pos"] = R @ d.xpos[b] + t
            out["cube_rot"] = R @ d.xmat[b].reshape(3, 3)
            out["jaws"] = np.array([d.sensordata[a] for a in self.jaw_adr])
            if self.task.target:
                a = self.cube_dadr
                out["cube_vel"] = R @ d.qvel[a:a + 3]
                out["target"] = self.target.copy()
        return out

    def _obs(self, s: dict[str, np.ndarray]) -> np.ndarray:
        q = s["q"] + np.radians(self.np_random.normal(0.0, self._noise, self.task.n_joints)) if self._noise > 0 else s["q"]
        cube_pos, cube_rot = self._cube_seen if self.task.cube else (np.zeros(3), np.eye(3))
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], s["tcp"][None],
                         self.prev_action[None], goal=self.goal[None],
                         cube_pos=cube_pos[None], cube_rot=cube_rot[None], target=self.target[None])
        return obs[0].astype(np.float32)

    def _perceive(self, s: dict[str, np.ndarray], force: bool = False) -> None:
        """The cube in the observation as the cameras see it: delayed, less often, with noise.

        The reward is computed from the truth; the policy sees what the twin's camera perception
        gives it on the desk.
        """
        if not self.task.cube:
            return
        self._cube_hist.append((s["cube_pos"].copy(), s["cube_rot"].copy()))
        del self._cube_hist[:-(self._cube_delay + 1)]
        if not force and self.t % self._cube_period:
            return
        pos, rot = self._cube_hist[max(0, len(self._cube_hist) - 1 - self._cube_delay)]
        pos = pos.copy()
        if self.rand.cube_noise > 0:
            pos[:2] += self.np_random.normal(0.0, self.rand.cube_noise, 2)
        if self.rand.fold_yaw:
            rot = tk.fold_yaw(np, rot)
        self._cube_seen = (pos, rot)

    # ------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        m, d = self.scene.model, self.scene.data
        s = {k: float(v[0]) for k, v in self.rand.sample(rng, 1).items()}
        self.fields.apply(m, d, s)
        self._noise = self.rand.obs_noise_deg
        self._delay = [np.zeros(self.task.act_dim)] * int(s["delay"])
        self._cube_delay, self._cube_period = int(s["cube_delay"]), int(s["cube_period"])
        self._cube_hist = []

        mujoco.mj_resetData(m, d)
        q0 = np.asarray(options["q"], float) if "q" in options else tk.sample_start(self.task, self.limits, rng, 1)[0]
        self.kin.write_qpos(d.qpos, q0)
        d.ctrl[self.scene.act_ids] = q0
        if self.task.name == "reach":
            self.goal = np.asarray(options["goal"], float) if "goal" in options else \
                tk.sample_goals(self.kin, self.task, rng, 1)[0]
        else:
            if "cube" in options:
                pos, quat = np.asarray(options["cube"][0], float), np.asarray(options["cube"][1], float)
            else:
                p, qt = tk.sample_cubes(self.task, rng, 1)
                pos, quat = p[0], qt[0]
            self.set_cube(pos, quat)
            if self.task.target:
                tgt = np.asarray(options["target"], float) if "target" in options else \
                    tk.sample_targets(self.task, rng, pos[None])[0]
                self.set_target(tgt)
        mujoco.mj_forward(m, d)
        self.q_cmd = q0.copy()
        self.prev_action = np.zeros(self.task.act_dim)
        self.t = 0
        self._streak = 0
        self._carried = False
        s0 = self.state()
        self._perceive(s0, force=True)
        return self._obs(s0), {"success": False, "finished": False}

    def set_cube(self, pos_base: np.ndarray, quat_base: np.ndarray) -> None:
        """Puts the cube at a pose given in the base frame (quaternion w, x, y, z)."""
        d = self.scene.data
        T = self.scene.T_base2world
        q_base = np.zeros(4)
        mujoco.mju_mat2Quat(q_base, T[:3, :3].ravel())
        q_world = np.zeros(4)
        mujoco.mju_mulQuat(q_world, q_base, np.asarray(quat_base, float))
        a = self.cube_qadr
        d.qpos[a:a + 3] = T[:3, :3] @ pos_base + T[:3, 3]
        d.qpos[a + 3:a + 7] = q_world
        d.qvel[self.cube_dadr:self.cube_dadr + 6] = 0.0

    def set_target(self, pos_base: np.ndarray) -> None:
        """Sets the pickplace target at (x, y) of the base frame; z is always 0 (the table top).

        The mocap pad follows it.
        """
        p = np.asarray(pos_base, float)
        self.target = np.array([p[0], p[1], 0.0])
        T = self.scene.T_base2world
        self.scene.data.mocap_pos[self.target_mocap] = T[:3, :3] @ np.array([p[0], p[1], TARGET_SIZE[2]]) + T[:3, 3]

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        executed = action
        if self._delay:
            self._delay.append(action)
            executed = self._delay.pop(0)
        self.q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], executed[None])[0]
        d = self.scene.data
        d.ctrl[self.scene.act_ids] = self.q_cmd
        mujoco.mj_step(self.scene.model, d, nstep=self.task.substeps)
        # After mj_step qpos is already integrated, but site_xpos/xpos/xmat come from the forward
        # pass BEFORE it (5 ms older). In motion that TCP differs from FK(q) by 4-8 mm, while the
        # strategy on the arm computes the TCP exactly from q: the policy would learn a relation the
        # desk does not have. So the kinematics run again: everything from the same q.
        mujoco.mj_kinematics(self.scene.model, d)
        self.t += 1

        s = self.state()
        kw = {}
        if self.task.target:
            carried = tk.carry_latch(np, self.task, np.array([self._carried]), s["jaws"][None], s["cube_pos"][None])
            self._carried = bool(carried[0])
            kw = dict(target=self.target[None], cube_vel=s["cube_vel"][None], carried=carried,
                      grip_open=tk.grip_is_open(np, self.limits, self.q_cmd[None]))
        r, success, failure = tk.reward(np, self.task, s["tcp"][None], action[None], self.prev_action[None],
                                        goal=self.goal[None], cube_pos=s.get("cube_pos", np.zeros(3))[None],
                                        jaw_contacts=s.get("jaws", np.zeros(2))[None],
                                        q_cmd=self.q_cmd[None], limits=self.limits, **kw)
        self.prev_action = action
        streak, finished = tk.success_streak(np, self.task, np.array([self._streak]), success)
        self._streak = int(streak[0])
        info = {"success": bool(success[0]), "finished": bool(finished[0])}
        if self.task.name == "reach":
            info["distance"] = float(np.linalg.norm(self.goal - s["tcp"]))
        else:
            info["height"] = float(s["cube_pos"][2] - self.task.cube_half)
            info["grasped"] = bool(s["jaws"].min() > 0.5)
        if self.task.target:
            d_t = float(np.linalg.norm(s["cube_pos"][:2] - self.target[:2]))
            info["dist_target"] = d_t
            info["carried"] = self._carried
            info["placed"] = bool(d_t < self.task.place_radius and info["height"] < self.task.place_height)
            info["released"] = bool(not info["grasped"] and kw["grip_open"][0])
        terminated = bool(failure[0])
        # An end after a success streak is a truncation, like the time limit (see `task.success_streak`).
        truncated = self.t >= self.task.episode_steps or info["finished"]
        self._perceive(s)
        return self._obs(s), float(r[0]), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.render_camera is None:
            return None
        return self.scene.render(self.render_camera)

    def close(self):
        self.scene.close()
