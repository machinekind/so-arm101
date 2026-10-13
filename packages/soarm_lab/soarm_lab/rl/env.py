"""Srodowisko Gymnasium na CPU: `SoArmLab/Reach-v0`, `SoArmLab/Lift-v0`, `SoArmLab/PickPlace-v0`.

    import gymnasium as gym, soarm_lab.rl
    env = gym.make("SoArmLab/Reach-v0")
    obs, info = env.reset(seed=0)

To jest wersja "jedno srodowisko, zwykly MuJoCo" - do sprawdzania polityk,
debugowania nagrody, `check_env` i bibliotek w stylu SB3. Do uczenia na
tysiacach swiatow naraz sluzy `batch.BatchEnv` na MuJoCo Warp; obie licza
obserwacje i nagrode tymi samymi funkcjami z `task`.

Scena jest budowana RAZ (293 ms na laptopie, 111 ms na RTX A4500); reset zmienia
tylko stan i pola skompilowanego modelu. Stanowisko (stol, kamery, poza podstawy)
przychodzi z eksportu blizniaka (`station.Station`, `lerobot-twin export`);
domyslnie stanowisko `sim` dostarczane z pakietem.
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

#: Kamera podgladu, gdy stanowisko nie ma zadnej skalibrowanej.
VIEW = "widok"
#: Podkladka celu pickplace: polowy bokow [m] i kolor (niebieski - kostka jest czerwona).
TARGET_SIZE = (0.035, 0.035, 0.002)
TARGET_RGBA = (0.2, 0.4, 0.9, 1.0)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


def scene_config(task: tk.TaskConfig, station: Station | None = None, view: bool = False) -> sc.SceneConfig:
    """Scena zadania: stanowisko (stol, kamery) plus obiekty zadania."""
    ws = station or Station.load()
    cfg = sc.SceneConfig(ws)
    if task.cube:
        h = task.cube_half
        cfg.objects = [*cfg.objects, sc.Box(task.cube, (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                            mass=task.cube_mass)]
        cfg.grasp_sensors = [*cfg.grasp_sensors, task.cube]
    if task.target:
        # Mocap bez kolizji: kostka lezy na blacie, wiec `height` znaczy to samo co w lift.
        cfg.objects = [*cfg.objects, sc.Box(task.target, TARGET_SIZE, (0.3, 0.1), rgba=TARGET_RGBA, free=False,
                                            mocap=True)]
    if view and not cfg.cameras:
        # Skala stolu ramienia (SO-101: 0,56 m, mnoznik dokladnie 1).
        f = ws.spec().desk.map_side / 0.56
        cfg.cameras = [sc.CameraView.from_fov(VIEW, 480, 360, 50.0, look_at(np.array([0.62, -0.42, 0.42]) * f,
                                                                           np.array([0.18, 0.0, 0.05]) * f))]
    return cfg


class ModelFields:
    """Nominalne pola modelu, ktore randomizacja mnozy - zapamietane raz."""

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
        """Mnozniki `s` na pola modelu, liczone zawsze od nominalu (nie kumuluja sie)."""
        m.actuator_gainprm[self.act, 0] = self.kp * s["kp"]
        m.actuator_biasprm[self.act, 1] = self.bias[:, 0] * s["kp"]
        m.dof_damping[self.dof] = self.damping * s["damping"]
        m.dof_armature[self.dof] = self.armature * s["armature"]
        m.dof_frictionloss[self.dof] = self.frictionloss * s["frictionloss"]
        if self.cube_body is not None:
            # Masa razem z bezwladnoscia - to ten sam obiekt, tylko ciezszy.
            m.body_mass[self.cube_body] = self.cube_mass * s["cube_mass"]
            m.body_inertia[self.cube_body] = self.cube_inertia * s["cube_mass"]
            m.geom_friction[self.cube_geom] = self.cube_friction * np.array([s["cube_friction"], 1.0, 1.0])
        # Masa i armatura wchodza w stale liczone przy kompilacji (body_invweight0,
        # dof_invweight0), a od nich zalezy miekkosc kontaktow. Bez przeliczenia
        # kostka o zmienionej masie miala kontakty "od innej kostki" i przy
        # zakleszczeniu miedzy szczeka a blatem wylatywala z predkoscia 10-180 m/s.
        mujoco.mj_setConst(m, d)


class TwinEnv(gym.Env):
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
        #: pickplace: cel odlozenia na blacie (z = 0), uklad podstawy.
        self.target = np.zeros(3)
        #: pickplace: zatrzask "kostka byla niesiona" (`task.carry_latch`), zerowany w `reset`.
        self._carried = False
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(self.task.act_dim)
        self.t = 0
        self._streak = 0                     # takty sukcesu z rzedu (`task.end_on_success`)
        self._delay: list[np.ndarray] = []
        self._noise = 0.0
        self._cube_delay, self._cube_period = 0, 1
        self._cube_hist: list[tuple[np.ndarray, np.ndarray]] = []
        self._cube_seen = (np.zeros(3), np.eye(3))

    # ------------------------------------------------------------ stan
    def state(self) -> dict[str, np.ndarray]:
        """Wszystko, z czego licza sie obserwacja i nagroda - w ukladzie podstawy."""
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
        """Kostka w obserwacji tak, jak ja widza kamery: z opoznieniem, rzadziej, z szumem.

        Nagroda liczy sie z prawdy; polityka widzi to, co da jej `perception` na biurku.
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
        """Kostka w pozie podanej w ukladzie podstawy (kwaternion w, x, y, z)."""
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
        """Cel pickplace w (x, y) ukladu podstawy; z zawsze 0 (blat). Podkladka mocap idzie za nim."""
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
        # Po mj_step qpos jest juz po calkowaniu, a site_xpos/xpos/xmat - z przejscia w przod
        # PRZED nim (5 ms starsze). TCP w obserwacji rozjezdzal sie z FK(q) o 4-8 mm w ruchu,
        # a runner na ramieniu liczy TCP dokladnie z q - polityka uczyla sie relacji,
        # ktorej na biurku nie ma. Kinematyka od nowa: wszystko z tego samego q.
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
        # Koniec po serii sukcesow to uciecie jak limit czasu (patrz `task.success_streak`).
        truncated = self.t >= self.task.episode_steps or info["finished"]
        self._perceive(s)
        return self._obs(s), float(r[0]), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.render_camera is None:
            return None
        return self.scene.render(self.render_camera)

    def close(self):
        self.scene.close()
