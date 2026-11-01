"""MuJoCo Warp environments with JAX observations on `env.device`.

    env = BatchEnv("reach", num_envs=4096)
    obs = env.reset()                         # (N, obs_dim)
    obs, rew, done, info = env.step(actions)   # actions (N, act_dim) in [-1, 1]

Requires the `soarm-lab[train]` extra. `device="cpu"` runs physics on Warp's CPU device.
The scene and task formulas are shared with `env.ArmEnv`.

Worlds reset inside `step`. A true `done` marks the episode that just ended.
The returned observation belongs to the new episode. `info["time_outs"]` marks time limits
and success streaks for value bootstrapping. `info["finished"]` marks success streaks.

Physics arrays are copied into JAX so later physics steps cannot overwrite them.
Writes copy whole arrays back into Warp. Each write waits for JAX readers and writers.
Each physics launch finishes before its results are read.

`contact_budget` returns per-world limits for contacts (`nconmax`) and constraint rows (`njmax`).
For tasks with an object, `CONTACTS` is multiplied by its number of collision parts.
Warp silently drops rows beyond the budget. Check peak usage when adding collision geometry.

Perception and gripper speed limits follow the same rules as `env.ArmEnv`.

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if JAX's memory reservation leaves too little for Warp.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .. import sim as sc
from ..station import Station
from . import tasks as tk
from .env import TARGET_SIZE, object_geoms, scene_config
from .jax_policy import devices
from .randomize import Randomization
from .tasks.base import sample_start_jax

#: Contact and constraint-row limits per object part in each world.
CONTACTS = {"so101": (48, 288), "a1x": (64, 384)}
#: Randomized model fields are stored separately for each world.
FIELDS = ("actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature", "dof_frictionloss",
          "body_mass", "body_inertia", "geom_friction")
#: `set_const` recomputes these fields separately for each world.
DERIVED = ("body_invweight0", "dof_invweight0")
#: Object fields are omitted from randomization when the task has no object.
CUBE_FIELDS = ("body_mass", "body_inertia", "geom_friction")
#: Physics arrays needed for rewards and observations.
PHYSICS = ("qpos", "qvel", "site_xpos", "xpos", "xmat", "sensordata")
#: Reach goals are sampled from a pool built at initialization.
GOAL_POOL = 20000

Array = jax.Array
State = dict[str, Any]


def _warp():
    import warp as wp

    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    import mujoco_warp as mjw

    return wp, mjw


def _sel(done: Array, new: Array, old: Array) -> Array:
    """Select `new` where `done` (N,) is true along the first axis."""
    return jnp.where(done.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)


def _sel_hist(done: Array, new: Array, old: Array) -> Array:
    """Fill all history slots (H, N, ...) with `new` (N, ...) in done worlds."""
    return jnp.where(done.reshape((1, -1) + (1,) * (old.ndim - 2)), new[None], old)


def contact_budget(task: tk.TaskConfig, robot: str) -> tuple[int, int]:
    """Return per-world contact and constraint-row limits for the task and robot."""
    if task.cube is None:
        return 16, 64
    ncon, nefc = CONTACTS.get(robot, CONTACTS["so101"])
    parts = len(task.block_spec.parts)
    return ncon * parts, nefc * parts


class BatchEnv:
    def __init__(self, task: str | tk.TaskConfig = "reach", num_envs: int = 4096, *,
                 station: Station | None = None, randomization: Randomization | None = None,
                 device: str = "cuda:0", seed: int = 0):
        wp, mjw = _warp()
        self.wp, self.mjw = wp, mjw
        station = station or Station.load()
        spec = station.spec()
        self.task = task if isinstance(task, tk.TaskConfig) else tk.make_task(task, spec=spec)
        self.num_envs = N = int(num_envs)
        #: JAX outputs and Warp physics use corresponding devices.
        self.device, self.wp_device = devices(device)
        self.rand = randomization if randomization is not None else Randomization()
        # Dynamics use the same host sampler as ArmEnv. Other randomness uses the JAX state key.
        self.np_rng = np.random.default_rng(seed)

        scene = sc.build(scene_config(self.task, station))
        self.scene = scene
        mjm = scene.model
        mjm.opt.timestep = self.task.timestep
        self.kin = scene.kin
        self.limits = tk.Limits.of(self.kin)
        mujoco.mj_forward(mjm, scene.data)

        cube = self.task.cube is not None
        nconmax, njmax = contact_budget(self.task, spec.name)
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(mjm, batch_sizes={k: N for k in (*FIELDS, *DERIVED)})
            if hasattr(self.m.opt, "warn_overflow"):
                # Suppress repeated warnings when the solver reaches its iteration limit.
                self.m.opt.warn_overflow = 0
            self.d = mjw.put_data(mjm, scene.data, nworld=N, nconmax=nconmax, njmax=njmax)

        # Object fields and sensors are needed only for tasks with an object.
        self._fields = tuple(k for k in FIELDS if cube or k not in CUBE_FIELDS)
        self._reads = tuple(k for k in PHYSICS if cube or k != "sensordata")
        f32 = np.float32
        self.qadr = np.asarray(self.kin.qadr)
        self.dadr = np.asarray(self.kin.dadr)
        self.act = np.asarray(scene.act_ids)
        self.site = int(self.kin.site_id)
        T = scene.T_base2world
        self.R_b = T[:3, :3].astype(f32)
        self.t_b = T[:3, 3].astype(f32)
        self.qpos0 = mjm.qpos0.astype(f32)
        self.lo, self.hi = self.limits.lo.astype(f32), self.limits.hi.astype(f32)
        self.home = self.limits.home.astype(f32)
        self.goal_pool = None
        if cube:
            b = mjm.body(self.task.cube).id
            self.cube_body, self.cube_geoms = b, object_geoms(mjm, self.task.cube)
            self.yaw_period = float(self.task.block_spec.yaw_period)
            self.cube_qadr = int(mjm.jnt_qposadr[mjm.body_jntadr[b]])
            self.cube_dadr = int(mjm.jnt_dofadr[mjm.body_jntadr[b]])
            self.jaw_adr = np.array([mjm.sensor(f"{self.task.cube}_jaw{k}").adr[0] for k in range(2)])
            qb = np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            self.q_base = qb.astype(f32)
            if self.task.target:
                self.target_mocap = int(mjm.body_mocapid[mjm.body(self.task.target).id])
        else:
            pool = tk.sample_goals(self.kin, self.task, self.np_rng, GOAL_POOL)
            self.goal_pool = jax.device_put(pool.astype(f32), self.device)

        nj = self.task.act_dim
        # The action history covers the largest sampled delay in policy ticks.
        self.Hd = int(self.rand.max_delay) + 1
        self.noise = float(np.radians(self.rand.obs_noise_deg))
        # Object poses have a separate delay and refresh period in each world.
        self.H = int(self.rand.cube_delay) + 1
        fields = {k: self._read_one(getattr(self.m, k)) for k in self._fields}
        #: Randomization scales nominal fields to avoid accumulating multipliers across resets.
        self.nominal = {k: np.asarray(v[0]) for k, v in fields.items()}
        eye = np.broadcast_to(np.eye(3, dtype=f32), (N, 3, 3))
        state: State = {
            "key": jax.random.key(seed),
            "goal": np.zeros((N, 3), f32),
            # The place target uses the base frame with the table top at z = 0.
            "target": np.zeros((N, 3), f32),
            # Records whether the object has been carried during this episode.
            "carried": np.zeros(N, bool),
            "q_cmd": np.broadcast_to(self.home, (N, nj)).copy(),
            "prev_action": np.zeros((N, nj), f32),
            "act_hist": np.zeros((self.Hd, N, nj), f32),
            "delay_n": np.zeros(N, np.int32),
            "t": np.zeros(N, np.int32),
            "streak": np.zeros(N, np.int32),                     # consecutive success ticks
            "hist_pos": np.zeros((self.H, N, 3), f32),
            "hist_rot": np.broadcast_to(eye, (self.H, N, 3, 3)).copy(),
            "hist_i": np.int32(0),
            "cube_d": np.zeros(N, np.int32),
            "cube_p": np.ones(N, np.int32),
            "seen_pos": np.zeros((N, 3), f32),
            "seen_rot": eye.copy(),
            # Constant xy offset of the observed pose for each world [m].
            "bias": np.zeros((N, 2), f32),
            # Maximum gripper actuator step for each world [rad per policy tick].
            "grip_rate": np.full(N, np.inf, f32),
            # JAX owns fields that physics does not modify.
            "ctrl": self._read_one(self.d.ctrl),
            "fields": fields,
        }
        if self.task.target:
            # Mocap body positions have shape (N, nmocap, 3).
            state["mocap_pos"] = self._read_one(self.d.mocap_pos)
        #: Jitted steps replace this JAX state on `device`.
        self.s: State = jax.device_put(state, self.device)
        self._phys: dict[str, Array] = {}

        self._j_act = jax.jit(self._act)
        self._j_after = jax.jit(self._after)
        self._j_observe = jax.jit(self._observe)
        self._j_restart = jax.jit(self._restart)

        with wp.ScopedDevice(self.wp_device):
            mjw.step(self.m, self.d)                      # Compile and cache physics kernels.
            wp.synchronize_device()
            self.graph = None
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as cap:
                    self._steps()
                self.graph = cap.graph

    # ------------------------------------------------------------ warp <-> jax
    def _read_one(self, a) -> Array:
        # Copy the DLPack view before Warp reuses its buffer. CUDA requires a separate JAX copy.
        return jnp.copy(jax.dlpack.from_dlpack(a))

    def _read(self, names) -> dict[str, Array]:
        """Copy named physics arrays into JAX with vector dimensions last, e.g. (N, nsite, 3)."""
        return {k: self._read_one(getattr(self.d, k)) for k in names}

    def _write(self, pairs: list[tuple[Any, Array]], wait: Any = ()) -> None:
        """Copy `(dst, src)` pairs from JAX into Warp with matching shapes and vector dimensions last.

        Wait for JAX readers and sources before copying. `wait` covers additional readers of the
        destination buffers. The caller must keep the sources alive until Warp finishes.
        """
        jax.block_until_ready((self._phys, wait, [src for _, src in pairs]))
        wp = self.wp
        for dst, src in pairs:
            wp.copy(dst, wp.from_dlpack(src, dtype=dst.dtype))

    def _steps(self) -> None:
        for _ in range(self.task.substeps):
            self.mjw.step(self.m, self.d)
        # Refresh poses after integration so they match the observed joint angles.
        self.mjw.kinematics(self.m, self.d)

    def _physics(self, ctrl: Array) -> None:
        """Advance physics by one policy tick using `ctrl` (N, nu) and read the results."""
        wp = self.wp
        with wp.ScopedDevice(self.wp_device):
            pairs = [(self.d.ctrl, ctrl)]
            self._write(pairs)
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._steps()
            wp.synchronize_device()
        self._phys = self._read(self._reads)

    def _reset_worlds(self, done: Array) -> None:
        """Start new episodes where `done` (N,) is true."""
        smp = {k: np.asarray(v, np.int32 if k in ("delay", "cube_delay", "cube_period") else np.float32)
               for k, v in self.rand.sample(self.np_rng, self.num_envs).items()}
        old = self._read(("qpos", "qvel", "qacc_warmstart"))
        self.s, qpos, qvel, qacc = self._j_restart(self.s, old, done, jax.device_put(smp, self.device))
        wp, d, st = self.wp, self.d, self.s
        with wp.ScopedDevice(self.wp_device):
            fields = [(getattr(self.m, k), st["fields"][k]) for k in self._fields]
            self._write(fields)
            # Contact calculations need constants recomputed after mass and armature changes.
            self.mjw.set_const(self.m, d)
            state = [(d.qpos, qpos), (d.qvel, qvel), (d.qacc_warmstart, qacc), (d.ctrl, st["ctrl"])]
            if self.task.target:
                state.append((d.mocap_pos, st["mocap_pos"]))
            self._write(state, wait=old)
            self.mjw.kinematics(self.m, d)
            wp.synchronize_device()
        self._phys = self._read(self._reads)

    # ------------------------------------------------------------ pure functions (jit)
    def _state(self, phys: dict[str, Array], target: Array) -> dict[str, Array]:
        R, t = self.R_b, self.t_b
        out = {"q": phys["qpos"][:, self.qadr], "tcp": (phys["site_xpos"][:, self.site] - t) @ R}
        if self.task.cube:
            out["cube_pos"] = (phys["xpos"][:, self.cube_body] - t) @ R
            out["cube_rot"] = R.T @ phys["xmat"][:, self.cube_body]
            out["jaws"] = phys["sensordata"][:, self.jaw_adr]
            if self.task.target:
                a = self.cube_dadr
                out["cube_vel"] = phys["qvel"][:, a:a + 3] @ R          # Base-frame velocity as row vectors.
                out["target"] = target
        return out

    def _act(self, s: State, action: Array) -> tuple[State, Array, Array]:
        """Return updated state, delayed control (N, nu), and the clipped input action."""
        action = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)
        Hd, ar = self.Hd, jnp.arange(self.num_envs)
        # Use zero actions until the sampled delay has elapsed.
        t, delay = s["t"], s["delay_n"]
        hist = s["act_hist"].at[t % Hd, ar].set(action)
        executed = hist[(t - delay) % Hd, ar]
        executed = jnp.where((t >= delay)[:, None], executed, 0.0)
        q_cmd = tk.apply_action(jnp, self.task, self.limits, s["q_cmd"], executed)
        # The servo speed limit applies independently of the command rate.
        g = self.limits.grip
        grip_act = self.act[g]
        grip = tk.servo_follow(jnp, s["ctrl"][:, grip_act], q_cmd[:, g], s["grip_rate"])
        ctrl = s["ctrl"].at[:, self.act].set(q_cmd).at[:, grip_act].set(grip)
        return {**s, "act_hist": hist, "q_cmd": q_cmd, "ctrl": ctrl}, ctrl, action

    def _perceive(self, s: State, st: dict[str, Array], key: Array) -> State:
        """Refresh the delayed pose every `cube_p` ticks while the object is outside the occlusion radius."""
        H = self.H
        i = (s["hist_i"] + 1) % H
        hist_pos = s["hist_pos"].at[i].set(st["cube_pos"])
        hist_rot = s["hist_rot"].at[i].set(st["cube_rot"])
        upd = ((s["t"] % s["cube_p"]) == 0) & ~tk.occluded(jnp, st["tcp"], st["cube_pos"], self.rand.latch_m)
        idx = (i - s["cube_d"]) % H
        ar = jnp.arange(self.num_envs)
        pos, rot = hist_pos[idx, ar], hist_rot[idx, ar]
        n, shape = self.rand.cube_noise, (self.num_envs, 2)
        noise = n * jax.random.normal(key, shape) if n > 0 else jnp.zeros(shape, pos.dtype)
        pos, rot = tk.perceived(jnp, pos, rot, s["bias"], noise, self.rand.fold_yaw, self.yaw_period)
        return {**s, "hist_i": i, "hist_pos": hist_pos, "hist_rot": hist_rot,
                "seen_pos": jnp.where(upd[:, None], pos, s["seen_pos"]),
                "seen_rot": jnp.where(upd[:, None, None], rot, s["seen_rot"])}

    def _after(self, s: State, phys: dict[str, Array], action: Array):
        """Return updated state, reward, done flags, and info for the completed tick."""
        tsk = self.task
        key, k_cube = jax.random.split(s["key"])
        t = s["t"] + 1
        st = self._state(phys, s["target"])
        carried, kw = s["carried"], {}
        if tsk.target:
            carried = tk.carry_latch(jnp, tsk, carried, st["jaws"], st["cube_pos"], st["cube_rot"])
            kw = dict(target=s["target"], cube_vel=st["cube_vel"], carried=carried,
                      grip_open=tk.grip_is_open(jnp, self.limits, s["q_cmd"]))
        rew, success, failure = tsk.reward(jnp, st["tcp"], action, s["prev_action"], goal=s["goal"],
                                          cube_pos=st.get("cube_pos"), cube_rot=st.get("cube_rot"),
                                          jaw_contacts=st.get("jaws"),
                                          q_cmd=s["q_cmd"], limits=self.limits, **kw)
        streak, finished = tk.success_streak(jnp, tsk, s["streak"], success)
        s = {**s, "key": key, "t": t, "streak": streak, "carried": carried}
        if tsk.cube:
            s = self._perceive(s, st, k_cube)
        # End unstable episodes with zero reward before their values enter training.
        qvel = phys["qvel"]
        broken = ~jnp.isfinite(qvel).all(1) | (jnp.abs(qvel).max(1) > 100.0)
        rew = jnp.where(broken, 0.0, rew)
        s["prev_action"] = action
        # Bootstrap success truncations so ending an episode does not reduce its value.
        time_out = (t >= tsk.episode_steps) | finished
        terminated = failure | broken
        done = terminated | time_out
        info = {"success": success, "time_outs": time_out & ~terminated, "broken": broken,
                "finished": finished & ~terminated}
        if not tsk.cube:
            info["distance"] = jnp.sqrt(((s["goal"] - st["tcp"]) ** 2).sum(1))
        else:
            info["height"] = tsk.height(jnp, st["cube_pos"], st["cube_rot"])
        if tsk.target:
            info["dist_target"] = jnp.sqrt(((st["cube_pos"][:, :2] - s["target"][:, :2]) ** 2).sum(1))
            info["carried"] = carried
        return s, rew, done, info

    def _observe(self, s: State, phys: dict[str, Array]) -> tuple[State, Array]:
        key, k_noise = jax.random.split(s["key"])
        st = self._state(phys, s["target"])
        q = st["q"]
        if self.noise > 0:
            q = q + self.noise * jax.random.normal(k_noise, q.shape)
        cube = self.task.cube is not None
        obs = self.task.observe(jnp, self.limits, q, s["q_cmd"], st["tcp"], s["prev_action"],
                         goal=s["goal"], cube_pos=s["seen_pos"] if cube else None,
                         cube_rot=s["seen_rot"] if cube else None, target=s["target"])
        return {**s, "key": key}, obs

    def _randomize(self, s: State, done: Array, smp: dict[str, Array]) -> State:
        """Apply sampled multipliers to nominal dynamics in done worlds."""
        f, nom, act, dof = dict(s["fields"]), self.nominal, self.act, self.dadr

        def put(name, idx, value):
            f[name] = _sel(done, f[name].at[idx].set(value), f[name])

        kp = smp["kp"][:, None]
        put("actuator_gainprm", (slice(None), act, 0), nom["actuator_gainprm"][act, 0] * kp)
        put("actuator_biasprm", (slice(None), act, 1), nom["actuator_biasprm"][act, 1] * kp)
        put("dof_damping", (slice(None), dof), nom["dof_damping"][dof] * smp["damping"][:, None])
        put("dof_armature", (slice(None), dof), nom["dof_armature"][dof] * smp["armature"][:, None])
        put("dof_frictionloss", (slice(None), dof), nom["dof_frictionloss"][dof] * smp["frictionloss"][:, None])
        if self.task.cube:
            b, g, mass = self.cube_body, self.cube_geoms, smp["cube_mass"]
            put("body_mass", (slice(None), b), nom["body_mass"][b] * mass)
            put("body_inertia", (slice(None), b), nom["body_inertia"][b] * mass[:, None])
            put("geom_friction", (slice(None), g, 0), nom["geom_friction"][g, 0] * smp["cube_friction"][:, None])
        return {**s, "fields": f,
                "delay_n": _sel(done, jnp.clip(smp["delay"], 0, self.Hd - 1), s["delay_n"]),
                "cube_d": _sel(done, smp["cube_delay"], s["cube_d"]),
                "cube_p": _sel(done, smp["cube_period"], s["cube_p"]),
                "bias": _sel(done, jnp.stack([smp["pose_bias_x"], smp["pose_bias_y"]], 1), s["bias"]),
                "grip_rate": _sel(done, tk.grip_rate(jnp, self.task, self.limits, smp["grip_close_s"]),
                                  s["grip_rate"])}

    def _restart(self, s: State, old: dict[str, Array], done: Array, smp: dict[str, Array]):
        """Return new state, qpos, qvel, and qacc_warmstart with done worlds reset."""
        s = self._randomize(s, done, smp)
        tsk, N = self.task, self.num_envs
        key, k_q, k_g, k_r, k_b, k_yaw, k_t = jax.random.split(s["key"], 7)
        q0 = sample_start_jax(jax, tsk, self.limits, self.home, self.lo, self.hi, N, k_q, k_g)
        # Reset keys are ordered as gripper/goal, radius, bearing, yaw, and target.
        sampled = tsk.sample_reset(jax, N, (k_g, k_r, k_b, k_yaw, k_t), goal_pool=self.goal_pool)

        qpos = jnp.broadcast_to(self.qpos0, (N, len(self.qpos0))).at[:, self.qadr].set(q0)
        for k, adr, gain in self.kin.mirror:            # Coupled joints follow the commanded joint.
            qpos = qpos.at[:, adr].set(gain * q0[:, k])
        if tsk.cube:
            p, yaw = sampled["cube_pos"], sampled["cube_yaw"]
            qb = self.q_base
            # Compose base-frame yaw into world coordinates as q_base * q_yaw.
            cw, sw = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
            quat = jnp.stack([qb[0] * cw - qb[3] * sw, qb[1] * cw + qb[2] * sw,
                              qb[2] * cw - qb[1] * sw, qb[0] * sw + qb[3] * cw], 1)
            a = self.cube_qadr
            qpos = qpos.at[:, a:a + 3].set(p @ self.R_b.T + self.t_b)
            qpos = qpos.at[:, a + 3:a + 7].set(quat)
            # Fill perception history with the initial pose.
            z0, o1 = jnp.zeros_like(yaw), jnp.ones_like(yaw)
            R = jnp.stack([jnp.stack([jnp.cos(yaw), -jnp.sin(yaw), z0], -1),
                           jnp.stack([jnp.sin(yaw), jnp.cos(yaw), z0], -1),
                           jnp.stack([z0, z0, o1], -1)], -2)
            s["hist_pos"] = _sel_hist(done, p, s["hist_pos"])
            s["hist_rot"] = _sel_hist(done, R, s["hist_rot"])
            # Reset observations include the pose bias without noise.
            p_seen, _ = tk.perceived(jnp, p, R, s["bias"], jnp.zeros_like(s["bias"]), False, self.yaw_period)
            s["seen_pos"] = _sel(done, p_seen, s["seen_pos"])
            s["seen_rot"] = _sel(done, tk.fold_yaw(jnp, R, self.yaw_period) if self.rand.fold_yaw else R,
                                 s["seen_rot"])
            if tsk.target:
                # The pad centre sits half its thickness above the target at base-frame z = 0.
                target = sampled["target"]
                pad = target.at[:, 2].set(TARGET_SIZE[2])
                mocap = s["mocap_pos"].at[:, self.target_mocap].set(pad @ self.R_b.T + self.t_b)
                s["target"] = _sel(done, target, s["target"])
                s["mocap_pos"] = _sel(done, mocap, s["mocap_pos"])
        else:
            s["goal"] = _sel(done, sampled["goal"], s["goal"])
        s.update(key=key,
                 ctrl=_sel(done, s["ctrl"].at[:, self.act].set(q0), s["ctrl"]),
                 q_cmd=_sel(done, q0, s["q_cmd"]),
                 prev_action=_sel(done, 0.0, s["prev_action"]),
                 t=_sel(done, 0, s["t"]),
                 streak=_sel(done, 0, s["streak"]),
                 carried=_sel(done, False, s["carried"]))
        qpos = _sel(done, qpos, old["qpos"])
        qvel = _sel(done, 0.0, old["qvel"])
        qacc = _sel(done, 0.0, old["qacc_warmstart"])
        return s, qpos, qvel, qacc

    # ------------------------------------------------------------------ api
    def reset(self) -> Array:
        """Start new episodes in every world and return observations (N, obs_dim)."""
        self._reset_worlds(jax.device_put(jnp.ones(self.num_envs, bool), self.device))
        self.s, obs = self._j_observe(self.s, self._phys)
        return obs

    def step(self, action: Array) -> tuple[Array, Array, Array, dict[str, Array]]:
        """Return (obs, rew, done, info) after one policy tick and reset done worlds."""
        action = jax.device_put(jnp.asarray(action), self.device)
        self.s, ctrl, action = self._j_act(self.s, action)
        self._physics(ctrl)
        self.s, rew, done, info = self._j_after(self.s, self._phys, action)
        if bool(done.any()):
            self._reset_worlds(done)
        self.s, obs = self._j_observe(self.s, self._phys)
        return obs, rew, done, info

    @property
    def obs_dim(self) -> int:
        return self.task.obs_dim

    @property
    def act_dim(self) -> int:
        return self.task.act_dim
