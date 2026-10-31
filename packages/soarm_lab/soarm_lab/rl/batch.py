"""The GPU batch environment: thousands of MuJoCo Warp worlds, observations as JAX arrays.

    env = BatchEnv("reach", num_envs=4096)
    obs = env.reset()                                  # (N, obs_dim) on the GPU
    obs, rew, done, info = env.step(actions)           # actions (N, act_dim) in [-1, 1]

The same scene and the same formulas (`task`) as `env.ArmEnv` on the CPU; the only difference is
where the physics runs. On an RTX A4500 the cube scene did about 1.1 million physics steps per
second with 8192 worlds (measured with the torch version of this module; the physics is the same).
One world stepped with zero actions follows CPU MuJoCo within 0.00001 deg per joint over 40 ticks
(2 s) on Warp's CPU device (float32 against float64); the torch version measured 0.001 deg after 2 s
on an RTX A4500. `tests/rl/test_batch.py` holds both devices to 0.01 deg.

The convention follows legged_gym / rsl_rl: environments reset themselves, `done` marks the end of an
episode in this step, and the returned observation already belongs to the new episode.
`info["time_outs"]` says which worlds ended on the time limit or on a success streak
(`task.end_on_success`, then `info["finished"]` too): PPO bootstraps them with the state value
instead of zero.

This is the environment of the JAX trainer (`ppo`): it needs the `soarm-lab[train]` extra (jax,
mujoco-warp), no torch. Everything it returns is a plain JAX array on `env.device`.

JAX and Warp meet through DLPack. The physics state lives in Warp arrays (`env.d`, `env.m`). It is
read into JAX as COPIES (a zero-copy `jax.dlpack.from_dlpack` view, copied at once): an aliasing view would be
overwritten by the next physics launch, and JAX's jit may hand an input straight back as an output.
That is N x a few hundred floats per step, nothing next to the physics. Writes go the other way as
WHOLE arrays: the complete new `qpos` (`jnp.where(done, new, old)`), `qvel`, `ctrl` and so on, wrapped
by `wp.from_dlpack` and copied with `wp.copy`, so there are no scatter kernels. The model fields of
the randomization are written the same way. What physics never changes (`ctrl`, `mocap_pos`, the
model fields) is kept in JAX in the env's state and never read back.

Synchronization is explicit, as it was with torch: the JAX arrays a launch reads or overwrites are
waited for (`jax.block_until_ready`) before the copies into Warp, and Warp's device is synchronized
after the physics (two synchronizations per policy step, negligible next to tens of milliseconds of
physics). `wp.from_dlpack` passes JAX the Warp stream to wait on, so DLPack alone should order the
copies on the GPU; the explicit wait does not depend on that, and it holds on the CPU device too,
where JAX's dispatch is asynchronous as well.

The physics is one captured CUDA graph of `substeps` steps plus the kinematics. On `device="cpu"`
the same steps run as a plain loop on Warp's CPU device, without a graph: mujoco_warp runs there
(about 3 ms per physics step for 4 worlds on an M-series Mac; the first run compiles the kernels,
~20 s, then Warp caches them). It is for tests and debugging, not for training.

Memory: JAX preallocates 75 % of the GPU's memory when its backend starts, and Warp's worlds live
in the rest. If `put_data` runs out of memory, start with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The per-step computation is three jitted pure functions over the env's state (a dict of JAX
arrays, with the random key): `_act` (the action delay ring buffer, `task.apply_action`, the
control), `_after` (the state from the physics arrays, reward, success streak, cube perception,
the broken-world guard, done) and `_observe`; the reset of the done worlds is a fourth one
(`_restart`), run only in steps where some world ended. The formulas are the ones in `task`,
called with `xp=jax.numpy`.

The object's pose in the observation is the cameras' (`_perceive`, the same `task.perceived` and
`task.occluded` as `env.ArmEnv`): late by `cube_d` ticks from a history ring, refreshed every `cube_p`
ticks, offset by the world's constant `bias` for the episode, with noise. The occlusion latch is
computed from the truth and is not sticky: a world whose true TCP is within `Randomization.latch_m` of
the object skips its refreshes and keeps the last seen pose, and refreshes again once the TCP is
farther. A block carried in the jaws is always that near, so in training a carry is one long freeze.

pickplace: the place target is a mocap pad (`d.mocap_pos`, separate in every world) without
collisions, so the contact budget is the same as in lift. The "cube carried" latch (`carried` in the
state) is the same `task.carry_latch` that `env.ArmEnv` uses, and target sampling (`_sample_targets`)
follows the same rule as `task.sample_targets`.

The contact budget per world (`CONTACTS`: nconmax, njmax) for the tasks with a cube. Measured in
CPU MuJoCo (the most in one physics step, over random episodes and a scripted pickplace expert):
SO-101 24 contacts / 142 constraint rows (expert, grasp), A1X 12 / 49. MuJoCo Warp generates mesh
contacts (the convex hulls of the A1X links) differently from the CPU, so the A1X gets a margin of
64 / 384, still to be confirmed on a GPU (`int(env.d.ncon.numpy().max())` after 200 steps).

The wooden blocks (`objects`, `TaskConfig.object_kind`) have a budget of their own per collision part
(`BLOCK_CONTACTS`), times the block's number of parts. Measured on Warp's CPU device, SO-101, 8 worlds,
the most in one world in one physics step over 200 random ticks and then 400 ticks of lift-v3 grasping
the block (it lifts all four zero-shot): cube 33 contacts / 204 constraint rows, cuboid5 32 / 194,
cuboid85 26 / 160, cylinder 25 / 152; random actions alone reach 10 / 58. The legacy cube measured
32 / 194 the same way, two rows over its `CONTACTS`: Warp drops the rows past `njmax`, and with
`warn_overflow` off it says nothing. The blocks get 48 / 288 per part, about 1.4 times the most
measured.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .. import sim as sc
from ..station import Station
from . import task as tk
from .env import TARGET_SIZE, object_geoms, scene_config
from .jax_policy import devices
from .randomize import Randomization

#: (nconmax, njmax) per world for the legacy cube; without a cube (reach) 16 / 64. The SO-101 budget
#: was 32 / 192 until 2026-10-29, when lift-v3 grasping the cube on Warp's CPU device measured 33
#: contacts / 204 rows: Warp drops the rows past njmax and says nothing (`warn_overflow` is off),
#: so grasps were computed short. Now the block budget below, about 1.4 x the measured peak.
#: Warp warns "CCD pairs without multicontact support: (CAPSULE, CYLINDER). At most 1 contact": the
#: fixed jaw's capsule meets a cylinder at one point on Warp, several on the CPU, a known gap between
#: training and the CPU evaluation of the cylinder.
CONTACTS = {"so101": (48, 288), "a1x": (64, 384)}
#: (nconmax, njmax) per world and per collision part of a wooden block (`objects`), times the block's
#: number of parts. Measured for the SO-101 (module docstring); the A1X keeps its cube's margin.
BLOCK_CONTACTS = {"so101": (48, 288), "a1x": (64, 384)}
#: The model fields randomization changes, separate in every world (`put_model(batch_sizes=)`).
FIELDS = ("actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature", "dof_frictionloss",
          "body_mass", "body_inertia", "geom_friction")
#: The fields `set_const` recomputes from them: batched too, never written here.
DERIVED = ("body_invweight0", "dof_invweight0")
#: The fields of the cube; reach has no cube and leaves them alone.
CUBE_FIELDS = ("body_mass", "body_inertia", "geom_friction")
#: What `_after` and `_observe` read from the physics after a launch.
PHYSICS = ("qpos", "qvel", "site_xpos", "xpos", "xmat", "sensordata")
#: The number of reach goals drawn once at construction (`task.sample_goals`), then picked at random.
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
    """`new` in the worlds that are `done` (N,), `old` elsewhere; the world is the first axis."""
    return jnp.where(done.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)


def _sel_hist(done: Array, new: Array, old: Array) -> Array:
    """`_sel` for a history (H, N, ...): the world is the second axis, `new` (N, ...) fills every slot."""
    return jnp.where(done.reshape((1, -1) + (1,) * (old.ndim - 2)), new[None], old)


def contact_budget(task: tk.TaskConfig, robot: str) -> tuple[int, int]:
    """(nconmax, njmax) per world for `task` on the arm `robot` (see the module docstring)."""
    if task.cube is None:
        return 16, 64
    if tk.legacy_cube(task):
        return CONTACTS.get(robot, CONTACTS["so101"])
    ncon, nefc = BLOCK_CONTACTS.get(robot, BLOCK_CONTACTS["so101"])
    parts = len(tk.task_object(task).parts)
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
        #: The JAX device of everything the env returns, and the Warp device of the physics.
        self.device, self.wp_device = devices(device)
        self.rand = randomization if randomization is not None else Randomization()
        # The dynamics are drawn on the host with `Randomization.sample`, the function ArmEnv draws
        # them with, for all N worlds in a step where some world resets (only those take them):
        # one definition of the distribution, ~9 x N numbers per reset step. Everything else
        # (start poses, cubes, targets, goals, noise) comes from the JAX key in the state.
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
                # In MuJoCo a solver iteration overflow usually means "stop here", not an error;
                # the CPU does not complain about it. Here it would print every step.
                self.m.opt.warn_overflow = 0
            self.d = mjw.put_data(mjm, scene.data, nworld=N, nconmax=nconmax, njmax=njmax)

        # The fields this env writes: the arm's always, the cube's in the tasks with a cube.
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
        if cube:
            b = mjm.body(self.task.cube).id
            #: The object's body, and every geom of it: a block of several parts has one geom per part,
            #: and randomization scales the friction of all of them.
            self.cube_body, self.cube_geoms = b, object_geoms(mjm, self.task.cube)
            #: The yaw symmetry the detector cannot see (`task.yaw_period`): the perception's fold.
            self.yaw_period = float(tk.yaw_period(self.task))
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
        # Action delay: the latest actions in a ring buffer, each world with its own delay of
        # {min_delay .. max_delay} ticks, like the queue in ArmEnv. A single slot would cap the
        # delay at 0/1 tick: after an identification of 1.4 ticks (max_delay 3) training would
        # never see a delay of 2-3 ticks, while the CPU evaluation does.
        self.Hd = int(self.rand.max_delay) + 1
        self.noise = float(np.radians(self.rand.obs_noise_deg))
        # Cube perception (see Randomization.cube_*): the history in a ring buffer, the delay and
        # the refresh period separate for every world.
        self.H = int(self.rand.cube_delay) + 1
        fields = {k: self._read_one(getattr(self.m, k)) for k in self._fields}
        #: The nominal model fields (every world has them after `put_model`): randomization multiplies
        #: these, never the current values, so the multipliers never pile up.
        self.nominal = {k: np.asarray(v[0]) for k, v in fields.items()}
        eye = np.broadcast_to(np.eye(3, dtype=f32), (N, 3, 3))
        state: State = {
            "key": jax.random.key(seed),
            "goal": np.zeros((N, 3), f32),
            # pickplace: the place target on the table top (z = 0), in the base frame.
            "target": np.zeros((N, 3), f32),
            # pickplace: the "cube carried" latch (`task.carry_latch`).
            "carried": np.zeros(N, bool),
            "q_cmd": np.broadcast_to(self.home, (N, nj)).copy(),
            "prev_action": np.zeros((N, nj), f32),
            "act_hist": np.zeros((self.Hd, N, nj), f32),
            "delay_n": np.zeros(N, np.int32),
            "t": np.zeros(N, np.int32),
            "streak": np.zeros(N, np.int32),                     # success ticks in a row
            "hist_pos": np.zeros((self.H, N, 3), f32),
            "hist_rot": np.broadcast_to(eye, (self.H, N, 3, 3)).copy(),
            "hist_i": np.int32(0),
            "cube_d": np.zeros(N, np.int32),
            "cube_p": np.ones(N, np.int32),
            "seen_pos": np.zeros((N, 3), f32),
            "seen_rot": eye.copy(),
            # This episode's constant xy offset of the seen object pose [m] (`Randomization.pose_bias_m`).
            "bias": np.zeros((N, 2), f32),
            # The JAX copies of what physics never changes: written whole into Warp, never read back.
            "ctrl": self._read_one(self.d.ctrl),
            "fields": fields,
        }
        if self.task.target:
            # pickplace: positions of the mocap bodies (N, nmocap, 3): the target pad.
            state["mocap_pos"] = self._read_one(self.d.mocap_pos)
        #: The env's state: JAX arrays on `device` and the random key, replaced by every jitted step.
        self.s: State = jax.device_put(state, self.device)
        self._phys: dict[str, Array] = {}

        self._j_act = jax.jit(self._act)
        self._j_after = jax.jit(self._after)
        self._j_observe = jax.jit(self._observe)
        self._j_restart = jax.jit(self._restart)

        with wp.ScopedDevice(self.wp_device):
            mjw.step(self.m, self.d)                      # compiles the kernels (once, then cached)
            wp.synchronize_device()
            self.graph = None
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as cap:
                    self._steps()
                self.graph = cap.graph

    # ------------------------------------------------------------ warp <-> jax
    def _read_one(self, a) -> Array:
        # A zero-copy view first, then a copy made by JAX: `from_dlpack(copy=True)` works on the
        # CPU backend but the CUDA PJRT plugin refuses it ("does not support HostBufferSemantics
        # other than ... kImmutableZeroCopy"; jax 0.11.2). The copy is what `_phys` keeps, and
        # `_write` waits for it before the next launch overwrites the buffer it was taken from.
        return jnp.copy(jax.dlpack.from_dlpack(a))

    def _read(self, names) -> dict[str, Array]:
        """Copies of the Warp arrays `names` of `d`, vector dimensions last (`site_xpos` (N, nsite, 3))."""
        return {k: self._read_one(getattr(self.d, k)) for k in names}

    def _write(self, pairs: list[tuple[Any, Array]], wait: Any = ()) -> None:
        """Whole JAX arrays into their Warp arrays (`(dst, src)`, the same shape with vector dims last).

        First the wait: for the JAX copies of the last read and for `wait` (the copies these writes
        overwrite the source of), and for the arrays written (jit dispatch is asynchronous). The
        caller synchronizes Warp after its launch, while `pairs` still holds the JAX arrays the
        copies read.
        """
        jax.block_until_ready((self._phys, wait, [src for _, src in pairs]))
        wp = self.wp
        for dst, src in pairs:
            wp.copy(dst, wp.from_dlpack(src, dtype=dst.dtype))

    def _steps(self) -> None:
        for _ in range(self.task.substeps):
            self.mjw.step(self.m, self.d)
        # As in ArmEnv.step: after the steps xpos/site_xpos are from before the last integration,
        # so the kinematics run again: the TCP and the cube from the same q as the observed angles.
        self.mjw.kinematics(self.m, self.d)

    def _physics(self, ctrl: Array) -> None:
        """One policy tick of physics with the control `ctrl` (N, nu), then a fresh read."""
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
        """New episodes in the worlds that are `done` (N,); the others keep their state."""
        smp = {k: np.asarray(v, np.int32 if k in ("delay", "cube_delay", "cube_period") else np.float32)
               for k, v in self.rand.sample(self.np_rng, self.num_envs).items()}
        old = self._read(("qpos", "qvel", "qacc_warmstart"))
        self.s, qpos, qvel, qacc = self._j_restart(self.s, old, done, jax.device_put(smp, self.device))
        wp, d, st = self.wp, self.d, self.s
        with wp.ScopedDevice(self.wp_device):
            fields = [(getattr(self.m, k), st["fields"][k]) for k in self._fields]
            self._write(fields)
            # Without it the contacts of a cube with a changed mass are computed "for another cube"
            # (body_invweight0); on the CPU that shoots the cube away (see env.ModelFields.apply).
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
            # The height of the object's lowest point above the table (`task.object_bottom`): the legacy
            # cube's centre minus `cube_half`, a block's from its parts and its rotation.
            out["bottom"] = tk.object_bottom(jnp, self.task, out["cube_pos"], out["cube_rot"])
            out["jaws"] = phys["sensordata"][:, self.jaw_adr]
            if self.task.target:
                a = self.cube_dadr
                out["cube_vel"] = phys["qvel"][:, a:a + 3] @ R          # R^T v as rows (as in ArmEnv)
                out["target"] = target
        return out

    def _act(self, s: State, action: Array) -> tuple[State, Array, Array]:
        """The executed action and the new joint targets: (state, control (N, nu), clipped action)."""
        action = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)
        Hd, ar = self.Hd, jnp.arange(self.num_envs)
        # The action from `delay_n` ticks ago; zero in the episode's first ticks (like the queue of
        # zeros on the CPU).
        t, delay = s["t"], s["delay_n"]
        hist = s["act_hist"].at[t % Hd, ar].set(action)
        executed = hist[(t - delay) % Hd, ar]
        executed = jnp.where((t >= delay)[:, None], executed, 0.0)
        q_cmd = tk.apply_action(jnp, self.task, self.limits, s["q_cmd"], executed)
        ctrl = s["ctrl"].at[:, self.act].set(q_cmd)
        return {**s, "act_hist": hist, "q_cmd": q_cmd, "ctrl": ctrl}, ctrl, action

    def _perceive(self, s: State, st: dict[str, Array], key: Array) -> State:
        """A new entry in the cube history, and a refresh of what the policy sees: every `cube_p` ticks,
        except while the true TCP is within `latch_m` of the cube (`task.occluded`)."""
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
        """Reward, done and info of the tick just simulated: (state, rew, done, info)."""
        tsk = self.task
        key, k_cube = jax.random.split(s["key"])
        t = s["t"] + 1
        st = self._state(phys, s["target"])
        carried, kw = s["carried"], {}
        if tsk.target:
            carried = tk.carry_latch(jnp, tsk, carried, st["jaws"], st["cube_pos"], bottom=st["bottom"])
            kw = dict(target=s["target"], cube_vel=st["cube_vel"], carried=carried,
                      grip_open=tk.grip_is_open(jnp, self.limits, s["q_cmd"]))
        rew, success, failure = tk.reward(jnp, tsk, st["tcp"], action, s["prev_action"], goal=s["goal"],
                                          cube_pos=st.get("cube_pos"), jaw_contacts=st.get("jaws"),
                                          q_cmd=s["q_cmd"], limits=self.limits, bottom=st.get("bottom"), **kw)
        streak, finished = tk.success_streak(jnp, tsk, s["streak"], success)
        s = {**s, "key": key, "t": t, "streak": streak, "carried": carried}
        if tsk.cube:
            s = self._perceive(s, st, k_cube)
        # Guard: a world that blew up anyway ends its episode without reward before its numbers
        # poison the gradient.
        qvel = phys["qvel"]
        broken = ~jnp.isfinite(qvel).all(1) | (jnp.abs(qvel).max(1) > 100.0)
        rew = jnp.where(broken, 0.0, rew)
        s["prev_action"] = action
        # A success streak ends the episode like the time limit: PPO bootstraps the state value
        # (see `task.success_streak`), otherwise not finishing would pay.
        time_out = (t >= tsk.episode_steps) | finished
        terminated = failure | broken
        done = terminated | time_out
        info = {"success": success, "time_outs": time_out & ~terminated, "broken": broken,
                "finished": finished & ~terminated}
        if tsk.name == "reach":
            info["distance"] = jnp.sqrt(((s["goal"] - st["tcp"]) ** 2).sum(1))
        else:
            info["height"] = st["bottom"]
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
        obs = tk.observe(jnp, self.task, self.limits, q, s["q_cmd"], st["tcp"], s["prev_action"],
                         goal=s["goal"], cube_pos=s["seen_pos"] if cube else None,
                         cube_rot=s["seen_rot"] if cube else None, target=s["target"])
        return {**s, "key": key}, obs

    def _sample_targets(self, key: Array, cube_p: Array) -> Array:
        """pickplace targets (N, 3) for the cubes `cube_p` (N, 3), as `task.sample_targets` draws them.

        The same rule: polar sampling, rows too close to their cube drawn again (only those), at most
        `TARGET_TRIES` times, then a target on the outer radius at the far end of the bearing range.
        A bounded loop (`lax.while_loop`), so it stays inside jit; it stops when no row is left.
        """
        tsk = self.task
        n = cube_p.shape[0]

        def bad_rows(out):
            return jnp.sqrt(((out[:, :2] - cube_p[:, :2]) ** 2).sum(1)) <= tsk.target_min_sep

        def draw(carry):
            i, out, bad = carry
            kr, kb = jax.random.split(jax.random.fold_in(key, i))
            r = jax.random.uniform(kr, (n,), minval=tsk.target_radius[0], maxval=tsk.target_radius[1])
            b = jax.random.uniform(kb, (n,), minval=tsk.target_bearing[0], maxval=tsk.target_bearing[1])
            new = jnp.stack([r * jnp.cos(b), r * jnp.sin(b), jnp.zeros_like(r)], 1)
            out = jnp.where(bad[:, None], new, out)
            return i + 1, out, bad_rows(out)

        def more(carry):
            i, _, bad = carry
            return (i < tk.TARGET_TRIES) & bad.any()

        init = (jnp.int32(0), jnp.zeros((n, 3), cube_p.dtype), jnp.ones(n, bool))
        _, out, bad = jax.lax.while_loop(more, draw, init)
        cb = jnp.arctan2(cube_p[:, 1], cube_p[:, 0])
        lo, hi = tsk.target_bearing
        far = jnp.where(cb > 0.5 * (lo + hi), lo, hi)
        r = tsk.target_radius[1]
        fallback = jnp.stack([r * jnp.cos(far), r * jnp.sin(far), jnp.zeros_like(far)], 1)
        return jnp.where(bad[:, None], fallback, out)

    def _randomize(self, s: State, done: Array, smp: dict[str, Array]) -> State:
        """New dynamics for the done worlds: multipliers of the nominal values (`set_const` follows)."""
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
            # The same multiplier on every part: it is one block of one material.
            friction = nom["geom_friction"][g, 0] * smp["cube_friction"][:, None]
            put("geom_friction", (slice(None), g, 0), friction)
        return {**s, "fields": f,
                "delay_n": _sel(done, jnp.clip(smp["delay"], 0, self.Hd - 1), s["delay_n"]),
                "cube_d": _sel(done, smp["cube_delay"], s["cube_d"]),
                "cube_p": _sel(done, smp["cube_period"], s["cube_p"]),
                "bias": _sel(done, jnp.stack([smp["pose_bias_x"], smp["pose_bias_y"]], 1), s["bias"])}

    def _restart(self, s: State, old: dict[str, Array], done: Array, smp: dict[str, Array]):
        """The done worlds' new episodes: (state, qpos, qvel, qacc_warmstart), whole arrays for Warp."""
        s = self._randomize(s, done, smp)
        tsk, N = self.task, self.num_envs
        key, k_q, k_g, k_r, k_b, k_yaw, k_t = jax.random.split(s["key"], 7)
        u = jax.random.uniform
        lo, hi = self.lo, self.hi
        noise = np.radians(tsk.start_noise_deg)
        na, g = self.limits.n_arm, self.limits.grip
        q0 = jnp.broadcast_to(self.home, (N, len(self.home)))
        q0 = q0.at[:, :na].add(u(k_q, (N, na), minval=-noise, maxval=noise))
        if tsk.cube:
            q0 = q0.at[:, g].set(lo[g] + u(k_g, (N,), minval=0.6, maxval=1.0) * (hi[g] - lo[g]))
        q0 = jnp.clip(q0, lo, hi)

        qpos = jnp.broadcast_to(self.qpos0, (N, len(self.qpos0))).at[:, self.qadr].set(q0)
        for k, adr, gain in self.kin.mirror:            # coupled joints (e.g. the A1X's second jaw)
            qpos = qpos.at[:, adr].set(gain * q0[:, k])
        if tsk.cube:
            r = u(k_r, (N,), minval=tsk.cube_radius[0], maxval=tsk.cube_radius[1])
            b = u(k_b, (N,), minval=tsk.cube_bearing[0], maxval=tsk.cube_bearing[1])
            p = jnp.stack([r * jnp.cos(b), r * jnp.sin(b), jnp.full_like(r, tk.rest_height(tsk))], 1)
            yaw = u(k_yaw, (N,), minval=-np.pi, maxval=np.pi)
            qb = self.q_base
            # a rotation about z in the base frame, then into the world: q_base * q_yaw
            cw, sw = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
            quat = jnp.stack([qb[0] * cw - qb[3] * sw, qb[1] * cw + qb[2] * sw,
                              qb[2] * cw - qb[1] * sw, qb[0] * sw + qb[3] * cw], 1)
            a = self.cube_qadr
            qpos = qpos.at[:, a:a + 3].set(p @ self.R_b.T + self.t_b)
            qpos = qpos.at[:, a + 3:a + 7].set(quat)
            # The first detection at the start: the cube lies still, so the whole history is one pose.
            # It is seen with the episode's bias (`_randomize` gave the done worlds theirs), no noise.
            z0, o1 = jnp.zeros_like(yaw), jnp.ones_like(yaw)
            R = jnp.stack([jnp.stack([jnp.cos(yaw), -jnp.sin(yaw), z0], -1),
                           jnp.stack([jnp.sin(yaw), jnp.cos(yaw), z0], -1),
                           jnp.stack([z0, z0, o1], -1)], -2)
            s["hist_pos"] = _sel_hist(done, p, s["hist_pos"])
            s["hist_rot"] = _sel_hist(done, R, s["hist_rot"])
            p_seen, _ = tk.perceived(jnp, p, R, s["bias"], jnp.zeros_like(s["bias"]), False, self.yaw_period)
            s["seen_pos"] = _sel(done, p_seen, s["seen_pos"])
            s["seen_rot"] = _sel(done, tk.fold_yaw(jnp, R, self.yaw_period) if self.rand.fold_yaw else R,
                                 s["seen_rot"])
            if tsk.target:
                # The target on the table top (z = 0) in the base frame; the mocap pad follows it.
                target = self._sample_targets(k_t, p)
                pad = target.at[:, 2].set(TARGET_SIZE[2])
                mocap = s["mocap_pos"].at[:, self.target_mocap].set(pad @ self.R_b.T + self.t_b)
                s["target"] = _sel(done, target, s["target"])
                s["mocap_pos"] = _sel(done, mocap, s["mocap_pos"])
        else:
            k = jax.random.randint(k_g, (N,), 0, self.goal_pool.shape[0])
            s["goal"] = _sel(done, self.goal_pool[k], s["goal"])
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
        """New episodes in every world: the observations (N, obs_dim)."""
        self._reset_worlds(jax.device_put(jnp.ones(self.num_envs, bool), self.device))
        self.s, obs = self._j_observe(self.s, self._phys)
        return obs

    def step(self, action: Array) -> tuple[Array, Array, Array, dict[str, Array]]:
        """One policy tick in every world: (obs, rew, done, info); done worlds start anew inside."""
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
