"""The JAX PPO trainer (`rl.ppo`, `rl.jax_policy`) on the CPU, without MuJoCo Warp and without torch.

The pieces against plain numpy: GAE, the normalizer's merge (`torch_policy.Normalizer.update`'s
formula), the actor against `Policy.act`, the KL-adapted learning rate. The whole loop runs on a toy
environment with `BatchEnv`'s interface (`PointEnv`): a point on a plane driven to a goal. Skipped
without the `train` extra (jax, optax). Training on the arm's tasks needs a GPU: rl/test_batch.py.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

import jax.numpy as jnp  # noqa: E402

from soarm_lab.rl import jax_policy as jp  # noqa: E402
from soarm_lab.rl import ppo  # noqa: E402
from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.objects import DESCRIPTOR_DIM, descriptor  # noqa: E402
from soarm_lab.rl.policy import POLICY_FILE, Policy  # noqa: E402
from soarm_lab.station import Station  # noqa: E402

#: The toy environment: 30-step episodes, a step of at most 0.1 per action, success within 0.1 of the goal.
EPISODE, SPEED, RADIUS = 30, 0.1, 0.1
#: A run small enough for the CPU: one episode per world per iteration.
CFG = ppo.PPOConfig(num_envs=256, horizon=EPISODE, iterations=40, hidden=(32, 32), save_every=0)


def _sample(key, n):
    k_pos, k_goal = jax.random.split(key)
    return (jax.random.uniform(k_pos, (n, 2), jnp.float32, -0.8, 0.8),
            jax.random.uniform(k_goal, (n, 2), jnp.float32, -0.8, 0.8))


@jax.jit
def _point_step(pos, goal, t, action, key):
    pos = jnp.clip(pos + SPEED * jnp.clip(action, -1.0, 1.0), -1.0, 1.0)
    dist = jnp.linalg.norm(goal - pos, axis=-1)
    t = t + 1
    done = t >= EPISODE
    info = {"success": dist < RADIUS, "time_outs": done, "finished": jnp.zeros_like(done),
            "broken": jnp.zeros_like(done), "distance": dist}
    new_pos, new_goal = _sample(key, pos.shape[0])
    pos = jnp.where(done[:, None], new_pos, pos)
    goal = jnp.where(done[:, None], new_goal, goal)
    return pos, goal, jnp.where(done, 0, t), -dist, done, info


class PointEnv:
    """`batch.BatchEnv`'s interface in plain JAX: N points on a plane, each driven to its goal.

    The observation is (position, goal), the action a step of at most `SPEED` per axis (clipped to
    [-1, 1], as the batch environment does), the reward minus the distance. Every episode lasts
    `EPISODE` steps and ends on the time limit (`time_outs`); success is the point within `RADIUS`.
    """

    made: list[dict] = []

    def __init__(self, task, num_envs, *, station=None, randomization=None, device="cpu", seed=0):
        self.task, self.num_envs = task, int(num_envs)
        self.device = jax.devices("cpu")[0]
        self.key = jax.random.PRNGKey(seed + 1000)
        PointEnv.made.append({"device": device, "seed": seed, "task": task.name})

    @property
    def obs_dim(self) -> int:
        return 4

    @property
    def act_dim(self) -> int:
        return 2

    def reset(self):
        self.key, k = jax.random.split(self.key)
        self.pos, self.goal = _sample(k, self.num_envs)
        self.t = jnp.zeros(self.num_envs, jnp.int32)
        return jnp.concatenate([self.pos, self.goal], 1)

    def step(self, action):
        self.key, k = jax.random.split(self.key)
        self.pos, self.goal, self.t, rew, done, info = _point_step(self.pos, self.goal, self.t, action, k)
        return jnp.concatenate([self.pos, self.goal], 1), rew, done, info


@pytest.fixture(scope="module")
def station():
    return Station.load("sim")


def run(station, cfg=CFG, **kw):
    history = []
    kw.setdefault("on_progress", lambda p: history.append({"it": p.iteration, "success": p.success,
                                                           "reward": p.reward, "lr": p.lr, "std": p.std}))
    pol = ppo.train("reach", cfg, station=station, device="cpu", make_env=PointEnv, **kw)
    return pol, history


@pytest.fixture(scope="module")
def trained(station, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    pol, history = run(station, out_dir=out)
    return pol, history, out


# ----------------------------------------------------------------------- the pieces
def test_gae_matches_a_plain_loop():
    rng = np.random.default_rng(0)
    T, N, gamma, lam = 17, 9, 0.99, 0.95
    rew, val = rng.normal(size=(T, N)).astype(np.float32), rng.normal(size=(T, N)).astype(np.float32)
    done = (rng.random((T, N)) < 0.2).astype(np.float32)
    last_val = rng.normal(size=N).astype(np.float32)
    want, g = np.zeros((T, N)), np.zeros(N)
    for t in reversed(range(T)):                                    # the torch trainer's loop
        nxt = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt * (1 - done[t]) - val[t]
        g = delta + gamma * lam * (1 - done[t]) * g
        want[t] = g
    adv, ret = jax.jit(ppo.gae, static_argnums=(4, 5))(rew, val, done, last_val, gamma, lam)
    np.testing.assert_allclose(adv, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, want + val, rtol=1e-5, atol=1e-5)


def test_the_normalizer_merges_batches_as_torch():
    rng = np.random.default_rng(1)
    dim = 5
    mean, var, count = np.zeros(dim, np.float32), np.ones(dim, np.float32), np.float32(1e-4)
    norm = jp.Norm.new(dim)
    for n in (1, 7, 300, 64):
        x = (rng.normal(size=(n, dim)) * rng.uniform(0.1, 3, dim) + rng.normal(size=dim)).astype(np.float32)
        b_mean, b_var = x.mean(0), x.var(0)                       # torch: var(unbiased=False)
        delta, tot = b_mean - mean, count + n
        mean = mean + delta * n / tot
        var = (var * count + b_var * n + delta**2 * count * n / tot) / tot
        count = tot
        norm = jax.jit(jp.norm_update)(norm, x)
        np.testing.assert_allclose(norm.mean, mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(norm.var, var, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(norm.count, count, rtol=1e-6)
    assert float(norm.clip) == 5.0
    obs = (rng.normal(size=(50, dim)) * 4).astype(np.float32)
    pol = jp.to_policy(_meta(dim, 2, [8]), norm, jp.init_mlp(jax.random.PRNGKey(0), [dim, 8, 2]))
    np.testing.assert_allclose(jp.normalize(norm, obs), pol.normalize(obs), rtol=1e-6, atol=1e-6)


def test_the_actor_is_the_numpy_policy():
    """params -> `Policy` -> params changes no array, and `Policy.act` is the JAX actor's mean."""
    obs_dim, act_dim, hidden = 11, 3, [32, 16]
    k_actor, k_critic, k_obs = jax.random.split(jax.random.PRNGKey(2), 3)
    actor = jp.init_mlp(k_actor, [obs_dim, *hidden, act_dim])
    critic = jp.init_mlp(k_critic, [obs_dim, *hidden, 1])
    for (w, b), n_in in zip(actor, [obs_dim, *hidden], strict=True):   # torch's nn.Linear bounds
        assert np.abs(w).max() <= n_in**-0.5 and np.abs(b).max() <= n_in**-0.5
        assert np.abs(w).max() > 0.9 * n_in**-0.5
    obs = jax.random.normal(k_obs, (400, obs_dim)) * 2 + 0.5
    norm = jp.norm_update(jp.Norm.new(obs_dim), obs[:200])
    pol = jp.to_policy(_meta(obs_dim, act_dim, hidden), norm, actor, critic)
    assert pol.sizes == [obs_dim, *hidden, act_dim] and pol.critic is not None
    arrays = [pol.obs_mean, pol.obs_var, *(x for layer in pol.actor + pol.critic for x in layer)]
    assert all(type(x) is np.ndarray and x.dtype == np.float32 for x in arrays)  # numpy, not JAX
    assert type(pol.obs_count) is np.float32 and type(pol.obs_clip) is np.float32
    norm2, actor2, critic2 = jp.from_policy(pol)
    leaves = jax.tree_util.tree_leaves
    for a, b in zip(leaves((norm, actor, critic)), leaves((norm2, actor2, critic2)), strict=True):
        np.testing.assert_array_equal(a, b)
    mean = jnp.clip(jp.mlp(actor, jp.normalize(norm, obs)), -1, 1)
    np.testing.assert_allclose(pol.act(np.asarray(obs)), mean, rtol=0, atol=2e-6)
    np.testing.assert_allclose(pol.act(np.asarray(obs[0])), mean[0], rtol=0, atol=2e-6)
    assert jp.from_policy(_with_critic(pol, None))[2] is None


def test_the_learning_rate_follows_the_kl():
    dk, lr = 0.01, 3e-4
    adapt = lambda lr, kl: float(ppo.adapt_lr(jnp.float32(lr), jnp.float32(kl), dk))  # noqa: E731
    assert adapt(lr, 0.021) == pytest.approx(lr / 1.5)
    assert adapt(lr, 0.0049) == pytest.approx(lr * 1.5)
    for kl in (0.0, 0.005, 0.01, 0.02):                             # within the band, or no change at all
        assert adapt(lr, kl) == pytest.approx(lr)
    assert adapt(1.2e-5, 1.0) == pytest.approx(1e-5) and adapt(9e-3, 1e-4) == pytest.approx(1e-2)


@pytest.mark.parametrize("desired_kl, factor", [(1e6, 1.5), (1e-9, 1 / 1.5)])
def test_the_learning_rate_changes_after_every_minibatch(desired_kl, factor):
    cfg = replace(CFG, epochs=2, minibatches=3, desired_kl=desired_kl)
    state, roll, last_obs = _state_and_rollout(cfg)
    state = state._replace(params={**state.params, "log_std": jnp.full(2, np.log(5.0), jnp.float32)})
    new = jax.jit(lambda s, r, o, k: ppo._update(s, r, o, k, cfg=cfg, warming=False))(
        state, roll, last_obs, jax.random.PRNGKey(4))
    assert float(new.lr) == pytest.approx(cfg.lr * factor**6, rel=1e-5)
    assert np.all(np.asarray(new.params["log_std"]) <= np.log(ppo.STD_MAX) + 1e-6)   # clamped
    assert int(new.opt["policy"].count) == int(new.opt["critic"].count) == 6


def test_a_critic_warm_up_moves_the_critic_only():
    cfg = replace(CFG, epochs=2, minibatches=3, desired_kl=1e6)
    state, roll, last_obs = _state_and_rollout(cfg)
    new = jax.jit(lambda s, r, o, k: ppo._update(s, r, o, k, cfg=cfg, warming=True))(
        state, roll, last_obs, jax.random.PRNGKey(5))
    for key in ("actor", "log_std"):
        leaves = jax.tree_util.tree_leaves
        for a, b in zip(leaves(state.params[key]), leaves(new.params[key]), strict=True):
            np.testing.assert_array_equal(a, b)
    assert not np.array_equal(state.params["critic"][0][0], new.params["critic"][0][0])
    assert float(new.lr) == pytest.approx(cfg.lr)
    # The actor's Adam has not started: its first step after the warm-up is Adam's first, as in torch.
    assert int(new.opt["policy"].count) == 0 and int(new.opt["critic"].count) == 6


def test_the_device_names():
    assert ppo.jax_device("cpu") == jax.devices("cpu")[0]
    with pytest.raises(ValueError, match="unknown device"):
        ppo.jax_device("tpu:0")
    if not any(d.platform == "gpu" for d in jax.devices()):
        with pytest.raises(RuntimeError, match="--device cpu"):
            ppo.jax_device("cuda:0")


# ----------------------------------------------------------------------- the whole loop
def test_training_learns_the_toy_task_and_saves_the_policy(trained):
    pol, history, out = trained
    first, last = history[0], history[-1]
    assert [h["it"] for h in history] == list(range(1, CFG.iterations + 1))
    assert first["success"] < 0.2 and last["success"] > max(0.7, first["success"] + 0.5), history
    assert last["reward"] > first["reward"] + 5.0, history
    assert PointEnv.made[0] == {"device": "cpu", "seed": CFG.seed, "task": "reach"}
    saved = Policy.load(out / POLICY_FILE)
    assert saved.sizes == [4, 32, 32, 2] and saved.critic is not None
    assert saved.critic[-1][0].shape == (1, 32)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["iterations"] == CFG.iterations and meta["steps"] == CFG.iterations * EPISODE * CFG.num_envs
    assert meta["success"] == pytest.approx(last["success"])
    assert meta["evals"]["gpu"]["success"] == pytest.approx(last["success"])
    for key, value in pol.arrays().items():
        np.testing.assert_array_equal(saved.arrays()[key], value, err_msg=key)
    prog = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "gotowe" and prog["iteration"] == CFG.iterations
    assert len(prog["history"]) == CFG.iterations
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["ppo"]["hidden"] == [32, 32] and config["task"]["name"] == "reach"
    # The deterministic policy reaches the goal: the mean action of a point 0.3 off points at the goal.
    obs = np.array([[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, -0.3]], np.float32)
    a = pol.act(obs)
    assert a[0, 0] > 0.3 and abs(a[0, 1]) < a[0, 0] and a[1, 1] < -0.3 and abs(a[1, 0]) < -a[1, 1]


def test_checkpoints_and_the_stop_event(station, tmp_path):
    """`save_every` writes policy.npz on its iterations; `stop` ends the run with the policy saved."""
    stop = threading.Event()
    seen = []

    def on_progress(p):
        meta = tmp_path / "meta.json"
        seen.append(json.loads(meta.read_text())["iterations"] if meta.exists() else None)
        if p.iteration == 4:
            stop.set()

    pol, _ = run(station, replace(CFG, iterations=9, save_every=2), out_dir=tmp_path, stop=stop,
                 on_progress=on_progress)
    # The checkpoint is saved after on_progress; the event is checked at the end of an iteration,
    # before on_progress, so one set during iteration 4's ends the run after iteration 5.
    assert seen == [None, None, 2, 2, 4]
    assert pol.meta.iterations == 5
    assert Policy.load(tmp_path).meta.iterations == 5
    prog = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "przerwane" and prog["iteration"] == 5


def test_fine_tuning_starts_from_the_saved_actor_and_critic(station, trained):
    pol, history, _ = trained
    tuned, tuned_history = run(station, replace(CFG, iterations=2, init_std=0.25), init=pol)
    assert tuned.meta.notes.startswith("douczana z polityki z")
    assert tuned.sizes == pol.sizes and tuned.critic is not None
    # It starts where the first run ended, not from scratch.
    assert tuned_history[0]["success"] > 0.7 > history[0]["success"], tuned_history
    assert tuned.obs_count > pol.obs_count                 # the normalizer went on from the saved one


def test_fine_tuning_without_a_critic_warms_one_up_first(station, trained):
    """The actor and the normalizer hold for `critic_warmup` iterations while a new critic learns."""
    pol, _, _ = trained
    no_critic = _with_critic(pol, None)
    cfg = replace(CFG, iterations=2, init_std=0.25, critic_warmup=2)
    warm, history = run(station, cfg, init=no_critic)
    for key, value in no_critic.arrays().items():         # actor and normalizer: not a bit changed
        np.testing.assert_array_equal(warm.arrays()[key], value, err_msg=key)
    assert warm.critic is not None
    assert history[-1]["lr"] == pytest.approx(cfg.lr)      # the learning rate holds too
    after, _ = run(station, replace(cfg, iterations=3), init=no_critic)
    assert not np.array_equal(after.actor[0][0], pol.actor[0][0])
    assert after.obs_count > pol.obs_count


def test_a_policy_for_another_environment_is_refused(station, trained):
    pol, _, _ = trained
    other = _with_critic(pol, None)
    other.meta = replace(other.meta, obs_dim=5)
    other.actor = [(np.zeros((32, 5), np.float32), np.zeros(32, np.float32)), *pol.actor[1:]]
    with pytest.raises(ValueError, match="the actor of the init policy"):
        run(station, replace(CFG, iterations=1), init=other)


# ----------------------------------------------------------------------- the warm start to the blocks
def test_widening_to_the_descriptor_keeps_the_actions():
    """lift-v3's 33 inputs widened to 40: old columns copied, new ones zero, the normalizer extended.

    The widened actor gives the old action for any value of the new inputs.
    """
    obs_dim, wide, act_dim, hidden = 33, 33 + DESCRIPTOR_DIM, 6, [32, 16]
    k = jax.random.split(jax.random.PRNGKey(7), 5)
    actor = jp.init_mlp(k[0], [obs_dim, *hidden, act_dim])
    critic = jp.init_mlp(k[1], [obs_dim, *hidden, 1])
    norm = jp.norm_update(jp.Norm.new(obs_dim), jax.random.normal(k[2], (300, obs_dim)) * 2 + 0.3)
    like_actor = jp.init_mlp(k[3], [wide, *hidden, act_dim])
    like_critic = jp.init_mlp(k[3], [wide, *hidden, 1])
    w_actor = ppo._widen_layers("actor", actor, like_actor, DESCRIPTOR_DIM)
    w_critic = ppo._widen_layers("critic", critic, like_critic, DESCRIPTOR_DIM)
    w_norm = jp.widen_norm(norm, wide)
    for old, new in ((actor, w_actor), (critic, w_critic)):
        assert [w.shape for w, _ in new] == [(hidden[0], wide), *(w.shape for w, _ in old[1:])]
        np.testing.assert_array_equal(new[0][0][:, :obs_dim], old[0][0])
        np.testing.assert_array_equal(new[0][0][:, obs_dim:], 0.0)
        for (w0, _), (w1, _) in zip(old[1:], new[1:], strict=True):
            np.testing.assert_array_equal(w0, w1)
        for (_, b0), (_, b1) in zip(old, new, strict=True):
            np.testing.assert_array_equal(b0, b1)
    np.testing.assert_array_equal(w_norm.mean[:obs_dim], norm.mean)
    np.testing.assert_array_equal(w_norm.var[:obs_dim], norm.var)
    np.testing.assert_array_equal(w_norm.mean[obs_dim:], 0.0)
    np.testing.assert_array_equal(w_norm.var[obs_dim:], 1.0)
    assert float(w_norm.count) == float(norm.count) and float(w_norm.clip) == float(norm.clip)

    old_pol = jp.to_policy(_meta(obs_dim, act_dim, hidden), norm, actor)
    new_pol = jp.to_policy(_meta(wide, act_dim, hidden), w_norm, w_actor)
    obs = np.asarray(jax.random.normal(k[4], (50, obs_dim)) * 2, np.float32)
    zeros = np.concatenate([obs, np.zeros((50, DESCRIPTOR_DIM), np.float32)], 1)
    blocks = np.concatenate([obs, np.tile(descriptor("cylinder"), (50, 1))], 1)
    want = old_pol.act(obs)
    assert want.dtype == np.float32 and np.abs(want).max() > 0.05             # the actions are not all zero
    np.testing.assert_allclose(new_pol.act(zeros), want, rtol=0, atol=1e-6)
    np.testing.assert_allclose(new_pol.act(blocks), want, rtol=0, atol=1e-6)

    # Wider by anything but the descriptor, or narrower, is still refused.
    with pytest.raises(ValueError, match="the actor of the init policy"):
        ppo._widen_layers("actor", actor, jp.init_mlp(k[3], [obs_dim + 5, *hidden, act_dim]), DESCRIPTOR_DIM)
    with pytest.raises(ValueError, match="the actor of the init policy"):
        ppo._widen_layers("actor", actor, like_actor, 0)


#: The toy environment's object kinds: the first half of the worlds is a cube, the second a cylinder.
TOY_KINDS = ("cube", "cylinder")


class DescribedPointEnv(PointEnv):
    """`PointEnv` with an object descriptor after its observation and the kind of every world in `info`.

    It stands in for `mixed.MixedBatchEnv`: the descriptor does not change the task, so a policy of
    `PointEnv` widened to it keeps reaching its goals.
    """

    kinds = TOY_KINDS

    def __init__(self, task, num_envs, **kw):
        super().__init__(task, num_envs, **kw)
        half = self.num_envs // 2
        self.kind = jnp.repeat(jnp.arange(2, dtype=jnp.int32), half)
        self.desc = jnp.asarray(np.repeat(np.stack([descriptor(k) for k in TOY_KINDS]), half, axis=0))

    @property
    def obs_dim(self) -> int:
        return 4 + DESCRIPTOR_DIM

    def reset(self):
        return jnp.concatenate([super().reset(), self.desc], 1)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return jnp.concatenate([obs, self.desc], 1), rew, done, {**info, "kind": self.kind}


def test_a_policy_is_widened_to_a_shape_conditioned_task(station, trained, tmp_path):
    """`train` widens a policy without the descriptor, starts where it was, and reports success per kind."""
    pol, history, _ = trained
    lift_pol = _with_critic(pol, pol.critic)                    # the toy policy, as if trained on lift
    lift_pol.meta.task = {"name": "lift"}
    task = replace(tk.make_task("lift"), shape_conditioned=True)
    seen = []
    tuned = ppo.train(task, replace(CFG, iterations=2, init_std=0.25), station=station, device="cpu",
                      make_env=DescribedPointEnv, init=lift_pol, init_name="toy-v1", out_dir=tmp_path,
                      on_progress=lambda p: seen.append(replace(p, success_by_kind=dict(p.success_by_kind))))
    wide = 4 + DESCRIPTOR_DIM
    assert tuned.sizes == [wide, 32, 32, 2] and tuned.critic[0][0].shape == (32, wide)
    assert tuned.task.shape_conditioned and tuned.meta.obs_dim == wide
    assert "(toy-v1, " in tuned.meta.notes and f"z 4 do {wide} wejsc" in tuned.meta.notes
    # It starts where the toy run ended: the zero columns change nothing.
    assert seen[0].success > 0.7 > history[0]["success"], seen
    assert set(seen[0].success_by_kind) == set(TOY_KINDS)
    assert all(v > 0.5 for v in seen[0].success_by_kind.values()), seen[0].success_by_kind
    prog = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert set(prog["success_by_kind"]) == set(TOY_KINDS) and "success_cylinder" in prog["history"][-1]
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["kinds"] == list(TOY_KINDS)
    # Without `shape_conditioned` the wider environment is a mismatch.
    with pytest.raises(ValueError, match="the actor of the init policy"):
        ppo.train(tk.make_task("lift"), replace(CFG, iterations=1), station=station, device="cpu",
                  make_env=DescribedPointEnv, init=lift_pol)
    # A task of another name than the init policy's is refused.
    with pytest.raises(ValueError, match="trained on lift, the task is pickplace"):
        ppo.train(tk.make_task("pickplace"), replace(CFG, iterations=1), station=station, device="cpu",
                  make_env=DescribedPointEnv, init=lift_pol)


def test_success_by_kind_names_the_kinds():
    assert ppo._success_by_kind([1.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], ("cube", "cylinder")) == \
        {"cube": 0.5, "cylinder": 1.0}
    assert ppo._success_by_kind([1.0, 0.0], [2, 2], ()) == {"2": 0.5}


# ----------------------------------------------------------------------- helpers
def _meta(obs_dim, act_dim, hidden):
    from soarm_lab.rl.policy import PolicyMeta

    return PolicyMeta(task={"name": "reach"}, hidden=list(hidden), obs_dim=obs_dim, act_dim=act_dim)


def _with_critic(pol: Policy, critic) -> Policy:
    """A copy of `pol` (its own meta) with another critic."""
    out = Policy(replace(pol.meta))
    out.obs_mean, out.obs_var = pol.obs_mean, pol.obs_var
    out.obs_count, out.obs_clip = pol.obs_count, pol.obs_clip
    out.actor, out.critic = list(pol.actor), critic
    return out


def _state_and_rollout(cfg):
    """A training state and a random rollout (T, N) of the toy environment's sizes, as `train` builds them."""
    k = jax.random.split(jax.random.PRNGKey(3), 8)
    T, N, obs_dim, act_dim = 6, 40, 4, 2
    actor = jp.init_mlp(k[0], [obs_dim, *cfg.hidden, act_dim])
    critic = jp.init_mlp(k[1], [obs_dim, *cfg.hidden, 1])
    log_std = jnp.full(act_dim, np.log(cfg.init_std), jnp.float32)
    params = {"actor": actor, "log_std": log_std, "critic": critic}
    opt = {"policy": ppo._ADAM.init({"actor": actor, "log_std": log_std}), "critic": ppo._ADAM.init(critic)}
    state = ppo._State(params, opt, jp.Norm.new(obs_dim), jnp.asarray(cfg.lr, jnp.float32))
    normal = jax.random.normal
    roll = ppo._Rollout(obs=normal(k[2], (T, N, obs_dim)), act=normal(k[3], (T, N, act_dim)),
                        logp=normal(k[4], (T, N)) - 3.0, val=normal(k[5], (T, N)), rew=normal(k[6], (T, N)),
                        done=(jax.random.uniform(k[7], (T, N)) < 0.1) * 1.0)
    return state, roll, jax.random.normal(k[0], (N, obs_dim))
