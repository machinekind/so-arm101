"""The policy file: `policy.npz`, run in numpy, and torch checkpoints converted to it.

Running a policy needs numpy only (`soarm_lab.rl.policy.Policy`). The tests that compare with torch skip
without it (`soarm-lab[torch]`): numpy gives the torch policy's actions to 1e-6 on random observations,
for the bundled policies and for a checkpoint that `soarm-lab convert` turns into `policy.npz`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.policy import (  # noqa: E402
    FORMAT,
    POLICY_FILE,
    Policy,
    PolicyMeta,
    TorchCheckpoint,
    bundled_dir,
    find_policy,
)

BUNDLED = ("reach-v3", "lift-v3")
#: numpy against torch, both float32: their matrix products round differently (BLAS blocking,
#: torch adds the bias inside the product). Measured on the bundled policies: below 6e-7 for one
#: observation (the strategy, evaluation), below 1.1e-6 over 100 000 observations in batches of 5000.
#: torch's own actions for a batch and for its observations one by one differ by up to 9e-7.
ATOL = 1e-6


def observations(pol: Policy, n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Random observations around the training's mean, `scale` standard deviations wide."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, pol.meta.obs_dim))
    return (pol.obs_mean + scale * np.sqrt(pol.obs_var) * noise).astype(np.float32)


def random_policy(task: str = "lift", seed: int = 0, hidden=(32, 16)) -> Policy:
    t = tk.make_task(task)
    pol = Policy(PolicyMeta(task=asdict(t), hidden=list(hidden), obs_dim=t.obs_dim, act_dim=t.act_dim))
    rng = np.random.default_rng(seed)
    pol.obs_mean = rng.normal(0, 0.3, t.obs_dim).astype(np.float32)
    pol.obs_var = rng.uniform(0.01, 2.0, t.obs_dim).astype(np.float32)
    pol.obs_count = np.float32(1e6)
    sizes = pol.sizes

    def layers(n_out: list[int]):
        return [(rng.normal(0, a**-0.5, (b, a)).astype(np.float32), rng.normal(0, 0.1, b).astype(np.float32))
                for a, b in zip(sizes[:-1], n_out, strict=True)]

    pol.actor, pol.critic = layers(sizes[1:]), layers([*sizes[1:-1], 1])
    return pol


def as_json(meta: PolicyMeta) -> str:
    """Meta as `meta.json` holds it (`success` is NaN until training sets it, and NaN != NaN)."""
    return json.dumps(asdict(meta), default=float, sort_keys=True)


def reference(pol: Policy, obs: np.ndarray) -> np.ndarray:
    """The format's formula (the docstring of `soarm_lab.rl.policy`) in float64."""
    x = np.clip((obs - pol.obs_mean.astype(float)) / np.sqrt(pol.obs_var.astype(float) + 1e-8),
                -float(pol.obs_clip), float(pol.obs_clip))
    for i, (w, b) in enumerate(pol.actor):
        x = x @ w.astype(float).T + b
        if i < len(pol.actor) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return np.clip(x, -1, 1)


@pytest.mark.parametrize("name", BUNDLED)
def test_a_bundled_policy_is_policy_npz_with_its_critic(name):
    folder = bundled_dir() / name
    assert {p.name for p in folder.iterdir() if p.is_file()} == {"meta.json", POLICY_FILE}
    assert find_policy(name) == folder / POLICY_FILE
    with np.load(folder / POLICY_FILE, allow_pickle=False) as z:
        assert int(z["format"]) == FORMAT
        assert all(z[k].dtype == np.float32 for k in z.files if k not in ("format", "sizes"))
    pol = Policy.load(folder / POLICY_FILE)
    assert pol.sizes == [pol.task.obs_dim, 256, 256, 128, 6]
    assert pol.critic is not None and pol.critic[-1][0].shape == (1, 128)
    obs = observations(pol, 64, seed=1, scale=3.0)
    a = pol.act(obs)
    assert a.shape == (64, 6) and a.dtype == np.float32 and np.abs(a).max() <= 1.0
    assert pol.act(obs[0]).shape == (6,)
    np.testing.assert_allclose(a, reference(pol, obs), atol=1e-5)


def test_save_and_load_keep_every_array(tmp_path):
    pol = random_policy()
    path = pol.save(tmp_path / "run" / POLICY_FILE)
    meta = json.loads((tmp_path / "run" / "meta.json").read_text(encoding="utf-8"))
    assert json.dumps(meta, sort_keys=True) == as_json(pol.meta)
    back = Policy.load(tmp_path / "run")
    assert as_json(back.meta) == as_json(pol.meta)
    for key, value in pol.arrays().items():
        np.testing.assert_array_equal(back.arrays()[key], value, err_msg=key)
    obs = observations(pol, 100, seed=2, scale=2.0)
    np.testing.assert_array_equal(back.act(obs), pol.act(obs))
    np.testing.assert_allclose(pol.act(obs), reference(pol, obs), atol=1e-5)
    assert path == tmp_path / "run" / POLICY_FILE


def test_a_new_policy_acts_zero_and_a_torch_checkpoint_is_refused(tmp_path):
    t = tk.make_task("reach")
    pol = Policy(PolicyMeta(task=asdict(t), hidden=[8], obs_dim=t.obs_dim, act_dim=t.act_dim))
    assert not pol.act(np.ones(t.obs_dim)).any()
    (tmp_path / "policy.pt").write_bytes(b"not read")
    assert find_policy(tmp_path) == tmp_path / "policy.pt"
    with pytest.raises(TorchCheckpoint, match="soarm-lab convert .*soarm-lab\\[torch\\]"):
        Policy.load(tmp_path / "policy.pt")


@pytest.mark.parametrize("name", BUNDLED)
def test_numpy_acts_as_torch_on_the_bundled_policies(name):
    pytest.importorskip("torch")
    from soarm_lab.rl.torch_policy import TorchPolicy, layers_of, load_layers, mlp

    pol = Policy.load(find_policy(name))
    tp = TorchPolicy.from_numpy(pol)
    batch = observations(pol, 1000, seed=3)
    np.testing.assert_allclose(pol.act(batch), tp.act(batch), atol=ATOL, rtol=0)
    for obs in observations(pol, 200, seed=4, scale=2.0):              # one at a time, as the strategy
        np.testing.assert_allclose(pol.act(obs), tp.act(obs), atol=ATOL, rtol=0)
    # Back to numpy without a change, critic too (fine-tuning starts from the npz: `ppo.train(init=)`).
    critic = mlp(pol.sizes[:-1], 1)
    load_layers(critic, pol.critic)
    again = tp.to_numpy(critic)
    for key, value in pol.arrays().items():
        np.testing.assert_array_equal(again.arrays()[key], value, err_msg=key)
    assert [w.shape for w, _ in layers_of(tp.actor)] == [w.shape for w, _ in pol.actor]


@pytest.mark.parametrize("with_critic", [True, False])
def test_convert_turns_a_torch_checkpoint_into_the_same_policy(tmp_path, capsys, with_critic):
    torch = pytest.importorskip("torch")
    from soarm_lab.cli import main
    from soarm_lab.rl.torch_policy import TorchPolicy, layers_of, mlp

    torch.manual_seed(0)
    t = tk.make_task("lift")
    meta = PolicyMeta(task=asdict(t), hidden=[64, 32], obs_dim=t.obs_dim, act_dim=t.act_dim, steps=123,
                      notes="z torcha")
    tp = TorchPolicy(meta)
    with torch.no_grad():                    # a normalizer that has seen data, as after training
        tp.norm.update(torch.randn(500, t.obs_dim) * 0.4 + 0.1)
    critic = mlp([t.obs_dim, *meta.hidden], 1)
    blob = {"meta": asdict(meta), "state": tp.state_dict()}
    if with_critic:
        blob["critic"] = critic.state_dict()
    (tmp_path / "old").mkdir()
    torch.save(blob, tmp_path / "old" / "policy.pt")

    assert main(["convert", str(tmp_path / "old")]) == 0
    assert "policy.npz" in capsys.readouterr().out
    pol = Policy.load(tmp_path / "old" / POLICY_FILE)
    assert as_json(pol.meta) == as_json(meta)
    obs = observations(pol, 500, seed=5, scale=2.0)
    np.testing.assert_allclose(pol.act(obs), tp.eval().act(obs), atol=ATOL, rtol=0)
    for o in obs[:100]:
        np.testing.assert_allclose(pol.act(o), tp.act(o), atol=ATOL, rtol=0)
    if with_critic:
        for (w, b), (w0, b0) in zip(pol.critic, layers_of(critic), strict=True):
            np.testing.assert_array_equal(w, w0)
            np.testing.assert_array_equal(b, b0)
    else:
        assert pol.critic is None
