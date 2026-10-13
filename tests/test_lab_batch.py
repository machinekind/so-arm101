"""The GPU batch environment (MuJoCo Warp) and a short training run. Skipped without mujoco_warp or CUDA.

`uv sync --extra train` (or `pip install "soarm-lab[train]"`) brings mujoco-warp; training also needs a
CUDA build of torch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device for MuJoCo Warp", allow_module_level=True)

from soarm_lab.cli import main  # noqa: E402
from soarm_lab.rl.batch import BatchEnv  # noqa: E402
from soarm_lab.rl.policy import Policy  # noqa: E402
from soarm_lab.station import Station  # noqa: E402


@pytest.mark.parametrize("task", ["reach", "lift", "pickplace"])
def test_the_batch_steps_on_the_station(task):
    env = BatchEnv(task, num_envs=64, station=Station.load("sim"))
    obs = env.reset()
    assert obs.shape == (64, env.obs_dim)
    for _ in range(20):
        obs, rew, done, info = env.step(torch.rand(64, env.act_dim, device=env.device) * 2 - 1)
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    assert not info["broken"].any()


def test_a_short_training_run_saves_a_policy(tmp_path, capsys):
    assert main(["train", "--task", "reach", "--envs", "256", "--iters", "2", "--eval-episodes", "1",
                 "--out", str(tmp_path), "--name", "smoke"]) == 0
    pol = Policy.load(tmp_path / "smoke" / "policy.pt")
    assert pol.meta.iterations == 2 and set(pol.meta.evals) >= {"gpu", "cpu_nominal"}
