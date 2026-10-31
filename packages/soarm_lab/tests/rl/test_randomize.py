"""Domain randomization (`rl/randomize.py`): the defaults change nothing, the new draws stay in range.

The perception additions (a per-episode offset of the seen pose, a least pose delay, the occlusion
latch) are off by default and draw nothing then: `ArmEnv.reset` draws the start pose and the cube from
the same generator after `Randomization.sample`, and `test_env.py` pins that stream. numpy only.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from soarm_lab.rl.randomize import Randomization


def _todays_draws(rng: np.random.Generator, r: Randomization, n: int = 1) -> None:
    """The nine draws `sample` made before the perception additions, in their order."""
    for lo_hi in (r.kp, r.damping, r.armature, r.frictionloss, r.cube_mass, r.cube_friction):
        rng.uniform(*lo_hi, n)
    rng.integers(min(r.min_delay, r.max_delay), r.max_delay + 1, n)
    rng.integers(0, r.cube_delay + 1, n)
    rng.integers(1, max(1, r.cube_period) + 1, n)


def test_the_defaults_keep_todays_perception():
    for r in (Randomization(), Randomization.none(), Randomization.nominal()):
        assert (r.pose_bias_m, r.cube_min_delay, r.latch_m) == (0.0, 0, 0.0)
        assert {"pose_bias_m", "cube_min_delay", "latch_m"} <= set(asdict(r))          # into meta.json
    # none() still builds its six ranges positionally: the new fields are appended, never inserted.
    none = Randomization.none()
    assert (none.kp, none.cube_friction, none.cube_delay, none.cube_period) == ((1.0, 1.0), (1.0, 1.0), 0, 1)

    # The generator ends where today's nine draws leave it: nothing else is drawn.
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    smp = Randomization().sample(a, 1)
    _todays_draws(b, Randomization())
    assert a.bit_generator.state == b.bit_generator.state
    assert smp["pose_bias_x"].tolist() == smp["pose_bias_y"].tolist() == [0.0]


def test_sample_draws_the_delay_and_bias():
    rng = np.random.default_rng(0)
    n = 2000
    smp = replace(Randomization(), cube_min_delay=4, cube_delay=9).sample(rng, n)
    assert set(smp["cube_delay"].tolist()) == set(range(4, 10))               # both ends hit
    # A least delay above the largest clamps to it.
    smp = replace(Randomization(), cube_min_delay=7, cube_delay=3).sample(rng, n)
    assert set(smp["cube_delay"].tolist()) == {3}

    smp = replace(Randomization(), pose_bias_m=0.03).sample(rng, n)
    bx, by = smp["pose_bias_x"], smp["pose_bias_y"]
    r = np.hypot(bx, by)
    assert r.max() <= 0.03 + 1e-12 and r.max() > 0.028
    quadrants = {(bool(x > 0), bool(y > 0)) for x, y in zip(bx, by, strict=True)}
    assert len(quadrants) == 4                                                # every bearing
