"""Test randomization plans and preservation of the random stream when new options are disabled.

Reset sampling shares this stream with dynamics sampling. These tests need only NumPy.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from soarm_lab.rl.randomize import Dynamics, Randomization, describe, train_plan


def test_zero_spread_keeps_measured_dynamics_and_skips_randomized_evaluation():
    measured = Dynamics(kp=0.8, damping=1.2, delay=2, source="measured")
    rand, evaluations = train_plan(measured, 0)
    assert rand == Randomization.around(measured, 0)
    assert rand.centre == measured
    assert rand.fold_yaw and rand.cube_delay > 0
    assert not rand.randomized
    assert evaluations == [("bez randomizacji", "cpu_nominal", Randomization.nominal(measured))]


def _base_draws(rng: np.random.Generator, r: Randomization, n: int = 1) -> None:
    """Advance the generator through the draws shared by all sampling configurations."""
    for lo_hi in (r.kp, r.damping, r.armature, r.frictionloss, r.cube_mass, r.cube_friction):
        rng.uniform(*lo_hi, n)
    rng.integers(min(r.min_delay, r.max_delay), r.max_delay + 1, n)
    rng.integers(0, r.cube_delay + 1, n)
    rng.integers(1, max(1, r.cube_period) + 1, n)


def test_the_pose_and_jaw_options_are_off_by_default():
    around = Randomization.around(Dynamics())
    for r in (Randomization(), Randomization.none(), Randomization.nominal(), around):
        assert (r.pose_bias_m, r.cube_min_delay, r.latch_m, r.grip_close) == (0.0, 0, 0.0, (1.0, 1.0))
        assert r.block_tolerance_mm == 0.0
        keys = {"pose_bias_m", "cube_min_delay", "latch_m", "grip_close", "block_tolerance_mm"}
        assert keys <= set(asdict(r))
    assert Dynamics().grip_close_s == 0.0
    none = Randomization.none()
    assert (none.kp, none.cube_friction, none.cube_delay, none.cube_period) == ((1.0, 1.0), (1.0, 1.0), 0, 1)

    a, b = np.random.default_rng(0), np.random.default_rng(0)
    smp = Randomization().sample(a, 1)
    _base_draws(b, Randomization())
    assert a.bit_generator.state == b.bit_generator.state
    assert [smp[k].tolist() for k in ("pose_bias_x", "pose_bias_y", "grip_close_s")] == [[0.0]] * 3
    assert [smp[k].tolist() for k in ("size_x", "size_y", "size_z")] == [[0.0]] * 3
    # A fixed closing time consumes no extra random draw.
    nominal = Randomization.nominal(Dynamics(grip_close_s=1.3))
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    smp = nominal.sample(a, 1)
    _base_draws(b, nominal)
    assert a.bit_generator.state == b.bit_generator.state
    assert smp["grip_close_s"].tolist() == [1.3]


def test_sample_draws_the_delay_bias_and_closing_time():
    rng = np.random.default_rng(0)
    n = 2000
    smp = replace(Randomization(), cube_min_delay=4, cube_delay=9).sample(rng, n)
    assert set(smp["cube_delay"].tolist()) == set(range(4, 10))
    # The lower delay bound is clamped to the upper bound.
    smp = replace(Randomization(), cube_min_delay=7, cube_delay=3).sample(rng, n)
    assert set(smp["cube_delay"].tolist()) == {3}

    smp = replace(Randomization(), pose_bias_m=0.03).sample(rng, n)
    bx, by = smp["pose_bias_x"], smp["pose_bias_y"]
    r = np.hypot(bx, by)
    assert r.max() <= 0.03 + 1e-12 and r.max() > 0.028
    quadrants = {(bool(x > 0), bool(y > 0)) for x, y in zip(bx, by, strict=True)}
    assert len(quadrants) == 4

    slow = replace(Randomization(), centre=Dynamics(grip_close_s=1.3), grip_close=(0.8, 1.25))
    close = slow.sample(rng, n)["grip_close_s"]
    assert close.min() >= 1.3 * 0.8 and close.max() <= 1.3 * 1.25 and np.ptp(close) > 0
    # A zero centre closing time consumes no extra random draw.
    free = replace(slow, centre=Dynamics())
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    assert free.sample(a, n)["grip_close_s"].tolist() == [0.0] * n
    _base_draws(b, free, n)
    assert a.bit_generator.state == b.bit_generator.state


def test_a_size_tolerance_draws_three_side_changes_per_world_after_the_jaw():
    """Side changes [mm] follow the existing draws. A zero tolerance consumes no extra random draws."""
    n = 500
    sized = replace(Randomization(), block_tolerance_mm=2.0)
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    smp = sized.sample(a, n)
    _base_draws(b, sized, n)
    for k in ("size_x", "size_y", "size_z"):
        assert np.array_equal(smp[k], b.uniform(-2.0, 2.0, n)), k
        assert smp[k].min() >= -2.0 and smp[k].max() <= 2.0 and np.ptp(smp[k]) > 3.5
    assert a.bit_generator.state == b.bit_generator.state
    assert not np.array_equal(smp["size_x"], smp["size_y"])

    assert replace(Randomization.none(), block_tolerance_mm=2.0).randomized
    assert not Randomization.none().randomized
    off = replace(sized, block_tolerance_mm=0.0)
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert off.sample(a, 3)["size_z"].tolist() == [0.0] * 3
    _base_draws(b, off, 3)
    assert a.bit_generator.state == b.bit_generator.state
    assert asdict(sized)["block_tolerance_mm"] == 2.0
    assert "; wymiary klocka +-2.0 mm" in describe(sized) and "wymiary klocka" not in describe(off)


def test_around_randomizes_the_measured_closing_time_only():
    assert Randomization.around(Dynamics()).grip_close == (1.0, 1.0)
    assert not Randomization.around(Dynamics(), 0.0).randomized
    slow = Dynamics(grip_close_s=1.3)
    assert Randomization.around(slow).grip_close == (0.8, 1.25)
    assert replace(Randomization.nominal(slow), grip_close=(0.8, 1.25)).randomized
    assert not Randomization.nominal(slow).randomized
    assert Randomization.around(slow, 0.0).grip_close == (1.0, 1.0)
    assert Randomization.around(slow, 0.5).grip_close == pytest.approx((0.9, 1.125))
    assert Randomization.nominal(slow).centre.grip_close_s == 1.3


def test_from_dict_reads_a_station_without_the_closing_time():
    assert Dynamics.from_dict({}).grip_close_s == 0.0
    dyn = Dynamics.from_dict({"grip_close_s": 1.3})
    assert dyn.grip_close_s == 1.3 and dyn.to_dict()["grip_close_s"] == 1.3


def test_the_plan_applies_the_perception_to_training_and_its_randomized_evaluation():
    changes = {"pose_bias_m": 0.02, "cube_min_delay": 4, "cube_delay": 20, "cube_period": 4, "latch_m": 0.05}
    rand, evals = train_plan(Dynamics(grip_close_s=1.3), 1.0, changes)
    assert (rand.pose_bias_m, rand.cube_min_delay, rand.cube_delay, rand.cube_period, rand.latch_m) == \
        (0.02, 4, 20, 4, 0.05)
    assert rand.grip_close == (0.8, 1.25)
    plan = {key: r for _, key, r in evals}
    assert plan["cpu_rand"] is rand
    nominal = plan["cpu_nominal"]
    assert (nominal.pose_bias_m, nominal.cube_period, nominal.centre.grip_close_s) == (0.0, 1, 1.3)
    text = describe(rand)
    assert "zamykanie chwytaka 1.30 s" in text and "odswiezanie co 1-4 takt." in text
    assert "przesuniecie do 2 cm" in text and "opoznienie pozy 4-20 takt." in text and "< 5 cm" in text
