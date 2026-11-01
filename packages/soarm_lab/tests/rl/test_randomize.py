"""Training randomization planning needs no training dependencies."""

from soarm_lab.rl.randomize import Dynamics, Randomization, train_plan


def test_zero_spread_keeps_measured_dynamics_and_skips_randomized_evaluation():
    measured = Dynamics(kp=0.8, damping=1.2, delay=2, source="measured")
    rand, evaluations = train_plan(measured, 0)
    assert rand == Randomization.around(measured, 0)
    assert rand.centre == measured
    assert rand.fold_yaw and rand.cube_delay > 0
    assert not rand.randomized
    assert evaluations == [("bez randomizacji", "cpu_nominal", Randomization.nominal(measured))]
