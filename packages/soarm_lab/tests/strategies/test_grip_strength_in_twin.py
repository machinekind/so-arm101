"""The calibration wave's and the block strategies' grip strength reach the SO-101's gripper servo through the
twin's own loop and the `feetech` backend on the fake bus, and the stand's own comes back after them.

The stand's own gripper P comes from the config's `feetech: p_gain:`, as on the stand. The wave runs the
twin's own job with a stand-in session; `grasp` is built from its config section by the twin's loader.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.twin
pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

import machinekind_twin.calib.session as session_mod  # noqa: E402
from machinekind_twin import runtime  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy  # noqa: E402
from machinekind_twin.strategy import StrategyRunner  # noqa: E402
from machinekind_twin.ui import jobs  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from fake_bus import FakeBusLink  # noqa: E402
from soarm.feetech import ADDR_P_GAIN  # noqa: E402

#: The stand's own gripper P in these tests, the wave's and the gripper servo's id.
STAND_P = 4
WAVE_P = 16
GRIPPER_ID = 6
#: The twin loop's period [s].
TWIN_PERIOD = 0.02


@pytest.fixture
def stand(tmp_path, monkeypatch):
    """A twin driving the SO-101 through `feetech` on a fake bus, with the stand's config file."""
    link = FakeBusLink()
    config = tmp_path / "local.yaml"
    config.write_text(yaml.safe_dump({"feetech": {"p_gain": {"gripper": STAND_P}},
                                      "wave": {"grip_strength": WAVE_P}}), encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(config))
    real = runtime.create_backend

    def on_fake_bus(cfg, spec):
        arm = real(cfg, spec)
        arm.bus.open(link=link)
        return arm

    monkeypatch.setattr(runtime, "create_backend", on_fake_bus)
    twin = Twin(Workspace(robot="so101", path=tmp_path / "twin.json"), clock=VirtualClock(t0=0.0))
    twin.connect("feetech", port="COM_TEST", threaded=False)
    try:
        yield twin, link
    finally:
        twin.close()


def gripper_p(link: FakeBusLink) -> int:
    return link.registers[GRIPPER_ID][ADDR_P_GAIN]


def run(twin: Twin, seconds: float) -> None:
    for _ in range(round(seconds / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)


def drive(twin: Twin, runner: StrategyRunner, seconds: float) -> None:
    """The twin's loop and the strategy, one tick each per loop period."""
    for _ in range(round(seconds / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
        if runner.status.running:
            runner.step_once(twin.clock.now())


def test_connect_starts_with_the_stands_own_grip(stand):
    _, link = stand
    assert gripper_p(link) == STAND_P


@pytest.mark.parametrize("end", ["done", "stop", "error"])
def test_the_wave_holds_the_card_with_its_grip_and_the_stands_comes_back(stand, monkeypatch, end):
    twin, link = stand
    held: list[int] = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.done = False

        def step(self):
            run(twin, 0.1)
            held.append(gripper_p(link))
            if end == "stop":
                twin.estop("STOP z panelu")
                raise RuntimeError("fala przerwana")
            if end == "error":
                raise ZeroDivisionError("blad sesji")
            self.done = True
            return SimpleNamespace(progress={}, index=1, aim=None, phase="", seen={}, note="")

        def status(self):
            return {}

        def solve(self):
            return "wynik"

    monkeypatch.setattr(session_mod, "Session", Session)
    job = jobs.Job("fala")
    twin.claim(jobs.CALIB_OWNER, preempt=job.stop)               # as the panel takes the arm for the wave
    job.start(lambda j: jobs.run_card_calibration(j, twin, [], quick=True))
    assert job.wait(60.0)
    run(twin, 0.1)

    assert held == [WAVE_P]
    assert twin.owner is None and gripper_p(link) == STAND_P


def grasp(twin: Twin, **section) -> tuple[StrategyRunner, object]:
    strategy = load_strategy("grasp").create({"block": "cube", **section})
    runner = StrategyRunner(twin, strategy)
    runner.start(threaded=False)
    return runner, strategy


@pytest.mark.parametrize("end", ["stop", "estop", "home", "error"])
def test_grasp_squeezes_with_its_grip_and_the_stands_comes_back_however_it_ends(stand, monkeypatch, end):
    twin, link = stand
    runner, strategy = grasp(twin)                                # the default: gentle on a hard block
    drive(twin, runner, 0.2)
    assert runner.status.running and gripper_p(link) == 8

    if end == "stop":
        runner.stop()
    elif end == "estop":
        twin.estop("STOP z panelu")
    elif end == "home":
        twin.home()
    else:
        def broken(obs):
            raise ZeroDivisionError("act padl")
        monkeypatch.setattr(strategy, "act", broken)
    drive(twin, runner, 0.2)

    assert not runner.status.running and twin.owner is None
    assert gripper_p(link) == STAND_P


def test_the_next_run_gets_its_own_grip_or_the_stands(stand):
    twin, link = stand
    first, _ = grasp(twin, grip_strength=12)
    drive(twin, first, 0.2)
    assert gripper_p(link) == 12

    first.stop()
    second, _ = grasp(twin, grip_strength=0)                      # 0: the stand's own
    drive(twin, second, 0.2)
    assert second.status.running and gripper_p(link) == STAND_P

    second.stop()
    third, _ = grasp(twin, grip_strength=10)
    drive(twin, third, 0.2)
    assert gripper_p(link) == 10


def test_disconnect_puts_the_stands_grip_back_before_the_port_closes(stand):
    twin, link = stand
    runner, _ = grasp(twin, grip_strength=12)
    drive(twin, runner, 0.2)
    assert gripper_p(link) == 12

    twin.disconnect()
    assert link.closed and gripper_p(link) == STAND_P
