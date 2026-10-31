from __future__ import annotations

import dataclasses
from importlib import metadata

import pytest
from machinekind_twin_api.entry_points import STRATEGIES
from machinekind_twin_api.strategy import Action, Observation, RunInfo

import soarm
from soarm_lab.robot import load_robot

#: The distribution's entry points: (group, name) -> value.
EXPECTED = {
    "soarm-lab": {
        (STRATEGIES, "policy"): "soarm_lab.strategies.policy:PolicyStrategy",
        (STRATEGIES, "ik"): "soarm_lab.strategies.ik:IKStrategy",
        (STRATEGIES, "dream"): "soarm_lab.strategies.dream:DreamStrategy",
        (STRATEGIES, "touch"): "soarm_lab.strategies.touch:TouchStrategy",
        (STRATEGIES, "touch_jev"): "soarm_lab.strategies.touch_jev:JevTouchStrategy",
        (STRATEGIES, "grasp"): "soarm_lab.strategies.grasp:GraspStrategy",
        (STRATEGIES, "grasp_wrist"): "soarm_lab.strategies.wrist:WristGraspStrategy",
        ("console_scripts", "soarm-lab"): "soarm_lab.cli:main",
    },
}
CALLABLE = [(dist, key) for dist, eps in EXPECTED.items() for key in eps]
#: The lab's strategies, by entry point name.
STRATEGY_NAMES = [name for group, name in EXPECTED["soarm-lab"] if group == STRATEGIES]
#: The ones that run without a network: `dream` needs Dream Labs' service, and
#: tests/strategies/test_dream.py runs it on a fake client. `touch_jev` refuses to start without
#: TYPESAFE_API_KEY, and tests/strategies/test_touch_jev.py runs this protocol check with a fake judge.
#: `grasp_wrist`'s default settings name the wrist camera and so need OpenCV, which the dev group does not
#: install: tests/strategies/test_wrist.py runs this check with `wrist_camera=""` and, where cv2 imports,
#: with the defaults.
OFFLINE_STRATEGIES = [name for name in STRATEGY_NAMES if name not in ("dream", "touch_jev", "grasp_wrist")]


def own_entry_points(dist: str) -> dict[tuple[str, str], metadata.EntryPoint]:
    """The distribution's own entry points."""
    try:
        found = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        pytest.skip(f"{dist} is not installed; run `uv sync`")
    return {(ep.group, ep.name): ep for ep in found.entry_points}


@pytest.mark.parametrize("dist", list(EXPECTED))
def test_exactly_the_expected_entry_points(dist: str) -> None:
    assert {key: ep.value for key, ep in own_entry_points(dist).items()} == EXPECTED[dist]


@pytest.mark.parametrize("dist, key", CALLABLE, ids=lambda k: k if isinstance(k, str) else k[1])
def test_callable_entry_points(dist: str, key: tuple[str, str]) -> None:
    assert callable(own_entry_points(dist)[key].load())


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_the_strategy_entry_point_is_a_class_with_settings(name: str) -> None:
    """What the twin's loader checks (`machinekind_twin.strategies.load_strategy`), without the twin."""
    cls = own_entry_points("soarm-lab")[(STRATEGIES, name)].load()
    assert isinstance(cls, type) and callable(getattr(cls, "act", None))
    assert dataclasses.is_dataclass(cls.settings)
    kinds = {f.name: f.type for f in dataclasses.fields(cls.settings)}
    assert set(kinds.values()) <= {"str", "int", "float", "bool"}


@pytest.mark.parametrize("name", OFFLINE_STRATEGIES)
def test_the_strategy_follows_the_strategy_protocol(name: str) -> None:
    """`machinekind_twin_api.Strategy`: built as `cls(settings)`, a positive `hz`, an `Action` from `act`."""
    cls = own_entry_points("soarm-lab")[(STRATEGIES, name)].load()
    strategy = cls(cls.settings())
    assert isinstance(strategy.hz, float) and strategy.hz > 0
    spec = load_robot()
    run = RunInfo(robot=spec.name, joints=tuple(spec.joints), gripper=spec.gripper, limits={},
                  max_vel=dict(spec.max_vel), folder=soarm.ROBOT_DIR, simulated=True)
    strategy.start(run, Observation(t=0.0, step=0, joints=dict(spec.home), command={}))
    action = strategy.act(Observation(t=1 / strategy.hz, step=1, joints=dict(spec.home), command={}))
    assert isinstance(action, Action)
    assert set(action.target) == set(spec.joints) and not action.done
    strategy.stop("zatrzymana")
