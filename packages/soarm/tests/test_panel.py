"""The test panel (`soarm-panel`): the controller on the fake bus, driven tick by tick, and its HTTP API.

`fake_bus.FakeBusLink` keeps a servo's present position where the test puts it, so a test sets where
the arm "got to" and checks what the panel then sends.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

import soarm
from fake_bus import REST_TICKS, FakeBusLink
from soarm.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_MAX_ANGLE_LIMIT,
    ADDR_MIN_ANGLE_LIMIT,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    FeetechArm,
    FeetechBus,
)
from soarm.panel import (
    DEFAULT_SPEED,
    HEARTBEAT_TIMEOUT_S,
    MAX_DT_S,
    PanelController,
    PanelError,
    PanelServer,
    RobotDescription,
    build_parser,
)

DT = 0.05
TICKS_PER_DEGREE = 4096 / 360


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Rig:
    panel: PanelController
    link: FakeBusLink
    clock: FakeClock


@dataclass
class Reply:
    status: int
    content_type: str
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.load(soarm.ROBOT_DIR)


def rig(description: RobotDescription, link: FakeBusLink | None = None) -> Rig:
    link = link or FakeBusLink()
    bus = FeetechBus("COM_TEST")
    bus.open(link=link)
    arm = FeetechArm(description.context("COM_TEST"), bus=bus)
    arm.connect()
    clock = FakeClock()
    return Rig(panel=PanelController(arm, description, clock=clock), link=link, clock=clock)


def ticks(degrees: float) -> int:
    return round(REST_TICKS + degrees * TICKS_PER_DEGREE)


def goals_sent(link: FakeBusLink, dev_id: int) -> list[int]:
    return [entry[3] for entry in link.ops("sync") if entry[1] == dev_id and entry[2] == ADDR_GOAL_POSITION]


# ------------------------------------------------------------- description
def test_the_description_builds_the_context_the_twin_hands_a_backend(description, so101_context):
    assert description.context("COM_TEST") == so101_context()


def test_the_page_gets_the_long_joint_labels(description):
    assert description.joint_labels["wrist_flex"] == "Nadgarstek - zgiecie"


# ------------------------------------------------------------------ targets
def test_jog_moves_the_target_by_the_step(description):
    r = rig(description)
    result = r.panel.jog("shoulder_pan", 5.0)
    assert result.target == pytest.approx(5.0)
    assert not result.clipped
    assert r.panel.state()["target"]["shoulder_pan"] == pytest.approx(5.0)


def test_jog_beyond_a_limit_clips_and_says_so(description):
    r = rig(description)
    result = r.panel.jog("shoulder_pan", 500.0)
    assert result.target == description.limits["shoulder_pan"].high
    assert result.clipped
    assert r.panel.state()["clip"]["joint"] == "shoulder_pan"


def test_a_command_inside_the_limits_clears_the_clip_notice(description):
    r = rig(description)
    r.panel.jog("shoulder_pan", 500.0)
    r.panel.jog("shoulder_pan", -10.0)
    assert r.panel.state()["clip"] is None


def test_the_eeprom_limit_narrows_the_robot_yaml_limit(description):
    link = FakeBusLink()
    link.registers[1][ADDR_MIN_ANGLE_LIMIT] = ticks(-45.0)
    link.registers[1][ADDR_MAX_ANGLE_LIMIT] = ticks(45.0)
    r = rig(description, link)
    assert r.panel.limits["shoulder_pan"].high == pytest.approx(45.0)
    assert r.panel.jog("shoulder_pan", 90.0).target == pytest.approx(45.0)


def test_a_value_that_is_not_a_number_is_refused(description):
    r = rig(description)
    with pytest.raises(PanelError):
        r.panel.set_target("shoulder_pan", float("nan"))


def test_gripper_open_targets_100(description):
    r = rig(description)
    assert r.panel.open_gripper().target == 100.0


def test_gripper_close_targets_0(description):
    r = rig(description)
    assert r.panel.close_gripper().target == 0.0


def test_home_targets_the_home_pose_and_ramps_there(description):
    r = rig(description)
    r.panel.home()
    assert r.panel.state()["target"] == description.home
    r.panel.tick(DT)
    assert r.panel.state()["command"]["elbow_flex"] == pytest.approx(DEFAULT_SPEED * DT)


# --------------------------------------------------------------------- ramp
def test_a_tick_moves_the_command_by_at_most_speed_times_dt(description):
    r = rig(description)
    r.panel.jog("shoulder_pan", 15.0)
    steps: list[float] = []
    previous = 0.0
    for _ in range(20):
        r.panel.tick(DT)
        command = r.panel.state()["command"]["shoulder_pan"]
        steps.append(command - previous)
        previous = command
    assert max(steps) <= DEFAULT_SPEED * DT + 1e-9
    assert previous == pytest.approx(15.0)


def test_the_speed_is_capped_by_each_joints_max_vel(description):
    r = rig(description)
    r.panel.set_speed(1000.0)
    r.panel.jog("shoulder_lift", 90.0)
    r.panel.tick(DT)
    capped = description.max_vel["shoulder_lift"] * DT
    assert r.panel.state()["command"]["shoulder_lift"] == pytest.approx(capped)


def test_a_stalled_tick_takes_no_bigger_step_than_a_short_one(description):
    r = rig(description)
    r.panel.jog("shoulder_pan", 90.0)
    r.panel.tick(5.0)
    assert r.panel.state()["command"]["shoulder_pan"] == pytest.approx(DEFAULT_SPEED * MAX_DT_S)


def test_a_joint_outside_its_limit_is_led_back_without_a_jump(description):
    """An arm folded at rest can stand past robot.yaml's limit; clamping the first command would snap it."""
    link = FakeBusLink()
    link.registers[2][ADDR_PRESENT_POSITION] = ticks(-100.0)
    r = rig(description, link)
    r.panel.tick(DT)
    assert link.registers[2][ADDR_GOAL_POSITION] == ticks(-100.0)


# --------------------------------------------------------------------- stop
def ramping(description: RobotDescription) -> Rig:
    """shoulder_pan on its way to 30 degrees, measured at 4 while the command is ahead of it."""
    r = rig(description)
    r.panel.jog("shoulder_pan", 30.0)
    for _ in range(3):
        r.panel.tick(DT)
    r.link.registers[1][ADDR_PRESENT_POSITION] = ticks(4.0)
    r.panel.tick(DT)
    return r


def test_stop_sets_target_and_command_to_the_measured_pose(description):
    r = ramping(description)
    r.panel.stop()
    state = r.panel.state()
    measured = state["measured"]["shoulder_pan"]
    assert state["target"]["shoulder_pan"] == measured
    assert state["command"]["shoulder_pan"] == measured


def test_after_stop_the_ticks_send_only_where_the_arm_stands(description):
    """The arm coasts a little after the last reading; the next tick holds it there, not back."""
    r = ramping(description)
    r.panel.stop()
    r.link.registers[1][ADDR_PRESENT_POSITION] = ticks(4.5)
    r.link.log.clear()
    r.panel.tick(DT)
    r.panel.tick(DT)
    assert set(goals_sent(r.link, 1)) == {ticks(4.5)}


def test_stop_leaves_torque_on(description):
    r = rig(description)
    r.panel.stop()
    r.panel.tick(DT)
    assert all(registers[ADDR_TORQUE_ENABLE] == 1 for registers in r.link.registers.values())


def test_close_holds_the_pose_with_torque_on(description):
    r = rig(description)
    r.panel.start()
    r.panel.jog("shoulder_pan", 30.0)
    r.panel.close()
    assert r.link.registers[1][ADDR_GOAL_POSITION] == REST_TICKS
    assert all(registers[ADDR_TORQUE_ENABLE] == 1 for registers in r.link.registers.values())


# ------------------------------------------------------------------- torque
def test_torque_on_writes_the_goal_before_enabling_torque(description):
    """The goal register remembers the previous session: torque enabled first would jerk the arm there."""
    r = rig(description)
    r.panel.set_torque(False)
    r.panel.tick(DT)
    r.link.registers[2][ADDR_PRESENT_POSITION] = ticks(-30.0)
    r.link.log.clear()

    r.panel.set_torque(True)
    r.panel.tick(DT)
    first_torque = next(
        i for i, entry in enumerate(r.link.log)
        if entry[0] == "write" and entry[2] == ADDR_TORQUE_ENABLE and entry[3] == 1
    )
    goal_writes = [
        i for i, entry in enumerate(r.link.log)
        if entry[0] == "sync" and entry[1] == 2 and entry[3] == ticks(-30.0)
    ]
    assert goal_writes, "cel nie zostal w ogole ustawiony"
    assert min(goal_writes) < first_torque
    assert all(registers[ADDR_TORQUE_ENABLE] == 1 for registers in r.link.registers.values())


def test_torque_off_releases_every_servo(description):
    r = rig(description)
    r.panel.set_torque(False)
    r.panel.tick(DT)
    assert all(registers[ADDR_TORQUE_ENABLE] == 0 for registers in r.link.registers.values())


def test_nothing_is_sent_while_torque_is_off(description):
    r = rig(description)
    r.panel.set_torque(False)
    r.panel.tick(DT)
    r.link.log.clear()
    r.panel.tick(DT)
    assert r.link.ops("sync") == []


def test_moves_are_refused_while_torque_is_off(description):
    r = rig(description)
    r.panel.set_torque(False)
    r.panel.tick(DT)
    with pytest.raises(PanelError):
        r.panel.jog("shoulder_pan", 5.0)


# ---------------------------------------------------------------- heartbeat
def test_a_stale_heartbeat_stops_a_running_ramp(description):
    r = rig(description)
    r.panel.heartbeat()
    r.panel.jog("shoulder_pan", 30.0)
    r.panel.tick(DT)

    r.clock.now += HEARTBEAT_TIMEOUT_S + 0.1
    r.panel.tick(DT)
    state = r.panel.state()
    assert state["target"]["shoulder_pan"] == state["measured"]["shoulder_pan"]
    assert state["command"] == state["target"]


def test_a_page_that_polls_keeps_the_ramp_going(description):
    r = rig(description)
    r.panel.jog("shoulder_pan", 30.0)
    for _ in range(4):
        r.clock.now += 1.0
        r.panel.heartbeat()
        r.panel.tick(DT)
    assert r.panel.state()["command"]["shoulder_pan"] == pytest.approx(4 * DEFAULT_SPEED * DT)


def test_a_stale_heartbeat_leaves_a_still_arm_alone(description):
    r = rig(description)
    r.link.registers[1][ADDR_PRESENT_POSITION] = REST_TICKS + 3
    r.clock.now += HEARTBEAT_TIMEOUT_S + 0.1
    r.panel.tick(DT)
    assert r.panel.state()["target"]["shoulder_pan"] == 0.0


# --------------------------------------------------------------------- HTTP
@pytest.fixture
def served(description) -> Iterator[tuple[str, Rig]]:
    r = rig(description)
    server = PanelServer(("127.0.0.1", 0), r.panel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", r
    server.shutdown()
    server.server_close()


def send(request: urllib.request.Request) -> Reply:
    try:
        with urllib.request.urlopen(request, timeout=5) as reply:
            return Reply(reply.status, reply.headers.get_content_type(), reply.read())
    except urllib.error.HTTPError as error:
        return Reply(error.code, error.headers.get_content_type(), error.read())


def get(url: str) -> Reply:
    return send(urllib.request.Request(url))


def post(url: str, body: object, content_type: str = "application/json") -> Reply:
    data = json.dumps(body).encode("utf-8")
    return send(urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": content_type}))


def test_the_http_api(served, description):
    base, r = served

    state = get(base + "/api/state")
    assert state.status == 200
    assert state.json()["joints"] == list(description.joints)

    jog = post(base + "/api/jog", {"joint": "elbow_flex", "delta": 5})
    assert jog.status == 200
    assert jog.json()["target"] == 5.0
    assert r.panel.state()["target"]["elbow_flex"] == 5.0

    page = get(base + "/")
    assert page.content_type == "text/html"
    assert b"<html" in page.body


def test_polling_the_state_is_the_heartbeat(served):
    base, r = served
    r.panel.jog("shoulder_pan", 30.0)
    r.clock.now += 1.0
    get(base + "/api/state")
    r.clock.now += 1.0
    r.panel.tick(DT)
    assert r.panel.state()["command"]["shoulder_pan"] == pytest.approx(DEFAULT_SPEED * DT)


def test_a_post_that_is_not_json_is_refused(served):
    """A cross-site form or text/plain POST needs no CORS preflight; it must not move the arm."""
    base, r = served
    reply = post(base + "/api/jog", {"joint": "elbow_flex", "delta": 5}, content_type="text/plain")
    assert reply.status == 415
    assert r.panel.state()["target"]["elbow_flex"] == 0.0


def test_a_refused_command_answers_400(served):
    base, _ = served
    assert post(base + "/api/jog", {"joint": "elbow", "delta": 5}).status == 400


def test_an_unknown_command_answers_404(served):
    base, _ = served
    assert post(base + "/api/dance", {}).status == 404


# ---------------------------------------------------------------------- CLI
def test_the_command_line_defaults():
    args = build_parser().parse_args(["--port", "COM5"])
    assert args.listen == "127.0.0.1:8765"
    assert args.baudrate == 1_000_000
    assert args.robot == soarm.ROBOT_DIR
    assert not args.no_browser
