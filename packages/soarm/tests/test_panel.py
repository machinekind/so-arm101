"""The test panel (`soarm-panel`) on the fake bus, which keeps each servo where the test puts it."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.client import HTTPConnection, HTTPResponse

import pytest
from lerobot_mp_api.backend import BackendContext

import soarm
from fake_bus import REST_TICKS, FakeBusLink
from soarm import feetech
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
DESCRIPTION = RobotDescription.load(soarm.ROBOT_DIR)


def ticks(degrees: float) -> int:
    return round(REST_TICKS + degrees * 4096 / 360)


def torque(link: FakeBusLink) -> set[int]:
    return {registers[feetech.ADDR_TORQUE_ENABLE] for registers in link.registers.values()}


def connect(link: FakeBusLink, context: BackendContext) -> PanelController:
    bus = feetech.FeetechBus("COM_TEST")
    bus.open(link=link)
    arm = feetech.FeetechArm(context, bus=bus)
    arm.connect()
    return PanelController(arm, DESCRIPTION)


@pytest.fixture
def link() -> FakeBusLink:
    return FakeBusLink()


@pytest.fixture
def panel(link: FakeBusLink, so101_context) -> PanelController:
    return connect(link, so101_context())


def test_description(so101_context):
    assert DESCRIPTION.context("COM_TEST", {}) == so101_context()
    assert DESCRIPTION.labels["wrist_flex"] == "Nadgarstek - zgiecie"


def test_eeprom_narrows_the_limit(so101_context):
    link = FakeBusLink()
    link.registers[1][feetech.ADDR_MIN_ANGLE_LIMIT] = ticks(-45.0)
    link.registers[1][feetech.ADDR_MAX_ANGLE_LIMIT] = ticks(120.0)
    panel = connect(link, so101_context())
    assert panel.limits["shoulder_pan"] == (-45.0, 105.0)


def test_clip_notice(panel):
    panel.jog("shoulder_pan", 500.0)
    assert panel.target["shoulder_pan"] == 105.0
    assert panel.clip == "przycieto do limitu: Obrot podstawy 500.0 -> 105.0"

    panel.jog("shoulder_pan", -10.0)
    assert panel.clip is None


def test_ramp(panel):
    panel.jog("shoulder_pan", 2.0)
    panel.tick(DT)
    assert panel.command["shoulder_pan"] == pytest.approx(DEFAULT_SPEED * DT)
    panel.tick(DT)
    assert panel.command["shoulder_pan"] == 2.0

    panel.jog("elbow_flex", 90.0)
    panel.tick(5.0)
    assert panel.command["elbow_flex"] == pytest.approx(DEFAULT_SPEED * MAX_DT_S)

    panel.set_speed(1000.0)
    panel.jog("wrist_flex", 90.0)
    panel.tick(DT)
    assert panel.command["wrist_flex"] == pytest.approx(DESCRIPTION.max_vel["wrist_flex"] * DT)


def test_stop(panel, link):
    panel.jog("shoulder_pan", 30.0)
    link.registers[1][feetech.ADDR_PRESENT_POSITION] = ticks(1.0)
    panel.tick(DT)

    panel.stop()
    assert panel.target == panel.measured
    assert panel.command == panel.measured

    # The arm coasts on after the last reading; the hold stays where STOP saw it.
    link.registers[1][feetech.ADDR_PRESENT_POSITION] = ticks(1.2)
    link.log.clear()
    panel.tick(DT)
    assert {value for _, dev_id, _, value in link.ops("sync") if dev_id == 1} == {ticks(1.0)}
    assert torque(link) == {1}


def test_torque_on_writes_the_goal_first(panel, link):
    """The goal register keeps where the servo was sent last: torque enabled first would jerk the arm."""
    panel.set_torque(False)
    link.registers[2][feetech.ADDR_PRESENT_POSITION] = ticks(-30.0)
    link.log.clear()

    panel.set_torque(True)

    goal = link.log.index(("sync", 2, feetech.ADDR_GOAL_POSITION, ticks(-30.0)))
    first_torque = link.log.index(("write", 1, feetech.ADDR_TORQUE_ENABLE, 1))
    assert goal < first_torque
    assert torque(link) == {1}


def test_torque_off(panel, link):
    panel.set_torque(False)
    assert torque(link) == {0}
    with pytest.raises(PanelError):
        panel.jog("shoulder_pan", 5.0)

    link.registers[1][feetech.ADDR_PRESENT_POSITION] = ticks(20.0)
    link.log.clear()
    panel.tick(DT)
    assert link.ops("sync") == []
    assert panel.target == panel.measured


def test_lost_page(panel, link):
    link.registers[1][feetech.ADDR_PRESENT_POSITION] = REST_TICKS + 3
    panel.last_heartbeat -= HEARTBEAT_TIMEOUT_S + 0.1
    panel.tick(DT)
    assert panel.target["shoulder_pan"] == 0.0

    panel.jog("shoulder_pan", 30.0)
    panel.tick(DT)
    assert panel.target == panel.measured
    assert panel.command == panel.measured


def test_dead_link_holds_the_command(panel, link):
    panel.jog("shoulder_pan", 30.0)
    link.silent = True
    panel.tick(DT)
    assert panel.command["shoulder_pan"] == 0.0


def test_joint_outside_its_limit_is_not_snapped(so101_context):
    """An arm folded at rest can stand past robot.yaml's limit; clamping the first command would snap it."""
    link = FakeBusLink()
    link.registers[2][feetech.ADDR_PRESENT_POSITION] = ticks(-100.0)
    panel = connect(link, so101_context())

    panel.jog("shoulder_lift", 10.0)
    panel.tick(DT)
    assert link.registers[2][feetech.ADDR_GOAL_POSITION] == ticks(-100.0 + DEFAULT_SPEED * DT)


@pytest.fixture
def port(panel: PanelController) -> Iterator[int]:
    server = PanelServer(("127.0.0.1", 0), panel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port
    server.shutdown()
    server.server_close()


def call(port: int, method: str, path: str, body: object = None,
         content_type: str = "application/json") -> HTTPResponse:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, json.dumps(body), {"Content-Type": content_type})
    return connection.getresponse()


def test_http_api(port, panel):
    assert b"<html" in call(port, "GET", "/").read()

    panel.last_heartbeat = 0.0
    assert json.load(call(port, "GET", "/api/state"))["joints"] == list(DESCRIPTION.joints)
    assert panel.last_heartbeat > 0.0

    assert call(port, "POST", "/api/jog", {"joint": "elbow_flex", "delta": 5}).status == 200
    assert panel.target["elbow_flex"] == 5.0

    # A cross-site form or text/plain POST needs no CORS preflight; it must not move the arm.
    assert call(port, "POST", "/api/jog", {"joint": "elbow_flex", "delta": 5}, "text/plain").status == 415
    assert call(port, "POST", "/api/jog", {"joint": "elbow", "delta": 5}).status == 400
    assert call(port, "POST", "/api/target", {"joint": "elbow_flex", "value": float("nan")}).status == 400
    assert call(port, "POST", "/api/dance", {}).status == 404
    assert panel.target["elbow_flex"] == 5.0


def test_close_leaves_torque_on(panel, link):
    """What Ctrl-C does before FeetechArm.disconnect, which leaves torque on too."""
    panel.jog("shoulder_pan", 30.0)
    panel.tick(DT)

    panel.start()
    panel.close()
    assert link.registers[1][feetech.ADDR_GOAL_POSITION] == REST_TICKS
    assert torque(link) == {1}


def test_cli_defaults():
    args = build_parser().parse_args(["--port", "COM5"])
    assert args.listen == "127.0.0.1:8765"
    assert args.baudrate == 1_000_000
    assert args.robot == soarm.ROBOT_DIR
    assert not args.no_browser
