"""The `lerobot` backend from the panel: it never disables torque and never asks anything in the console.

Without a calibration file `SOFollower.connect()` calls `calibrate()`, which disables
torque on all servos (an arm held by a `feetech` session drops onto the table) and hangs
on `input()` in the server console (reproduced). The tests run on stand-ins for LeRobot's
classes, without the library and without hardware.

Also the servo protection bits (`faults`) and lost position reads (while the link is
silent nothing goes to the servos), as in `feetech`.
"""

from __future__ import annotations

import builtins
import dataclasses
import enum

import pytest

from soarm.lerobot_backend import STATUS_EVERY_READS, LeRobotArm


@dataclasses.dataclass
class FakeConfig:
    port: str
    id: str | None = None
    use_degrees: bool = True
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True


class NormMode(enum.Enum):
    """Names as in `lerobot.motors.MotorNormMode`."""

    DEGREES = "degrees"
    RANGE_M100_100 = "range_m100_100"
    RANGE_0_100 = "range_0_100"


@dataclasses.dataclass
class Cal:
    """Fields as in `lerobot.motors.MotorCalibration`: what an entry in the calibration file looks like."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclasses.dataclass
class Motor:
    id: int
    model: str
    norm_mode: NormMode


#: A typical SO-101 calibration from `lerobot-calibrate` (ranges recorded by hand, wrist_roll a full turn).
TYPICAL_CALIBRATION = {
    "shoulder_pan": (1, 0, -1470, 758, 3292),
    "shoulder_lift": (2, 0, 157, 815, 3264),
    "elbow_flex": (3, 0, 1270, 858, 3093),
    "wrist_flex": (4, 0, -1014, 857, 3220),
    "wrist_roll": (5, 0, 1453, 0, 4095),
    "gripper": (6, 0, 1041, 2031, 3524),
}


class FakeBus:
    def __init__(self, calibrated: bool, torque: int, calibration: dict | None = None):
        self.calibrated = calibrated
        self.torque = torque
        self.connected = False
        self.log: list[tuple] = []
        self.calibration = calibration or {}
        self.apply_drive_mode = True
        self.model_resolution_table = {"sts3215": 4096}
        self.motors = {
            name: Motor(fields[0], "sts3215", NormMode.RANGE_0_100 if name == "gripper" else NormMode.DEGREES)
            for name, fields in TYPICAL_CALIBRATION.items()
        }
        #: The Status register of each servo (0x20 = overload) and whether reading it succeeds.
        self.status = {name: 0 for name in self.motors}
        self.status_fails = False
        self.status_reads = 0

    def connect(self) -> None:
        self.connected = True
        self.log.append(("connect",))

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False
        self.log.append(("disconnect", disable_torque))
        if disable_torque:
            self.torque = 0

    @property
    def is_calibrated(self) -> bool:
        # As in `FeetechMotorsBus`: the property reads the servos, so on an unconnected bus it
        # raises (DeviceNotConnectedError is a ConnectionError), which `hasattr` does not swallow.
        if not self.connected:
            raise ConnectionError("FeetechMotorsBus is not connected. Run `.connect()` first.")
        return self.calibrated

    def sync_read(self, name: str, motors=None, *, normalize: bool = True, num_retry: int = 0) -> dict[str, int]:
        if name == "Status":
            assert not normalize
            self.status_reads += 1
            if self.status_fails:
                raise ConnectionError("Failed to sync read 'Status' [TxRxResult] There is no status packet!")
            return dict(self.status)
        assert name == "Torque_Enable"
        return {"shoulder_pan": self.torque, "gripper": self.torque}

    def disable_torque(self) -> None:
        self.torque = 0
        self.log.append(("torque_off",))


class FakeRobot:
    """As much of `SOFollower` as the backend touches, with the same `connect(calibrate=True)` logic."""

    calibration_file = True
    servos_calibrated = True
    servos_torque = 1
    last: FakeRobot | None = None

    def __init__(self, config: FakeConfig):
        self.config = config
        self.calibration = ({name: Cal(*fields) for name, fields in TYPICAL_CALIBRATION.items()}
                            if self.calibration_file else {})
        self.calibration_fpath = "C:/cache/lerobot/calibration/robots/so_follower/so101_follower.json"
        self.bus = FakeBus(self.servos_calibrated, self.servos_torque, self.calibration)
        self.configured = False
        self.positions = {name: 0.0 for name in TYPICAL_CALIBRATION}
        self.sent: list[dict] = []
        FakeRobot.last = self

    @property
    def is_connected(self) -> bool:
        return self.bus.connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()

    def calibrate(self) -> None:
        self.bus.disable_torque()
        input("Move the arm to the middle ...")

    def configure(self) -> None:
        # As in LeRobot: the writes happen under `torque_disabled()`.
        self.bus.disable_torque()
        self.configured = True
        self.bus.torque = 1

    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

    @property
    def action_features(self) -> dict:
        return {f"{name}.pos": float for name in TYPICAL_CALIBRATION}

    #: This many consecutive position reads fail, as in LeRobot after `num_read_retries + 1` silences.
    lost_reads = 0

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise ConnectionError("SOFollower is not connected.")
        if self.lost_reads:
            self.lost_reads -= 1
            raise ConnectionError("Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 "
                                  "tries. [TxRxResult] There is no status packet!")
        return {f"{name}.pos": value for name, value in self.positions.items()}

    def send_action(self, action: dict) -> dict:
        self.sent.append(dict(action))
        return dict(action)


@pytest.fixture
def arm(monkeypatch, so101_context):
    FakeRobot.calibration_file, FakeRobot.servos_calibrated, FakeRobot.servos_torque = True, True, 1
    FakeRobot.last = None
    monkeypatch.setattr(LeRobotArm, "_resolve_classes", staticmethod(lambda kind: (FakeRobot, FakeConfig, "atrapa")))

    def no_console(*_a, **_k):
        raise AssertionError("backend zapytal o cos w konsoli serwera")

    monkeypatch.setattr(builtins, "input", no_console)
    return LeRobotArm(so101_context())


def test_missing_calibration_file_is_refused_before_touching_the_port(arm):
    FakeRobot.calibration_file = False
    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        arm.connect()
    robot = FakeRobot.last
    assert robot.bus.log == []                      # port untouched: torque stays as it was
    assert robot.bus.torque == 1
    assert not arm.is_connected


def test_calibration_that_differs_from_the_servos_is_refused_without_dropping_torque(arm):
    FakeRobot.servos_calibrated = False
    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        arm.connect()
    robot = FakeRobot.last
    assert ("torque_off",) not in robot.bus.log
    assert robot.bus.log[-1] == ("disconnect", False)
    assert robot.bus.torque == 1


def test_an_arm_holding_a_pose_is_connected_without_lerobot_configure(arm):
    """`configure()` disables torque for its writes: an arm in the air would drop."""
    arm.connect()
    robot = FakeRobot.last
    assert arm.is_connected and robot.bus.connected
    assert not robot.configured
    assert ("torque_off",) not in robot.bus.log


def test_lerobot_never_clamps_the_step_and_the_setting_is_gone(arm, so101_context):
    """LeRobot's +-12 degree step clamp hides collisions from the twin's tracking guard (25 degrees)."""
    arm.connect()
    assert FakeRobot.last.config.max_relative_target is None
    with pytest.raises(ValueError, match="max_relative_target"):
        LeRobotArm(so101_context(max_relative_target=12.0))


def test_disconnect_keeps_torque_unless_configured_otherwise(arm):
    arm.connect()
    robot = FakeRobot.last
    arm.disconnect()
    assert robot.bus.log[-1] == ("disconnect", False)
    assert robot.bus.torque == 1


# ------------------------------------------------------------ servo protection bits
@pytest.fixture
def live(arm):
    arm.connect()
    return arm, FakeRobot.last


def cycle(arm, n: int = 1) -> None:
    """`n` windows of STATUS_EVERY_READS position reads, each with exactly one Status read (at its start)."""
    for _ in range(n * STATUS_EVERY_READS):
        arm.read_joints()


def test_servo_protection_bits_are_reported_as_faults(live):
    """An overloaded shoulder releases torque but keeps answering: without Status it looks healthy
    and the wave carries on with the rest of the arm. The sentence is the same as in `feetech`."""
    arm, robot = live
    cycle(arm)
    assert arm.faults() == []
    robot.bus.status["shoulder_lift"] = 0x20
    cycle(arm)
    assert arm.faults() == ["shoulder_lift (serwo 2): przeciazenie"]
    robot.bus.status["shoulder_lift"] = 0
    cycle(arm)
    assert arm.faults() == []


# ------------------------------------------------------------ lost position reads
def test_a_lost_position_read_keeps_the_last_positions_and_marks_the_link_silent(live):
    """Three lost replies (num_read_retries=2) make LeRobot raise; that must not end the twin's loop."""
    arm, robot = live
    robot.positions["shoulder_pan"] = 12.5
    assert arm.read_joints()["shoulder_pan"] == 12.5 and not arm.link_silent
    robot.positions["shoulder_pan"] = 40.0
    robot.lost_reads = 1
    assert arm.read_joints()["shoulder_pan"] == 12.5     # the old position, not an exception
    assert arm.link_silent
    assert arm.faults() == []                            # one hiccup is not a failure yet
    # As in `feetech`: until a read confirms the link, nothing goes to the servos.
    assert arm.send_joints({"shoulder_pan": 30.0}) == {}
    assert robot.sent == []
    assert arm.read_joints()["shoulder_pan"] == 40.0
    assert not arm.link_silent
    assert arm.send_joints({"shoulder_pan": 30.0}) == {"shoulder_pan": 30.0}
