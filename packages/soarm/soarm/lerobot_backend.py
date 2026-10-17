"""Driving a real SO-101 arm through the LeRobot library.

LeRobot's module layout changed between releases (0.4 merged SO-100 and SO-101
into `lerobot.robots.so_follower`; before that there was a separate `so101_follower`,
and earlier still the whole package lived under `lerobot.common`). Instead of
assuming one version, the classes are looked up in every known location in turn.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from machinekind_twin_api.backend import BackendContext, RobotBackend, RobotInfo

from .feetech import LINK_LOSS_CYCLES, describe_servo_error
from .settings import from_mapping

logger = logging.getLogger(__name__)

#: Read the servos' Status register (protection bits: overload, overheating, voltage) every
#: this many position reads. LeRobot's `sync_read` does not return the reply error byte, so
#: without this a servo in protection (torque released, still answering politely) looks
#: healthy, and the wave and identification carry on with the rest of the arm.
#: At 10 Hz position reads that is one short SYNC READ every 0.3 s; the loop does not feel it.
#: Counted in reads, not seconds: the twin without its thread (tests) runs on its own clock,
#: the wall clock barely moves between its steps, and a time-based interval would never come round.
STATUS_EVERY_READS = 3

#: This many CONSECUTIVE failed Status reads (~1.5 s at 10 Hz) = servo protection is invisible,
#: which is a fault in itself. A single bus hiccup does not stop the arm.
STATUS_FAIL_LIMIT = 5

#: STS3215 ticks per turn, when LeRobot's bus gives no resolution table of its own.
DEFAULT_RESOLUTION = 4096

#: LeRobot's joint zero (the middle of the calibration range) may differ from the zero
#: the twin assumes (`lerobot.center_ticks`) by this many ticks (~1.8 degrees) before a warning.
ZERO_TOLERANCE_TICKS = 20

#: Locations of the SO-follower classes to try, newest first.
_MODULE_CANDIDATES: tuple[str, ...] = (
    "lerobot.robots.so_follower",
    "lerobot.robots.so101_follower",
    "lerobot.robots.so100_follower",
    "lerobot.common.robots.so101_follower",
    "lerobot.common.robots.so100_follower",
)


@dataclass
class LeRobotSettings:
    """The `lerobot:` section of the twin's config."""

    #: Arm id in LeRobot (selects the calibration file).
    robot_id: str = "so101_follower"
    #: Arm type passed to LeRobot: "so101" or "so100".
    kind: str = "so101"
    #: True -> joints in degrees (the default in LeRobot >= 0.4).
    use_degrees: bool = True
    calibration_dir: str | None = None
    #: The servo tick of joint zero in the twin's model (as in the `feetech` backend): LeRobot
    #: counts degrees from the middle of its calibration range, and the panel warns about the gap.
    center_ticks: int = 2048
    #: Torque off on exit (LeRobot's default is on: the arm drops under its own weight).
    torque_off_on_exit: bool = False


def create(ctx: BackendContext) -> LeRobotArm:
    """Entry point of the `lerobot` backend plugin."""
    return LeRobotArm(ctx)


class LeRobotArm(RobotBackend):
    """Adapter for LeRobot's SO follower (`SO101Follower` / `SO100Follower` / `SOFollower`)."""

    def __init__(self, ctx: BackendContext):
        self.settings: LeRobotSettings = from_mapping(LeRobotSettings, ctx.settings, "lerobot")
        self.port = ctx.port
        self.joints = tuple(ctx.joints)
        self.gripper = ctx.gripper
        self._robot: Any = None
        self._connected = False
        kind = self.settings.kind.lower()
        if kind not in ("so101", "so100"):
            raise ValueError(f"Nieznany typ ramienia: {self.settings.kind!r} (so101|so100)")
        self._kind = kind
        self._reset_status()
        self.info = RobotInfo(
            name=f"{kind.upper()} (LeRobot)",
            description=f"port {self.port}",
            simulated=False,
        )

    # ------------------------------------------------------------- imports
    @staticmethod
    def _resolve_classes(kind: str) -> tuple[Any, Any, str]:
        """Finds (robot class, config class, module name) in LeRobot."""
        robot_names = [f"{kind.upper()}Follower", "SOFollower"]
        config_names = [f"{kind.upper()}FollowerConfig", "SOFollowerRobotConfig", "SOFollowerConfig"]

        errors: list[str] = []
        for module_name in _MODULE_CANDIDATES:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                errors.append(f"{module_name}: {exc}")
                continue

            robot_cls = next((getattr(module, n) for n in robot_names if hasattr(module, n)), None)
            config_cls = next((getattr(module, n) for n in config_names if hasattr(module, n)), None)
            if robot_cls is not None and config_cls is not None:
                return robot_cls, config_cls, module_name
            errors.append(f"{module_name}: brak klas {robot_names} / {config_names}")

        raise ImportError(
            "Nie znalazlem klas SO-follower w zainstalowanej wersji LeRobot.\n"
            "Zainstaluj biblioteke:  pip install 'lerobot[feetech]'\n"
            "Szczegoly prob importu:\n  " + "\n  ".join(errors)
        )

    def _build_config(self, config_cls: Any) -> Any:
        """Builds the config object, skipping fields the installed version does not know."""
        rc = self.settings
        if not self.port:
            raise ValueError(
                "Nie podano portu szeregowego ramienia. Uzyj --port /dev/ttyACM0 "
                "(Linux), /dev/tty.usbmodem* (macOS) albo COM5 (Windows)."
            )

        wanted: dict[str, Any] = {
            "port": self.port,
            "id": rc.robot_id,
            "use_degrees": rc.use_degrees,
            # Always off: with LeRobot clamping a step to +-12 degrees from the measured position,
            # the gap between measurement and command never passed ~12 degrees, the twin's tracking
            # guard (25 degrees) could not trip, and the servo kept pushing into an obstacle
            # (measured on the real LeRobot stack over an emulator; without the clamp STOP came
            # after 0.78 s). The twin's supervisor limits speed anyway.
            "max_relative_target": None,
            # LeRobot disables torque on disconnect by default, and the arm drops under
            # its own weight. Here the same option as in `feetech` decides.
            "disable_torque_on_disconnect": rc.torque_off_on_exit,
        }
        if rc.calibration_dir:
            from pathlib import Path

            wanted["calibration_dir"] = Path(rc.calibration_dir)

        available = {f.name for f in dataclasses.fields(config_cls)}
        kwargs = {k: v for k, v in wanted.items() if k in available}
        skipped = sorted(set(wanted) - set(kwargs))
        if skipped:
            logger.warning(
                "Ta wersja LeRobot nie obsluguje pol %s - pomijam. "
                "Sprawdz, czy jednostki stawow zgadzaja sie z konfiguracja.",
                skipped,
            )
        return config_cls(**kwargs)

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        robot_cls, config_cls, module_name = self._resolve_classes(self._kind)
        logger.info("Uzywam klas LeRobot z modulu %s", module_name)

        config = self._build_config(config_cls)
        robot = robot_cls(config)
        self._require_calibration_file(robot)
        logger.info("Lacze z ramieniem na porcie %s ...", self.port)
        self._connect_without_calibration(robot)
        self._robot = robot
        self._connected = True
        self._reset_status()
        self._warn_about_joint_zero()

        motors = self.motor_names()
        missing = [n for n in self.joints if n not in motors]
        if missing:
            logger.warning(
                "Robot nie raportuje stawow %s - beda pomijane przy wysylaniu.", missing
            )
        logger.info("Polaczono. Stawy: %s", ", ".join(motors))

    def _calibrate_hint(self) -> str:
        return (f"Skalibruj ramie w konsoli:  lerobot-calibrate --robot.type={self._kind}_follower "
                f"--robot.port={self.port} --robot.id={self.settings.robot_id}  "
                "- albo uzyj backendu `feetech`.")

    def _require_calibration_file(self, robot: Any) -> None:
        """Refuses to connect, before anything touches the port, when LeRobot has no calibration file.

        Without the file `SOFollower.connect()` calls `calibrate()`: it disables torque on
        all servos (an arm held by the previous session drops onto the table) and waits
        for `input()` in the server console; nothing shows in the panel and the thread
        hangs. A finished calibration would also overwrite Homing_Offset and the limits
        in EEPROM, i.e. the zero and the ranges the `feetech` backend relies on.
        """
        if getattr(robot, "calibration", None):
            return
        where = getattr(robot, "calibration_fpath", None) or "katalog kalibracji LeRobota"
        raise RuntimeError(
            f"Brak kalibracji LeRobota dla ramienia '{self.settings.robot_id}' ({where}). "
            "Bez niej LeRobot przy laczeniu zdejmuje moment ze wszystkich serw i czeka na Enter "
            "w konsoli. " + self._calibrate_hint()
        )

    def _connect_without_calibration(self, robot: Any) -> None:
        """Connects so that LeRobot never disables torque or asks anything in the console."""
        bus = getattr(robot, "bus", None)
        # `getattr_static`, not `hasattr`: `is_calibrated` of LeRobot's bus is a property that
        # READS the servos, and on an unconnected bus it raises DeviceNotConnectedError (a
        # ConnectionError, not AttributeError). `hasattr` would call it before `bus.connect()`, and
        # LeRobot 0.6.1 with a calibration file would NEVER connect (reproduced on a bus emulator).
        missing = object()
        if (bus is not None and callable(getattr(bus, "connect", None))
                and inspect.getattr_static(bus, "is_calibrated", missing) is not missing):
            try:
                bus.connect()
                calibrated = bool(bus.is_calibrated)
                holding = calibrated and self._torque_on(bus)
            except Exception:
                self._close_bus_keeping_torque(bus)
                raise
            if not calibrated:
                self._close_bus_keeping_torque(bus)
                raise RuntimeError(
                    "Kalibracja LeRobota w pliku nie zgadza sie z zapisana w serwach (inne ramie "
                    "albo kalibracja zmieniona gdzie indziej). " + self._calibrate_hint()
                )
            if holding:
                # LeRobot's `configure()` does its writes under `torque_disabled()`. With the
                # arm holding a pose (e.g. right after a `feetech` session) that is torque off
                # for a few dozen transactions, and over the bridge a second of free fall.
                # The settings from earlier LeRobot connections are in the servos' EEPROM anyway.
                logger.warning("Serwa trzymaja pozycje - pomijam konfiguracje LeRobota, "
                               "zeby nie zdejmowac momentu (ustawienia zostaja z EEPROM).")
                return
            self._close_bus_keeping_torque(bus)

        if "calibrate" not in inspect.signature(robot.connect).parameters:
            raise RuntimeError(
                "Ta wersja LeRobota nie pozwala polaczyc bez interaktywnej kalibracji - "
                "zaktualizuj ja albo uzyj backendu `feetech`."
            )
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            try:
                robot.disconnect()
            except Exception:  # pragma: no cover - disconnecting must not hide the real error
                logger.exception("Blad przy rozlaczaniu robota")
            raise RuntimeError(
                "Kalibracja LeRobota w pliku nie zgadza sie z zapisana w serwach. " + self._calibrate_hint()
            )

    @staticmethod
    def _torque_on(bus: Any) -> bool:
        """Whether any servo holds torque. When in doubt, yes (safer)."""
        try:
            return any(int(v) for v in bus.sync_read("Torque_Enable").values())
        except Exception:
            logger.debug("Nie udalo sie odczytac Torque_Enable - zakladam, ze serwa trzymaja", exc_info=True)
            return True

    @staticmethod
    def _close_bus_keeping_torque(bus: Any) -> None:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # pragma: no cover - disconnecting must not hide the real error
            logger.exception("Blad przy zamykaniu magistrali LeRobota")

    def motor_names(self) -> list[str]:
        """Joint names reported by LeRobot (in the driver's order)."""
        if self._robot is None:
            return list(self.joints)
        try:
            return [key.removesuffix(".pos") for key in self._robot.action_features]
        except Exception:  # pragma: no cover - depends on the LeRobot version
            return list(self.joints)

    def disconnect(self) -> None:
        if self._robot is not None and self._connected:
            try:
                self._robot.disconnect()
            except Exception:  # pragma: no cover - disconnecting must not crash the application
                logger.exception("Blad przy rozlaczaniu robota")
        self._connected = False
        self._robot = None
        self._reset_status()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------- I/O
    def read_joints(self) -> dict[str, float]:
        """Joint positions; a failed read returns the previous ones and sets `link_silent` (as `feetech`).

        LeRobot raises ConnectionError when the SYNC READ of Present_Position got no reply
        in `num_read_retries + 1` tries: a short burst of noise while several joints move.
        Raising would end the twin's loop (disconnect, reconnect in the panel), while `feetech`
        rides out the same thing. Consecutive failed cycles go to `faults()`
        ("brak odpowiedzi serw ..."), which the twin treats as a dead link.
        """
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        try:
            observation = self._robot.get_observation()
        except ConnectionError as exc:
            if not self._positions or not self._robot_connected():
                raise                               # nothing to return, or the robot really is disconnected
            self._link_silent, self._stale = True, list(self._positions)
            self._failed_cycles += 1
            self._link_error = f"{type(exc).__name__}: {exc}"
            if self._failed_cycles == LINK_LOSS_CYCLES:
                logger.warning("Serwa nie odpowiadaja od %d cykli odczytu: %s", self._failed_cycles, self._link_error)
            else:
                logger.debug("Nieudany odczyt pozycji serw: %s", self._link_error)
            return dict(self._positions)
        self._poll_status()
        positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in observation.items()
            if key.endswith(".pos")
        }
        self._positions.update(positions)
        self._link_silent, self._stale = False, []
        self._failed_cycles, self._link_error = 0, ""
        self._last_full_read = time.monotonic()
        return dict(positions)

    def _robot_connected(self) -> bool:
        """Whether the robot is still connected: `DeviceNotConnectedError` is a ConnectionError too, and a
        disconnected robot is not a link hiccup."""
        try:
            return bool(getattr(self._robot, "is_connected", True))
        except Exception:  # pragma: no cover - depends on the LeRobot version
            return False

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        if self._link_silent:
            # As in `feetech`: the last read got no reply, so nothing is sent until a read
            # confirms that the servos hear and where the arm really is.
            return {}
        known = set(self.motor_names())
        # In degrees LeRobot does not clamp the goal (it clamps only the -100..100 and 0..100 scales),
        # and the servo silently clamps it to Min/Max_Position_Limit in EEPROM. Clamping here lets
        # the caller see what ACTUALLY went out; otherwise the supervisor, the policy and
        # identification would work from a position the arm never reached (as in `feetech`).
        limits = self.joint_limits()
        action: dict[str, float] = {}
        for name, value in targets.items():
            if name not in known:
                continue
            value = float(value)
            if name in limits:
                value = min(max(value, limits[name][0]), limits[name][1])
            action[f"{name}.pos"] = value
        try:
            sent = self._robot.send_action(action)
        except ConnectionError:
            # LeRobot raises ConnectionError when the bus write fails (and, with `max_relative_target`
            # set, which this backend never does, when the Present_Position read before the write
            # loses frames). Nothing is reported as sent; the caller stays on its previous command.
            if not self._robot_connected():
                raise
            logger.debug("Nieudany odczyt pozycji przed wyslaniem celu - nic nie wyslano", exc_info=True)
            return {}
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in (sent or action).items()
            if key.endswith(".pos")
        }

    # ------------------------------------------------------------ servo state
    def _reset_status(self) -> None:
        #: The last Status register read from each servo {joint: bits}.
        self._status: dict[str, int] = {}
        #: Position reads left until the next Status read (0 = at the next one).
        self._reads_to_status = 0
        #: CONSECUTIVE failed Status reads and the last error (for the message).
        self._status_failures = 0
        self._status_error = ""
        #: The last good positions, returned when a read gets no reply.
        self._positions: dict[str, float] = {}
        #: The last position read got no reply (`link_silent`), and the joints without a fresh value.
        self._link_silent = False
        self._stale: list[str] = []
        #: CONSECUTIVE failed position reads, the last error, and the time of the last good read.
        self._failed_cycles = 0
        self._link_error = ""
        self._last_full_read = time.monotonic()

    def _poll_status(self) -> None:
        """Reads the Status register of all servos every `STATUS_EVERY_READS` position reads. Never raises.

        The register has the same bits as the reply error byte (0x20 overload, 0x04
        overheating...), and even a servo in protection answers SYNC READ. A failed read
        keeps the previous state: overload bits do not vanish because of one lost frame.
        """
        if self._reads_to_status > 0:
            self._reads_to_status -= 1
            return
        self._reads_to_status = STATUS_EVERY_READS - 1
        try:
            raw = self._robot.bus.sync_read("Status", normalize=False)
            self._status = {str(name): int(bits) for name, bits in raw.items()}
            self._status_failures, self._status_error = 0, ""
        except Exception as exc:
            self._status_failures += 1
            self._status_error = f"{type(exc).__name__}: {exc}"
            if self._status_failures == STATUS_FAIL_LIMIT:
                logger.warning("Rejestr Status serw nie odpowiada od %d prob: %s",
                               self._status_failures, self._status_error)
            else:
                logger.debug("Nieudany odczyt Status serw: %s", self._status_error)

    def faults(self) -> list[str]:
        """Status protection bits, lost Status reads and link loss, as sentences for the operator.

        The same sentences as in `feetech` ("shoulder_lift (serwo 2): przeciazenie"), so the
        twin's supervisor stops the arm the same way on both backends.
        """
        try:
            out: list[str] = []
            ids = self._motor_ids()
            for name, bits in self._status.items():
                if bits:
                    out.append(describe_servo_error(name, ids.get(name, 0), bits))
            if self._connected and self._status_failures >= STATUS_FAIL_LIMIT:
                out.append(f"stan serw nieznany - {self._status_failures} nieudanych odczytow rejestru "
                           f"Status z rzedu, ochrona serw niewidoczna ({self._status_error})")
            if self._connected and self._failed_cycles >= LINK_LOSS_CYCLES:
                # The same sentence as in `feetech`: the twin recognises a dead link by it.
                ms = (time.monotonic() - self._last_full_read) * 1000.0
                out.append(f"brak odpowiedzi serw od {ms:.0f} ms ({self._link_error})")
            return out
        except Exception as exc:  # pragma: no cover - called every loop cycle, must not crash it
            logger.exception("Nie udalo sie sprawdzic stanu serw")
            return [f"nie udalo sie sprawdzic stanu serw: {exc}"]

    # ------------------------------------------------------------ calibration
    def _bus(self) -> Any:
        return getattr(self._robot, "bus", None) if self._robot is not None else None

    def _calibration(self) -> dict[str, Any]:
        """The calibration LeRobot actually applies (the bus's), otherwise the one from the file.

        Connecting checks (`is_calibrated`) that the file's ranges equal Min/Max_Position_Limit
        in the servos' EEPROM, so this is also what the servo clamps to.
        """
        cal = getattr(self._bus(), "calibration", None) or getattr(self._robot, "calibration", None)
        return dict(cal) if cal else {}

    def _motors(self) -> dict[str, Any]:
        return dict(getattr(self._bus(), "motors", None) or {})

    def _motor_ids(self) -> dict[str, int]:
        return {str(name): int(getattr(m, "id", 0)) for name, m in self._motors().items()}

    def _norm_mode(self, name: str) -> str:
        """LeRobot's normalisation mode of the joint: DEGREES, RANGE_M100_100 or RANGE_0_100."""
        mode = getattr(self._motors().get(name), "norm_mode", None)
        return str(getattr(mode, "name", mode) or "").upper()

    def _resolution(self, name: str) -> int:
        table = getattr(self._bus(), "model_resolution_table", None) or {}
        return int(table.get(getattr(self._motors().get(name), "model", None), DEFAULT_RESOLUTION))

    def _drive_inverted(self, cal: Any) -> bool:
        return bool(getattr(self._bus(), "apply_drive_mode", True)) and bool(getattr(cal, "drive_mode", 0))

    def _ticks_to_units(self, name: str, ticks: float) -> float | None:
        """Servo ticks -> LeRobot units, by the same formula as `MotorsBus._normalize`, unclamped.

        None = a joint without calibration or in an unknown mode.
        """
        cal = self._calibration().get(name)
        if cal is None:
            return None
        lo, hi = float(cal.range_min), float(cal.range_max)
        if hi == lo:
            return None
        mode = self._norm_mode(name)
        if mode == "DEGREES":
            # In degrees joint zero is the MIDDLE of the calibration range; drive_mode does not apply.
            return (ticks - (lo + hi) / 2.0) * 360.0 / (self._resolution(name) - 1)
        if mode == "RANGE_M100_100":
            norm = (ticks - lo) / (hi - lo) * 200.0 - 100.0
            return -norm if self._drive_inverted(cal) else norm
        if mode == "RANGE_0_100":
            norm = (ticks - lo) / (hi - lo) * 100.0
            return 100.0 - norm if self._drive_inverted(cal) else norm
        return None

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """The range from LeRobot's calibration (= Min/Max_Position_Limit in EEPROM) in app units.

        Only the joints the servo really limits: a full turn (0..4095, usually wrist_roll)
        is skipped, as in `feetech`. Without this the supervisor would work with targets
        outside the range, which the servo silently clamps.
        """
        out: dict[str, tuple[float, float]] = {}
        for name, cal in self._calibration().items():
            lo, hi = int(cal.range_min), int(cal.range_max)
            if lo >= hi or (lo <= 0 and hi >= self._resolution(name) - 1):
                continue
            a, b = self._ticks_to_units(name, lo), self._ticks_to_units(name, hi)
            if a is None or b is None:
                continue
            out[str(name)] = (min(a, b), max(a, b))
        return out

    def gripper_range_q(self) -> tuple[float, float] | None:
        """(closed, open) jaw angle [rad from joint zero] by LeRobot's calibration.

        LeRobot scales the gripper's 0..100 over `range_min..range_max` of its calibration
        (`drive_mode` reverses it), not over the ticks the twin's description assumes. Zero is
        half a turn (2047): LeRobot's calibration (`set_half_turn_homings`) puts the arm's middle
        pose there. The position register is the one `feetech` reads (the servo applies
        Homing_Offset), so the angles compare.
        """
        g = self.gripper
        cal = self._calibration().get(g) if g else None
        if cal is None or self._norm_mode(g) != "RANGE_0_100":
            return None
        lo, hi = float(cal.range_min), float(cal.range_max)
        closed, opened = (hi, lo) if self._drive_inverted(cal) else (lo, hi)
        res = self._resolution(g)
        zero = float((res - 1) // 2)
        return (closed - zero) * 2.0 * math.pi / res, (opened - zero) * 2.0 * math.pi / res

    def joint_zero_offsets(self) -> dict[str, float]:
        """How far LeRobot's joint zero lies from the twin's: {joint: degrees} (beyond the tolerance).

        In degrees LeRobot counts the angle from the MIDDLE of the calibration range, the twin
        (kinematics, policies) from `lerobot.center_ticks`, as the `feetech` backend does.
        For a typical SO-101 calibration that is a few degrees per joint; for a range recorded
        in one direction only (the limits of arm no. 1: wrist_flex 2025..2858) ~35 degrees.
        This is not converted silently: that would change the units of LeRobot-compatible recordings.
        """
        center = float(self.settings.center_ticks)
        out: dict[str, float] = {}
        for name, cal in self._calibration().items():
            if self._norm_mode(name) != "DEGREES":
                continue
            mid = (float(cal.range_min) + float(cal.range_max)) / 2.0
            if abs(mid - center) > ZERO_TOLERANCE_TICKS:
                out[str(name)] = (mid - center) * 360.0 / (self._resolution(name) - 1)
        return out

    def non_degree_joints(self) -> list[str]:
        """Arm joints (not the gripper) that LeRobot does NOT report in degrees.

        `lerobot.use_degrees: false` switches them to -100..100 percent of the calibration range,
        while kinematics, policies and IK treat every number as degrees: 100 "degrees" is then
        the end of the joint's range. `calibration_warnings` reports them to the panel.
        """
        return [str(name) for name in self._motors()
                if name != self.gripper and self._norm_mode(str(name)) != "DEGREES"]

    def calibration_warnings(self) -> list[str]:
        """Non-fatal caveats about this arm's calibration, as sentences for the panel. Never raises."""
        try:
            out: list[str] = []
            wrong_units = self.non_degree_joints()
            if wrong_units:
                out.append(f"UWAGA: stawy {', '.join(wrong_units)} ida z LeRobota w procentach zakresu "
                           f"(-100..100), a nie w stopniach (lerobot.use_degrees = {self.settings.use_degrees}) "
                           "- kinematyka i polityki blizniaka licza w stopniach, wiec kazdy ruch pojedzie "
                           "w zle miejsce. Ustaw lerobot.use_degrees: true albo uzyj backendu `feetech`.")
            off = self.joint_zero_offsets()
            if off:
                shifts = ", ".join(f"{name} {deg:+.1f} st." for name, deg in off.items())
                out.append(f"zero stawow w kalibracji LeRobota (srodek zakresu) nie jest zerem blizniaka "
                           f"(tik {int(self.settings.center_ticks)}) - katy z backendu `lerobot` sa "
                           f"przesuniete wzgledem modelu ({shifts}) - do pracy z blizniakiem uzyj backendu "
                           "`feetech`.")
            return out
        except Exception as exc:  # pragma: no cover - diagnostics alone must not block connecting
            logger.debug("Nie udalo sie porownac zera stawow z kalibracja LeRobota", exc_info=True)
            return [f"nie udalo sie sprawdzic kalibracji LeRobota: {exc}"]

    def _warn_about_joint_zero(self) -> None:
        for text in self.calibration_warnings():
            if text.startswith("UWAGA"):
                logger.error(text)
            else:
                logger.warning(text)

