"""A browser panel for the SO-101: move the joints and the gripper from a web page.

`soarm-panel` connects the arm through the `feetech` backend, runs a 20 Hz control loop
and serves one page: `GET /` the page, `GET /api/state` the state, `POST /api/<command>` a command.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from machinekind_twin_api.backend import BackendContext

from . import ROBOT_DIR
from .feetech import ADDR_TORQUE_ENABLE, FeetechArm
from .serial_bridge import split_listen

logger = logging.getLogger(__name__)

TICK_HZ = 20.0
#: Default panel speed [deg/s for the arm joints, units/s for the gripper].
DEFAULT_SPEED = 30.0
#: A stalled tick (a slow bridge, a busy machine) must not turn into one big step at servo speed.
MAX_DT_S = 0.1
#: No `GET /api/state` for this long means the page is gone: a running ramp stops.
HEARTBEAT_TIMEOUT_S = 1.5


class PanelError(Exception):
    """A command the panel refuses, with the HTTP status it answers."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RobotDescription:
    """The part of `robot.yaml` the panel uses."""

    name: str
    joints: tuple[str, ...]
    gripper: str | None
    limits: dict[str, tuple[float, float]]
    max_vel: dict[str, float]
    home: dict[str, float]

    @classmethod
    def load(cls, folder: Path) -> RobotDescription:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError('soarm-panel needs pyyaml:  pip install "soarm[panel]"') from exc

        data = yaml.safe_load((Path(folder) / "robot.yaml").read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            joints=tuple(data["joints"]),
            gripper=data.get("gripper"),
            limits={name: tuple(pair) for name, pair in data["limits"].items()},
            max_vel=dict(data["max_vel"]),
            home=dict(data.get("home") or {}),
        )

    def context(self, port: str, settings: dict[str, Any]) -> BackendContext:
        """The `BackendContext` for the `feetech` backend; `settings` is its `feetech:` section."""
        return BackendContext(robot=self.name, joints=self.joints, gripper=self.gripper, limits=self.limits,
                              max_vel=self.max_vel, port=port, settings=settings)


class PanelController:
    """The control loop and the operator's commands on a connected `FeetechArm`.

    `target` is where the operator wants each joint, `command` what went to the servos last. Each tick
    reads the joints, moves `command` towards `target` by at most `speed * dt` per joint and sends it.
    One lock guards the state and the bus, held for a whole tick and for each command.
    """

    def __init__(self, arm: FeetechArm, description: RobotDescription) -> None:
        self.arm = arm
        self.description = description
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="panel-loop", daemon=True)

        eeprom = arm.joint_limits()
        self.limits: dict[str, tuple[float, float]] = {}
        for joint in description.joints:
            low, high = description.limits[joint]
            if joint in eeprom:
                low = max(low, eeprom[joint][0])
                high = min(high, eeprom[joint][1])
            self.limits[joint] = (low, high)

        self.measured = arm.read_joints()
        self.command = dict(self.measured)
        self.target = dict(self.measured)
        self.torque = True  # FeetechArm.connect enables it
        self.speed = DEFAULT_SPEED
        #: The last target cut to its limit, as a sentence for the page; None when it fitted.
        self.clip: str | None = None
        self.faults = arm.faults()
        self.last_heartbeat = time.monotonic()
        self._tick_times: list[float] = []

    def jog(self, joint: str, delta: float) -> None:
        with self._lock:
            if joint not in self.limits:
                raise PanelError(f"Unknown joint: {joint}")
            self._aim({joint: self.target[joint] + delta})

    def set_target(self, joint: str, value: float) -> None:
        with self._lock:
            if joint not in self.limits:
                raise PanelError(f"Unknown joint: {joint}")
            self._aim({joint: value})

    def home(self) -> None:
        """Targets the `home` pose of `robot.yaml`; the loop ramps there at the panel speed."""
        with self._lock:
            self._aim(self.description.home)

    def stop(self) -> None:
        """Holds the arm where it was measured last. Torque stays on, so the arm does not drop."""
        with self._lock:
            self._hold()

    def set_torque(self, on: bool) -> None:
        with self._lock:
            if on:
                # The goal register keeps where the servo was sent last: torque enabled before the goal
                # is overwritten would jerk the arm there. Same order as FeetechArm.connect.
                self.measured = self.arm.read_joints()
                self._hold()
                if not self._send(self.measured):
                    raise PanelError("The goal did not reach the servos; torque stays off.")
            for dev_id in self.arm.ids.values():
                self.arm.bus.write(dev_id, ADDR_TORQUE_ENABLE, 1 if on else 0, 1)
            self.torque = on

    def set_speed(self, value: float) -> None:
        if value <= 0.0:
            raise PanelError("The speed must be positive.")
        with self._lock:
            self.speed = value

    def state(self) -> dict[str, Any]:
        """What the page shows. The page polls it, so a call is also its heartbeat."""
        with self._lock:
            self.last_heartbeat = time.monotonic()
            return {
                "robot": self.description.name,
                "port": self.arm.port,
                "joints": list(self.description.joints),
                "gripper": self.description.gripper,
                "limits": self.limits,
                "measured": dict(self.measured),
                "command": dict(self.command),
                "target": dict(self.target),
                "torque": self.torque,
                "speed": self.speed,
                "clip": self.clip,
                "faults": self.faults,
                "loop_hz": len(self._tick_times),
            }

    def tick(self, dt: float) -> None:
        """One control cycle: read the joints, ramp `command` towards `target`, send it, collect faults."""
        with self._lock:
            now = time.monotonic()
            self._tick_times = [t for t in self._tick_times if now - t < 1.0]
            self._tick_times.append(now)
            self.measured = self.arm.read_joints()

            if now - self.last_heartbeat > HEARTBEAT_TIMEOUT_S and self.command != self.target:
                logger.warning("The panel page stopped polling; holding the arm.")
                self._hold()

            if self.torque:
                self._step(min(max(dt, 0.0), MAX_DT_S))
            else:
                # A limp arm is moved by hand: the hold point follows it.
                self._hold()

            self.faults = self.arm.faults()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Stops the loop and holds the arm where it stands, torque on."""
        self._halt.set()
        self._thread.join()
        self.stop()
        self.tick(0.0)

    def _run(self) -> None:
        last = time.monotonic()
        while not self._halt.is_set():
            now = time.monotonic()
            try:
                self.tick(now - last)
            except Exception as exc:
                logger.exception("The panel control loop failed")
                with self._lock:
                    self.faults = [f"control loop error: {exc}"]
            last = now
            self._halt.wait(max(0.0, 1.0 / TICK_HZ - (time.monotonic() - now)))

    # The methods below run under the lock.
    def _aim(self, goals: dict[str, float]) -> None:
        if not self.torque:
            raise PanelError("Torque is off; turn it on to move the arm.")

        self.clip = None
        for joint, wanted in goals.items():
            low, high = self.limits[joint]
            self.target[joint] = min(max(wanted, low), high)
            if self.target[joint] != wanted:
                name = joint.replace("_", " ")
                self.clip = f"clipped to the limit: {name} {wanted:.1f} -> {self.target[joint]:.1f}"

    def _hold(self) -> None:
        self.command = dict(self.measured)
        self.target = dict(self.measured)

    def _step(self, dt: float) -> None:
        step: dict[str, float] = {}
        for joint in self.description.joints:
            reach = min(self.speed, self.description.max_vel[joint]) * dt
            gap = self.target[joint] - self.command[joint]
            if abs(gap) <= reach:
                step[joint] = self.target[joint]
            else:
                step[joint] = self.command[joint] + math.copysign(reach, gap)

        # A send held back on a dead link must not advance the command: once the link is back,
        # the servos would jump to it.
        for joint in self._send(step):
            self.command[joint] = step[joint]

    def _send(self, goals: dict[str, float]) -> dict[str, float]:
        # The backend clamps to its limits before sending. A joint that starts outside its range (an arm
        # folded at rest) would jump to the limit at servo speed; widened to where the command stands,
        # it is led back by the ramp instead.
        widened: dict[str, tuple[float, float]] = {}
        for joint, value in goals.items():
            low, high = self.limits[joint]
            widened[joint] = (min(low, value), max(high, value))
        self.arm.set_joint_limits(widened)
        return self.arm.send_joints(goals)


class PanelServer(ThreadingHTTPServer):
    """The page and the JSON API of one `PanelController`."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: PanelController) -> None:
        self.controller = controller
        self.page = resources.files("soarm").joinpath("panel.html").read_bytes()
        super().__init__(address, PanelHandler)


class PanelHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        assert isinstance(self.server, PanelServer)
        path = urlsplit(self.path).path
        if path == "/":
            self._answer(HTTPStatus.OK, self.server.page, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._answer(HTTPStatus.OK, self.server.controller.state())
        else:
            self._answer(HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"})

    def do_POST(self) -> None:
        assert isinstance(self.server, PanelServer)
        # A cross-site page may send a form or text/plain POST without asking; a JSON one needs a CORS
        # preflight this server never grants. So no other website in the browser can move the arm.
        if self.headers.get_content_type() != "application/json":
            self._answer(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "expected application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except ValueError as exc:
            self._answer(HTTPStatus.BAD_REQUEST, {"error": f"Bad JSON: {exc}"})
            return

        try:
            run_command(self.server.controller, urlsplit(self.path).path.removeprefix("/api/"), body)
        except PanelError as exc:
            self._answer(exc.status, {"error": str(exc)})
        else:
            self._answer(HTTPStatus.OK, {})

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - " + format, self.address_string(), *args)

    def _answer(self, status: HTTPStatus, payload: object, content_type: str = "application/json") -> None:
        """Sends `payload`: bytes as they are, anything else as JSON."""
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def run_command(controller: PanelController, command: str, body: Any) -> None:
    """Runs `POST /api/<command>` with its JSON body. Raises PanelError for a refused or unknown one."""
    if not isinstance(body, dict):
        raise PanelError("Expected a JSON object.")

    if command == "jog":
        controller.jog(_arg(body, "joint", str), _arg(body, "delta", float))
    elif command == "target":
        controller.set_target(_arg(body, "joint", str), _arg(body, "value", float))
    elif command == "home":
        controller.home()
    elif command == "stop":
        controller.stop()
    elif command == "torque":
        controller.set_torque(_arg(body, "on", bool))
    elif command == "speed":
        controller.set_speed(_arg(body, "value", float))
    else:
        raise PanelError(f"Unknown command: {command}", HTTPStatus.NOT_FOUND)


def _arg(body: dict[str, Any], key: str, kind: type) -> Any:
    """The field `key` of a command's body, checked to be a `kind`: str, bool or a finite float."""
    value = body.get(key)
    if kind is float:
        # A JSON number arrives as int or float; true and false are ints to Python, not numbers here.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    elif isinstance(value, kind):
        return value
    raise PanelError(f"Bad field `{key}`: {value!r} (expected {kind.__name__}).")


def detect_port(ports: Iterable[Any]) -> str:
    """The one USB-serial device among `ports` (from `serial.tools.list_ports.comports()`).

    Raises ValueError when there is none or more than one; the message says to pass `--port`.
    """
    devices = sorted(str(port.device) for port in ports if getattr(port, "vid", None) is not None)
    if not devices:
        raise ValueError("No USB-serial device found. Plug the arm in, or pass --port.")
    if len(devices) > 1:
        raise ValueError(f"Several USB-serial devices: {', '.join(devices)}. Pass --port.")
    return devices[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soarm-panel", description="A browser test panel for the arm: "
                                     "joints and gripper, Home, STOP and torque.")
    parser.add_argument("--port", default=None,
                        help="the arm's serial port, e.g. COM5, /dev/ttyACM0 or socket://host:5555 "
                             "(default: the one USB-serial device)")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="baud rate (default 1 Mbaud)")
    parser.add_argument("--listen", default="127.0.0.1:8765",
                        help="where to listen; default `127.0.0.1:8765`, this machine only")
    parser.add_argument("--robot", type=Path, default=ROBOT_DIR,
                        help="the folder with robot.yaml (default: the SO-101 in the package)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    host, tcp_port = split_listen(args.listen)
    if host not in ("127.0.0.1", "localhost"):
        logger.warning("The panel listens on %s. Anyone who can reach this port can move the arm.", host)

    # On a failed start the process exit closes the port; torque stays on, as disconnect leaves it.
    try:
        description = RobotDescription.load(args.robot)
        port = args.port
        if port is None:
            from serial.tools.list_ports import comports

            port = detect_port(comports())
            logger.info("Port: %s", port)
        arm = FeetechArm(description.context(port, {"baudrate": args.baudrate}))
        arm.connect()
        controller = PanelController(arm, description)
        server = PanelServer((host, tcp_port), controller)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("The panel did not start: %s", exc)
        return 2

    controller.start()
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown_host}:{server.server_port}/"
    logger.info("Panel: %s  (Ctrl-C ends it; the arm keeps torque on)", url)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        server.server_close()
        controller.close()
        # FeetechArm.disconnect leaves torque on by default: a limp arm would drop.
        arm.disconnect()
    return 0
