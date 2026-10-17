"""A browser test panel for the SO-101: move the joints and the gripper without the twin.

`soarm-panel --port <port>` connects the arm through the `feetech` backend, runs a 20 Hz control loop
and serves one page on a local HTTP server. The page jogs each joint, opens and closes the gripper,
goes home, stops and switches torque.

Three layers, so the logic is testable without HTTP and without hardware:

* `RobotDescription`: the part of `robot.yaml` the panel uses;
* `PanelController`: the control loop and the operator's commands, on a connected `FeetechArm`;
* `PanelServer` and `main`: the HTTP API (`GET /`, `GET /api/state`, `POST /api/<command>`) and the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lerobot_mp_api.backend import BackendContext

from . import ROBOT_DIR
from .feetech import ADDR_TORQUE_ENABLE, FeetechArm
from .serial_bridge import split_listen

logger = logging.getLogger(__name__)

ROBOT_FILE = "robot.yaml"
PAGE = "panel.html"

TICK_HZ = 20.0
#: Default panel speed [deg/s for the arm joints, units/s for the gripper].
DEFAULT_SPEED = 30.0
#: A stalled tick (a slow bridge, a busy machine) must not turn into one big step at servo speed.
MAX_DT_S = 0.1
#: No `GET /api/state` for this long means the page is gone: a running ramp stops.
HEARTBEAT_TIMEOUT_S = 1.5
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0


class PanelError(ValueError):
    """A command the panel refuses: an unknown joint, a value that is not a number, torque off."""


class UnknownCommand(PanelError):
    """`POST /api/<command>` with a command the panel does not have."""


@dataclass(frozen=True)
class Limit:
    """A joint range [app units]."""

    low: float
    high: float

    def clip(self, value: float) -> float:
        return min(max(value, self.low), self.high)

    def narrowed(self, low: float, high: float) -> Limit:
        return Limit(low=max(self.low, low), high=min(self.high, high))

    def widened_to(self, value: float) -> Limit:
        return Limit(low=min(self.low, value), high=max(self.high, value))


@dataclass(frozen=True)
class RobotDescription:
    """The part of `robot.yaml` the panel uses. Other keys are skipped."""

    name: str
    joints: tuple[str, ...]
    gripper: str | None
    limits: dict[str, Limit]
    max_vel: dict[str, float]
    home: dict[str, float]
    #: The long label of each joint, for the page.
    joint_labels: dict[str, str]

    @classmethod
    def load(cls, folder: Path) -> RobotDescription:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "soarm-panel wymaga biblioteki pyyaml. "
                'Zainstaluj:  pip install "soarm[panel]"'
            ) from exc

        path = Path(folder) / ROBOT_FILE
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        joints = tuple(str(name) for name in data["joints"])

        limits: dict[str, Limit] = {}
        for name, pair in data["limits"].items():
            limits[str(name)] = Limit(low=float(pair[0]), high=float(pair[1]))
        max_vel = {str(name): float(value) for name, value in data["max_vel"].items()}

        missing = [name for name in joints if name not in limits or name not in max_vel]
        if missing:
            raise ValueError(f"{path}: brak `limits` albo `max_vel` dla: {', '.join(missing)}")

        home = {str(name): float(value) for name, value in (data.get("home") or {}).items()}
        labels = {str(name): str(pair[1]) for name, pair in (data.get("joint_labels") or {}).items()}

        return cls(
            name=str(data["name"]),
            joints=joints,
            gripper=data.get("gripper"),
            limits=limits,
            max_vel=max_vel,
            home=home,
            joint_labels={name: labels.get(name, name) for name in joints},
        )

    def context(self, port: str, settings: Mapping[str, Any] | None = None) -> BackendContext:
        """What the twin would hand the `feetech` backend for this robot.

        `settings` is the backend's `feetech:` section.
        """
        return BackendContext(
            robot=self.name,
            joints=self.joints,
            gripper=self.gripper,
            limits={name: (limit.low, limit.high) for name, limit in self.limits.items()},
            max_vel=dict(self.max_vel),
            port=port,
            settings=dict(settings or {}),
        )


@dataclass(frozen=True)
class TargetResult:
    """Where a command put a joint's target, and whether a limit cut it."""

    joint: str
    requested: float
    target: float
    clipped: bool


class PanelController:
    """The panel's control loop on a connected `FeetechArm`.

    `target` is where the operator wants each joint, `command` what went to the servos last. Every tick
    reads the joints, moves `command` towards `target` by at most `speed * dt` per joint and sends it.
    Only the control thread talks to the bus; the HTTP threads change `target` and leave requests.
    """

    def __init__(
        self,
        arm: FeetechArm,
        description: RobotDescription,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.arm = arm
        self.description = description
        self._clock = clock
        self._lock = threading.Lock()

        hardware = arm.joint_limits()
        self.limits: dict[str, Limit] = {}
        for name in description.joints:
            limit = description.limits[name]
            if name in hardware:
                limit = limit.narrowed(hardware[name][0], hardware[name][1])
            self.limits[name] = limit

        measured = arm.read_joints()
        self._measured = dict(measured)
        self._command = dict(measured)
        self._target = dict(measured)
        self._faults: list[str] = arm.faults()
        # FeetechArm.connect enables torque.
        self._torque = True
        self._speed = DEFAULT_SPEED
        self._clip: TargetResult | None = None

        self._torque_request: bool | None = None
        self._hold_requested = False
        self._last_heartbeat = clock()
        self._last_tick_at: float | None = None
        self._period_s = 0.0
        self._loop_error: str | None = None

        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ operator
    def jog(self, joint: str, delta: float) -> TargetResult:
        with self._lock:
            self._require_torque()
            self._require_joint(joint)
            result = self._set_target(joint, self._target[joint] + _finite(delta))
            self._note_clips([result])
            return result

    def set_target(self, joint: str, value: float) -> TargetResult:
        with self._lock:
            self._require_torque()
            self._require_joint(joint)
            result = self._set_target(joint, _finite(value))
            self._note_clips([result])
            return result

    def open_gripper(self) -> TargetResult:
        return self.set_target(self._gripper(), GRIPPER_OPEN)

    def close_gripper(self) -> TargetResult:
        return self.set_target(self._gripper(), GRIPPER_CLOSED)

    def home(self) -> list[TargetResult]:
        """Targets the `home` pose of `robot.yaml`; the loop ramps there at the panel speed."""
        with self._lock:
            self._require_torque()
            results: list[TargetResult] = []
            for joint in self.description.joints:
                if joint in self.description.home:
                    results.append(self._set_target(joint, self.description.home[joint]))
            self._note_clips(results)
            return results

    def stop(self) -> None:
        """Holds the arm where it stands. Torque stays on, so the arm does not drop."""
        with self._lock:
            self._hold_at(self._measured)
            # The next tick holds at its own, fresher reading: the arm moved on since the last one.
            self._hold_requested = True

    def set_torque(self, on: bool) -> None:
        """Asks the control thread to switch torque on every servo at its next tick."""
        with self._lock:
            self._torque_request = bool(on)

    def set_speed(self, value: float) -> float:
        speed = _finite(value)
        if speed <= 0.0:
            raise PanelError("Predkosc musi byc dodatnia.")
        with self._lock:
            self._speed = min(speed, max(self.description.max_vel.values()))
            return self._speed

    def heartbeat(self) -> None:
        """The page is open: it polls the state."""
        with self._lock:
            self._last_heartbeat = self._clock()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "robot": self.description.name,
                "port": self.arm.port,
                "connected": self.arm.is_connected,
                "torque": self._torque,
                "joints": list(self.description.joints),
                "gripper": self.description.gripper,
                "labels": dict(self.description.joint_labels),
                "measured": dict(self._measured),
                "command": dict(self._command),
                "target": dict(self._target),
                "limits": {name: asdict(limit) for name, limit in self.limits.items()},
                "faults": self._all_faults(),
                "loop_hz": 1.0 / self._period_s if self._period_s > 0.0 else 0.0,
                "speed": self._speed,
                "clip": None if self._clip is None else asdict(self._clip),
                "home": dict(self.description.home),
            }

    # ---------------------------------------------------------------- loop
    def tick(self, dt: float) -> None:
        """One control cycle: read, apply requests, ramp `command` towards `target`, send, collect faults."""
        measured = self.arm.read_joints()

        with self._lock:
            self._measured = dict(measured)
            self._note_tick()
            torque_request = self._torque_request
            self._torque_request = None
            if self._hold_requested:
                self._hold_at(measured)
                self._hold_requested = False
            if self._heartbeat_lost() and self._command != self._target:
                logger.warning("Strona panelu nie odpowiada - zatrzymuje ruch.")
                self._hold_at(measured)

        if torque_request is not None:
            self._switch_torque(torque_request, measured)

        proposed: dict[str, float] | None = None
        with self._lock:
            if self._torque:
                proposed = self._ramped(min(max(dt, 0.0), MAX_DT_S))
            else:
                # A limp arm is moved by hand: the hold point follows it.
                self._hold_at(measured)

        if proposed is not None:
            sent = self._send(proposed)
            # A send held back on a dead link must not advance the command: once the link is back,
            # the servos would jump to it.
            with self._lock:
                for name in sent:
                    self._command[name] = proposed[name]

        faults = self.arm.faults()
        with self._lock:
            self._faults = faults

    def start(self) -> None:
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name="panel-petla", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stops the loop and holds the arm where it stands, torque on."""
        self._halt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.stop()
        try:
            self.tick(0.0)
        except Exception:  # pragma: no cover - shutting down must not crash the command
            logger.exception("Nie udalo sie zatrzymac ramienia przy zamykaniu panelu")

    def _run(self) -> None:
        period = 1.0 / TICK_HZ
        last = self._clock()
        next_at = last
        while True:
            next_at += period
            if self._halt.wait(max(0.0, next_at - self._clock())):
                return
            now = self._clock()
            if now - next_at > period:
                # A whole cycle behind (a slow bridge): start over rather than catch up in a burst.
                next_at = now
            try:
                self.tick(now - last)
                error = None
            except Exception as exc:
                logger.exception("Blad petli sterowania panelu")
                error = f"blad petli sterowania: {exc}"
            with self._lock:
                self._loop_error = error
            last = now

    # ------------------------------------------------------------- helpers
    def _set_target(self, joint: str, requested: float) -> TargetResult:
        target = self.limits[joint].clip(requested)
        self._target[joint] = target
        return TargetResult(joint=joint, requested=requested, target=target, clipped=target != requested)

    def _note_clips(self, results: list[TargetResult]) -> None:
        clipped = [result for result in results if result.clipped]
        self._clip = clipped[-1] if clipped else None

    def _hold_at(self, positions: Mapping[str, float]) -> None:
        for name in self.description.joints:
            self._command[name] = positions[name]
            self._target[name] = positions[name]

    def _ramped(self, dt: float) -> dict[str, float]:
        proposed: dict[str, float] = {}
        for name in self.description.joints:
            reach = min(self._speed, self.description.max_vel[name]) * dt
            gap = self._target[name] - self._command[name]
            if abs(gap) <= reach:
                proposed[name] = self._target[name]
            elif gap > 0.0:
                proposed[name] = self._command[name] + reach
            else:
                proposed[name] = self._command[name] - reach
        return proposed

    def _send(self, command: Mapping[str, float]) -> dict[str, float]:
        # The backend clamps to its limits before sending. A joint that starts outside its range (an arm
        # folded at rest) would jump to the limit at servo speed; widened to where the command stands,
        # it is led back by the ramp instead.
        widened: dict[str, tuple[float, float]] = {}
        for name, value in command.items():
            limit = self.limits[name].widened_to(value)
            widened[name] = (limit.low, limit.high)
        self.arm.set_joint_limits(widened)
        return self.arm.send_joints(dict(command))

    def _switch_torque(self, on: bool, measured: Mapping[str, float]) -> None:
        if on:
            # The goal register remembers the previous session: torque enabled before the goal is
            # overwritten would jerk the arm there. Same order as FeetechArm.connect.
            with self._lock:
                self._hold_at(measured)
            if not self._send(measured):
                logger.warning("Cel nie doszedl do serw - moment zostaje wylaczony.")
                return
        value = 1 if on else 0
        for dev_id in self.arm.ids.values():
            self.arm.bus.write(dev_id, ADDR_TORQUE_ENABLE, value, 1)
        with self._lock:
            self._torque = on

    def _note_tick(self) -> None:
        now = self._clock()
        if self._last_tick_at is not None and now > self._last_tick_at:
            period = now - self._last_tick_at
            self._period_s = period if self._period_s == 0.0 else 0.9 * self._period_s + 0.1 * period
        self._last_tick_at = now

    def _heartbeat_lost(self) -> bool:
        return self._clock() - self._last_heartbeat > HEARTBEAT_TIMEOUT_S

    def _all_faults(self) -> list[str]:
        if self._loop_error is None:
            return list(self._faults)
        return [*self._faults, self._loop_error]

    def _require_torque(self) -> None:
        if not self._torque:
            raise PanelError("Moment jest wylaczony - wlacz go, zeby ruszac ramieniem.")

    def _require_joint(self, joint: str) -> None:
        if joint not in self.limits:
            raise PanelError(f"Nieznany staw: {joint}")

    def _gripper(self) -> str:
        if self.description.gripper is None:
            raise PanelError("Ten robot nie ma chwytaka.")
        return self.description.gripper


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PanelError(f"To nie jest liczba: {value!r}") from exc
    if not math.isfinite(number):
        raise PanelError(f"To nie jest liczba: {value!r}")
    return number


# -------------------------------------------------------------------- HTTP
def load_page() -> bytes:
    return resources.files("soarm").joinpath(PAGE).read_bytes()


class PanelServer(ThreadingHTTPServer):
    """The panel's HTTP server: the page and the JSON API of one `PanelController`."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: PanelController) -> None:
        self.controller = controller
        self.page = load_page()
        super().__init__(address, PanelHandler)


class PanelHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", self._panel().page)
        elif path == "/api/state":
            controller = self._panel().controller
            controller.heartbeat()
            self._send_json(HTTPStatus.OK, controller.state())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"nie ma {path}"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"nie ma {path}"})
            return

        # A cross-site page may send a form or text/plain POST without asking; a JSON one needs a CORS
        # preflight this server never grants. So no other website in the browser can move the arm.
        if self.headers.get_content_type() != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "oczekiwano application/json"})
            return

        try:
            body = self._read_json()
            result = run_command(self._panel().controller, path.removeprefix("/api/"), body)
        except UnknownCommand as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        except PanelError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - " + format, self.address_string(), *args)

    def _panel(self) -> PanelServer:
        assert isinstance(self.server, PanelServer)
        return self.server

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise PanelError("Zly naglowek Content-Length.") from exc
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise PanelError(f"Zly JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise PanelError("Oczekiwano obiektu JSON.")
        return body

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def run_command(controller: PanelController, command: str, body: Mapping[str, Any]) -> object:
    """Runs `POST /api/<command>` with its JSON body; the result as plain data for JSON.

    Raises UnknownCommand for a command the panel does not have and PanelError for a refused one.
    """
    if command == "jog":
        return asdict(controller.jog(_text(body, "joint"), _field(body, "delta")))
    if command == "target":
        return asdict(controller.set_target(_text(body, "joint"), _field(body, "value")))
    if command == "open":
        return asdict(controller.open_gripper())
    if command == "close":
        return asdict(controller.close_gripper())
    if command == "home":
        return [asdict(result) for result in controller.home()]
    if command == "stop":
        controller.stop()
        return {}
    if command == "torque":
        controller.set_torque(_flag(body, "on"))
        return {}
    if command == "speed":
        return {"speed": controller.set_speed(_field(body, "value"))}
    raise UnknownCommand(f"Nieznane polecenie: {command}")


def _field(body: Mapping[str, Any], key: str) -> Any:
    if key not in body:
        raise PanelError(f"Brak pola `{key}`.")
    return body[key]


def _text(body: Mapping[str, Any], key: str) -> str:
    value = _field(body, key)
    if not isinstance(value, str):
        raise PanelError(f"Pole `{key}` musi byc tekstem.")
    return value


def _flag(body: Mapping[str, Any], key: str) -> bool:
    value = _field(body, key)
    if not isinstance(value, bool):
        raise PanelError(f"Pole `{key}` musi byc true albo false.")
    return value


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soarm-panel",
        description="Panel testowy ramienia w przegladarce: ruch stawow i chwytaka, Dom, STOP i moment, "
        "bez cyfrowego blizniaka.",
    )
    parser.add_argument(
        "--port", required=True, help="port szeregowy ramienia, np. COM5, /dev/ttyACM0 albo socket://host:5555"
    )
    parser.add_argument(
        "--baudrate", type=int, default=1_000_000, help="predkosc portu (domyslnie 1 Mbaud)"
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:8765",
        help="gdzie nasluchiwac; domyslnie `127.0.0.1:8765`, czyli tylko lokalnie",
    )
    parser.add_argument(
        "--robot", type=Path, default=ROBOT_DIR, help="folder z robot.yaml (domyslnie SO-101 z pakietu)"
    )
    parser.add_argument("--no-browser", action="store_true", help="nie otwieraj przegladarki")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    host, tcp_port = split_listen(args.listen)
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Panel slucha na %s. Kazdy, kto dosiegnie tego portu, moze ruszac ramieniem.", host
        )

    try:
        description = RobotDescription.load(args.robot)
        arm = FeetechArm(description.context(args.port, {"baudrate": args.baudrate}))
        arm.connect()
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 2

    controller = PanelController(arm, description)
    try:
        server = PanelServer((host, tcp_port), controller)
    except OSError as exc:
        logger.error("Nie moge nasluchiwac na %s:%d: %s", host, tcp_port, exc)
        arm.disconnect()
        return 2

    controller.start()
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown_host}:{server.server_port}/"
    logger.info("Panel: %s  (Ctrl-C konczy; ramie zostaje z wlaczonym momentem)", url)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Zatrzymano.")
    finally:
        server.server_close()
        controller.close()
        # FeetechArm.disconnect leaves torque on by default: a limp arm would drop.
        arm.disconnect()
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry
    raise SystemExit(main())
