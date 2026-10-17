"""Direct control of Feetech STS3215 servos, without LeRobot and without torch.

Why a second backend for the same arm
-------------------------------------
`lerobot_backend` is fully compatible with the LeRobot ecosystem (calibration,
dataset recording, training), but it pulls in `torch` and `torchvision`: several
gigabytes to write six numbers into servo registers. Teleoperation alone does not
need that: the servos speak a protocol that fits on one screen, and all the motion
is one `SYNC WRITE` packet per loop cycle.

The protocol is compatible with Dynamixel 1.0:
    FF FF <ID> <LEN> <INST> [parameters...] <CHECKSUM>
where ``LEN`` is the number of parameters + 2 and the checksum is the bitwise NOT of
the sum of everything after the header. Two-byte values go least significant byte first.

What this backend does NOT do
-----------------------------
There is no LeRobot calibration here, so converting ticks to degrees assumes that a
joint's mechanical zero is at `feetech.center_ticks` (default tick 2048, the middle
of the servo range). That is what the standard SO-101 assembly gives, with the servos
mounted centred. There is no per-joint zero offset or direction setting.
"""

from __future__ import annotations

import logging
import math
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from machinekind_twin_api.backend import BackendContext, RobotBackend, RobotInfo

from .settings import from_mapping

logger = logging.getLogger(__name__)

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

#: The servo's own angle limits, stored in its EEPROM. The servo silently clamps EVERY
#: command to them, so an application that does not know them commands poses the servo
#: cannot reach and has no way to find out.
ADDR_MIN_ANGLE_LIMIT = 9
ADDR_MAX_ANGLE_LIMIT = 11
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

TICKS_PER_REV = 4096
TICK_MIN = 0
TICK_MAX = 4095


#: Below this voltage the servo answers (the board is powered over USB) but has nothing
#: to move with. Better to say so plainly than to let the user hunt for a config error.
MIN_SANE_VOLTAGE = 9.0

#: Bits of the error byte in every servo reply (as ERRBIT_* in scservo_sdk).
#: A servo in protection releases torque but keeps answering; without these bits a joint
#: that dropped looks healthy (reproduced: an overloaded shoulder, 0x20, while the calibration
#: wave carried on with 63 degrees between command and measurement).
SERVO_ERROR_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "napiecie zasilania poza zakresem"),
    (0x02, "blad czujnika kata / kat poza zakresem"),
    (0x04, "przegrzanie"),
    (0x08, "za duzy prad"),
    (0x20, "przeciazenie"),
)

#: This many consecutive failed reads of the arm = a dead link (not one lost frame).
LINK_LOSS_CYCLES = 5

#: Shortest quiet window [s] after which the link is taken as empty after a lost reply.
#: The actual window is ~2x the longest recent round trip (see `FeetechBus._quiet`).
QUIET_MIN_S = 0.02

#: Two position reads at start may differ by this many ticks (~1 degree).
#: That position becomes the servo goal just before torque is enabled; one
#: garbled read would mean the arm jumping at full speed.
START_TOLERANCE_TICKS = 12


def checksum(payload: bytes) -> int:
    """Packet checksum: the bitwise NOT of the sum of everything after the header."""
    return (~sum(payload)) & 0xFF


def describe_servo_error(name: str, dev_id: int, bits: int) -> str:
    """Servo error bits as a sentence for the operator: "shoulder_lift (serwo 2): przeciazenie, ...".

    Shared by both backends (the reply error byte here, the Status register in `lerobot`:
    the same bits), so the panel and the supervisor get the same sentence whichever backend runs.
    """
    what = [text for bit, text in SERVO_ERROR_BITS if bits & bit]
    known = sum(bit for bit, _ in SERVO_ERROR_BITS)
    if bits & ~known:
        what.append(f"nieznany blad 0x{bits & ~known:02X}")
    return f"{name} (serwo {dev_id}): {', '.join(what)}"


class FeetechBus:
    """Protocol layer over the serial port. Kept separate so that it can be tested."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._link: Any = None
        #: Last error byte from a valid reply of each servo {id: bits}.
        self.errors: dict[int, int] = {}
        #: Replies dropped for a bad checksum (diagnostics).
        self.corrupt_replies = 0
        #: How the last receive ended: "ok", "lost" (no frame or a truncated one, the stream
        #: may be misaligned) or "corrupt" (a whole frame with bad content).
        self._last_rx = "ok"
        #: Longest recent round trip of a successful transaction [s] (decays slowly); the quiet
        #: window comes from it.
        self._rtt = 0.0
        #: After a lost reply the stream is suspect: a late reply may still arrive and land in the
        #: next transaction. Cleared only by a clean read.
        self._suspect = False
        #: Reads where a newer reply followed the first one; the newer one was taken (diagnostics).
        self.late_replies = 0

    # ------------------------------------------------------------------ port
    def open(self, link: Any = None) -> None:
        """Opens the port. `link` substitutes a ready object (tests) for pyserial."""
        if link is not None:
            self._link = link
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "Backend `feetech` wymaga biblioteki pyserial. "
                'Zainstaluj:  pip install "soarm[feetech]"'
            ) from exc
        if "://" in self.port:
            # `socket://host:5555`: the `soarm-bridge` on the machine next to the arm.
            # The remote machine then needs no virtual COM port driver. The reply
            # timeout goes up to at least 0.25 s to cover the network round trip.
            self.timeout = max(self.timeout, 0.25)
            self._link = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=self.timeout,
                                               write_timeout=1.0)
            self._disable_nagle()
        else:
            self._link = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=1.0)
        self._link.reset_input_buffer()
        self._link.reset_output_buffer()

    def _disable_nagle(self) -> None:
        """Turns Nagle's algorithm off on the `socket://` socket (pyserial leaves it on).

        SYNC WRITE has no reply, so the bridge has nothing to acknowledge it with right
        away and holds its ACK until the delayed-ACK timer runs out (200 ms on Windows,
        40 ms on Linux). With Nagle on, every following command and read waits for that
        ACK: goals reach the servos in batches ~5 times a second, and reads use up the
        whole 0.25 s timeout. The bridge sets TCP_NODELAY only on its own side, which
        covers the replies, not the commands.
        """
        sock = getattr(self._link, "_socket", None)
        if sock is None:
            return
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # pragma: no cover - depends on the OS
            logger.warning("Nie udalo sie wylaczyc Nagle'a na %s - rozkazy moga isc paczkami.", self.port)

    def close(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None

    @property
    def is_open(self) -> bool:
        return self._link is not None

    # -------------------------------------------------------------- transactions
    def _send(self, dev_id: int, instruction: int, params: bytes = b"") -> None:
        body = bytes([dev_id, len(params) + 2, instruction]) + params
        self._link.reset_input_buffer()
        self._link.write(b"\xff\xff" + body + bytes([checksum(body)]))
        self._link.flush()

    def _receive_frame(self) -> tuple[int, bytes] | None:
        """Reads one reply frame: (servo id, data without the error byte and checksum).

        A frame with a bad checksum is dropped. On the TTL bus between the adapter and
        the servos the checksum is the only data protection (the bridge's TCP covers only
        the network leg). Without it, one flipped bit in the high byte of a position turns
        0 degrees into -180, the twin makes that its first command, and the arm jumps
        105 degrees at full speed.
        """
        self._last_rx = "lost"
        head = self._link.read(4)
        if len(head) < 4 or head[0] != 0xFF or head[1] != 0xFF:
            return None
        dev_id, length = head[2], head[3]
        if length < 2:
            return None                     # not even an error byte and a checksum: not a reply
        rest = self._link.read(length)
        if len(rest) < length:
            return None
        if checksum(bytes(head[2:4]) + bytes(rest[:-1])) != rest[-1]:
            self.corrupt_replies += 1
            self._last_rx = "corrupt"
            logger.debug("Odpowiedz serwa %d z zla suma kontrolna - odrzucam", dev_id)
            return None
        self._last_rx = "ok"
        error = rest[0]
        if error and error != self.errors.get(dev_id, 0):
            logger.warning("Serwo %d zglasza blad 0x%02X", dev_id, error)
        self.errors[dev_id] = error
        return dev_id, bytes(rest[1:-1])

    def _receive(self, dev_id: int) -> bytes | None:
        """Receives the reply of servo `dev_id`: its data only (no header, length, error byte, checksum)."""
        frame = self._receive_frame()
        if frame is None:
            return None
        if frame[0] != dev_id:
            self._last_rx = "lost"          # another servo's frame: not this transaction
            return None
        return frame[1]

    # ------------------------------------------------------------- late replies
    def _pending(self) -> bool:
        """Whether bytes are waiting on the link (pyserial's `socket://` says only 0/1, not how many)."""
        try:
            return bool(getattr(self._link, "in_waiting", 0))
        except Exception:  # pragma: no cover - link closed
            return False

    def _quiet(self) -> float:
        """Quiet window: ~2x the longest recent round trip, at least QUIET_MIN_S, at most the timeout."""
        return min(max(2.0 * self._rtt + 0.005, QUIET_MIN_S), max(self.timeout, QUIET_MIN_S))

    def _note_rtt(self, t0: float) -> None:
        self._rtt = max(time.monotonic() - t0, self._rtt * 0.9)

    def _settle(self, ids: list[int], length: int) -> dict[int, int]:
        """Reads whatever still arrives until the link is quiet for `_quiet()`, then clears the buffer.

        Returns {id: value} from the NEWEST whole frames of servos `ids` (empty = nothing new).
        The protocol does not number transactions, and `_send` clears only what has ALREADY arrived.
        Measured over the bridge: a reply from before a network stall (or a TCP retransmission 0.3 s
        late) arrives during the NEXT read and, without this, is taken as its reply: the twin turns
        the 0.3 s old pose into a command and the servo moves back 6-7 degrees (also after STOP).
        The right reply follows right behind the late one, in TCP order, so the last frame of each
        servo that arrives before the quiet wins.
        """
        quiet = self._quiet()
        last = time.monotonic()
        hard = last + self.timeout + quiet          # continuous noise cannot hold the loop forever
        newer: dict[int, int] = {}
        while True:
            now = time.monotonic()
            if now - last >= quiet or now >= hard:
                break
            if not self._pending():
                time.sleep(0.001)
                continue
            frame = self._receive_frame()
            if frame is None:
                if self._last_rx == "lost":
                    break                           # truncated frame: stream misaligned, drop the buffer
                continue
            dev_id, data = frame
            if dev_id in ids and len(data) >= length:
                newer[dev_id] = int.from_bytes(data[:length], "little")
            last = time.monotonic()
        self._link.reset_input_buffer()
        return newer

    def _finish(self, out: dict[int, int], ids: list[int], length: int, t0: float) -> dict[int, int]:
        """Finishes a read: after a lost reply it drains the link, and it checks the first read after one.

        A read that got a reply while the stream was suspect also waits for quiet: if a newer reply
        followed, the first one was late and only the newer one counts. After a lost reply the link
        is drained (read and discarded until quiet), so that a late frame does not land in the next
        transaction.
        """
        lost = self._last_rx == "lost"
        if out and self._suspect:
            newer = self._settle(ids, length)
            if newer:
                self.late_replies += 1
                logger.debug("Spozniona odpowiedz serw %s - biore nowsza", sorted(out))
                out = newer                         # what the newer reply lacks is stale
            self._suspect = bool(newer) or lost
        elif lost:
            self._settle([], length)
            self._suspect = True
        if out and not lost:
            self._note_rtt(t0)
        return out

    def ping(self, dev_id: int) -> bool:
        self._send(dev_id, INST_PING)
        ok = self._receive(dev_id) is not None
        if not ok and self._last_rx == "lost":
            self._settle([], 0)
            self._suspect = True
        return ok

    def read(self, dev_id: int, addr: int, length: int) -> int | None:
        t0 = time.monotonic()
        self._send(dev_id, INST_READ, bytes([addr, length]))
        data = self._receive(dev_id)
        out = {} if data is None or len(data) < length else {dev_id: int.from_bytes(data[:length], "little")}
        return self._finish(out, [dev_id], length, t0).get(dev_id)

    def write(self, dev_id: int, addr: int, value: int, length: int) -> None:
        self._send(dev_id, INST_WRITE, bytes([addr]) + value.to_bytes(length, "little"))
        if self._receive(dev_id) is None and self._last_rx == "lost":
            self._settle([], 0)
            self._suspect = True

    def sync_read(self, addr: int, length: int, ids: list[int]) -> dict[int, int]:
        """One packet, each servo replies in turn: {id: value} for those that replied.

        Over the network bridge this is one round trip instead of six: at 20 ms RTT
        a read of the arm costs 20 ms, not 120.
        """
        params = bytes([addr, length, *ids])
        t0 = time.monotonic()
        self._send(BROADCAST_ID, INST_SYNC_READ, params)
        out: dict[int, int] = {}
        # Servos reply in turn, but a silent one must not take the rest with it:
        # frames are matched by ID, not by their position in the stream.
        # Otherwise one deaf servo 1 would look like a dead link.
        for _ in ids:
            frame = self._receive_frame()
            if frame is None:
                if self._last_rx == "lost":
                    break                   # silence or a truncated frame: nothing after it lines up
                continue                    # whole frame read, stream aligned: the next ones are good
            dev_id, data = frame
            if dev_id in ids and dev_id not in out and len(data) >= length:
                out[dev_id] = int.from_bytes(data[:length], "little")
            if dev_id == ids[-1]:
                break                       # the last servo replied: do not wait for the missing ones
        return self._finish(out, list(ids), length, t0)

    def sync_write(self, addr: int, length: int, values: dict[int, int]) -> None:
        """One packet for all servos at once, with no reply, so no waiting.

        This is why the control loop does not pay per joint:
        six goals cost as much as one.
        """
        if not values:
            return
        params = bytearray([addr, length])
        for dev_id, value in sorted(values.items()):
            params.append(dev_id)
            params += value.to_bytes(length, "little")
        body = bytes([BROADCAST_ID, len(params) + 2, INST_SYNC_WRITE]) + bytes(params)
        self._link.write(b"\xff\xff" + body + bytes([checksum(body)]))
        self._link.flush()


@dataclass
class FeetechSettings:
    """The `feetech:` section of the twin's config."""

    #: Port speed. The STS3215 servos in the SO-101 ship set to 1 Mbaud.
    baudrate: int = 1_000_000
    #: Tick that corresponds to joint zero. The servo has 4096 ticks per turn, and the standard
    #: SO-101 assembly mounts it centred, which is 2048.
    center_ticks: int = 2048
    #: Gripper jaw range in ticks: 0 in app scale is `closed`, 100 is `open`. The values come
    #: from the limits the gripper servo had stored in its own EEPROM, i.e. from the jaw travel
    #: measured on a real arm. Check yours: open the jaw by hand and read `Present_Position`.
    #: The robot description's `gripper_range_q` must match them, or the twin warns at connect.
    gripper_closed_ticks: int = 1986
    gripper_open_ticks: int = 2670
    #: Torque off on exit. Off by default, because a limp arm drops under its own weight;
    #: enable it only when you know the arm is supported.
    torque_off_on_exit: bool = False


def create(ctx: BackendContext) -> FeetechArm:
    """Entry point of the `feetech` backend plugin."""
    return FeetechArm(ctx)


class FeetechArm(RobotBackend):
    """SO-101 arm driven directly over the serial port."""

    def __init__(self, ctx: BackendContext, bus: FeetechBus | None = None):
        self.settings: FeetechSettings = from_mapping(FeetechSettings, ctx.settings, "feetech")
        self.port = ctx.port
        if not self.port:
            raise ValueError(
                "Nie podano portu szeregowego ramienia. Uzyj --port COM5 (Windows) "
                "albo --port /dev/ttyACM0 (Linux)."
            )
        self.joints = tuple(ctx.joints)
        self.gripper = ctx.gripper
        self.limits: dict[str, tuple[float, float]] = dict(ctx.limits)
        self.bus = bus or FeetechBus(self.port, self.settings.baudrate)
        # The SO-101 numbers its servos from the base to the gripper, in joint order.
        self.ids = {name: index for index, name in enumerate(self.joints, start=1)}
        self._connected = False
        self._positions: dict[str, float] = {}
        #: Limits read from the servos [ticks]. No entry = the servo does not limit.
        self.servo_limits: dict[str, tuple[int, int]] = {}
        self._sync_read_ok = True
        #: SYNC READ has answered at least once this session: silence on it is then the
        #: link, not old firmware, and it must not be switched off for good.
        self._sync_ever_ok = False
        #: Cycles in which the servos answered single reads but not SYNC READ
        #: (counted only until SYNC READ first works).
        self._sync_misses = 0
        #: Consecutive read cycles in which at least one joint got no fresh position.
        self._failed_cycles = 0
        self._last_full_read = 0.0
        self._stale: list[str] = []
        #: The last read cycle got NOT ONE reply: the link is down.
        self._link_silent = False
        self.info = RobotInfo(
            name="SO-101 (feetech)",
            description=f"port {self.port} @ {self.settings.baudrate} bd",
            simulated=False,
        )

    # ------------------------------------------------------------- conversions
    def _to_ticks(self, name: str, value: float) -> int:
        rc = self.settings
        if name == self.gripper:
            # The gripper moves in 0..100 (0 = closed), not in degrees.
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            ticks = rc.gripper_closed_ticks + span * (value / 100.0)
        else:
            ticks = rc.center_ticks + value * TICKS_PER_REV / 360.0
        return int(round(min(max(ticks, TICK_MIN), TICK_MAX)))

    def _to_units(self, name: str, ticks: int) -> float:
        rc = self.settings
        if name == self.gripper:
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            if abs(span) < 1e-6:  # pragma: no cover - a config with no range
                return 0.0
            return (ticks - rc.gripper_closed_ticks) * 100.0 / span
        return (ticks - rc.center_ticks) * 360.0 / TICKS_PER_REV

    # ---------------------------------------------------------------- connection
    def connect(self) -> None:
        # The bus may already be open: the tests hand it in opened on a fake link.
        # Opening it again would break it.
        if not self.bus.is_open:
            self.bus.open()
        # A new session: the previous session's errors and counters do not apply.
        self.bus.errors.clear()
        self._sync_read_ok, self._sync_ever_ok, self._sync_misses = True, False, 0

        missing =[name for name, dev_id in self.ids.items() if not self.bus.ping(dev_id)]
        if missing:
            self.bus.close()
            raise RuntimeError(
                f"Serwa nie odpowiadaja: {', '.join(missing)}. "
                "Sprawdz zasilanie ramienia i port szeregowy."
            )

        self._check_power()
        self._read_servo_limits()
        ticks = self._read_start_positions()

        # ORDER MATTERS. The goal register keeps its value from the previous
        # session, so enabling torque alone would jerk the arm to that pose.
        # First set the goal to where the arm actually stands.
        self.bus.sync_write(
            ADDR_GOAL_POSITION, 2, {self.ids[name]: int(value) for name, value in ticks.items()}
        )
        for dev_id in self.ids.values():
            self.bus.write(dev_id, ADDR_TORQUE_ENABLE, 1, 1)

        self._positions = {name: self._to_units(name, int(value)) for name, value in ticks.items()}
        self._failed_cycles, self._stale, self._link_silent = 0, [], False
        self._last_full_read = time.monotonic()
        self._connected = True
        logger.info(
            "Polaczono z ramieniem na %s. Pozycja startowa: %s",
            self.port,
            ", ".join(f"{name}={value:.1f}" for name, value in self._positions.items()),
        )

    def _read_start_positions(self) -> dict[str, int]:
        """Start position from TWO matching reads of each joint.

        This position becomes the servo goal just before torque is enabled, so
        one garbled read (an 8-bit checksum does not catch every disturbance)
        would mean the arm jumping at full speed. Two independent reads that
        agree practically rule that out.
        """
        problem = "Nie udalo sie odczytac pozycji wszystkich stawow."
        for _ in range(3):
            first = {name: self.bus.read(dev_id, ADDR_PRESENT_POSITION, 2) for name, dev_id in self.ids.items()}
            second = {name: self.bus.read(dev_id, ADDR_PRESENT_POSITION, 2) for name, dev_id in self.ids.items()}
            if any(v is None for v in (*first.values(), *second.values())):
                problem = "Nie udalo sie odczytac pozycji wszystkich stawow."
                continue
            moving = [name for name in self.ids if abs(first[name] - second[name]) > START_TOLERANCE_TICKS]
            if not moving:
                return {name: int(value) for name, value in second.items()}
            problem = (f"Kolejne odczyty pozycji sie nie zgadzaja ({', '.join(moving)}) - ramie sie "
                       "rusza albo magistrala przeklamuje. Unieruchom ramie i polacz ponownie.")
        self.bus.close()
        raise RuntimeError(problem)

    def _read_servo_limits(self) -> None:
        """Reads the servos' own limits and warns loudly when they are tighter than the config.

        Without this the application commands poses the servo will not reach: it
        silently clamps them to its range and ends up somewhere else entirely. That
        shows up as "the joint ignores the command", although the command arrived intact.
        """
        self.servo_limits = {}
        conflicts: list[str] = []
        for name, dev_id in self.ids.items():
            low = self.bus.read(dev_id, ADDR_MIN_ANGLE_LIMIT, 2)
            high = self.bus.read(dev_id, ADDR_MAX_ANGLE_LIMIT, 2)
            if low is None or high is None or low >= high:
                continue
            if low <= TICK_MIN and high >= TICK_MAX:
                continue  # full range = the servo limits nothing
            self.servo_limits[name] = (low, high)

            lo, hi = self.limits[name]
            servo_lo, servo_hi = self._to_units(name, low), self._to_units(name, high)
            if servo_lo > lo + 1.0 or servo_hi < hi - 1.0:
                conflicts.append(
                    f"{name}: serwo {servo_lo:.0f}..{servo_hi:.0f}, "
                    f"konfiguracja {lo:.0f}..{hi:.0f}"
                )

        if conflicts:
            logger.warning(
                "Serwa maja WLASNE limity kata, ciasniejsze niz `joints` w konfiguracji. "
                "Rozkazy poza nimi zostana przyciete przez serwo, a staw zatrzyma sie "
                "wczesniej, niz aplikacja zaklada:\n  %s",
                "\n  ".join(conflicts),
            )

    def _check_power(self) -> None:
        """Warns when the arm answers but has nothing to move with."""
        raw = self.bus.read(self.ids[self.joints[0]], ADDR_PRESENT_VOLTAGE, 1)
        if raw is None:
            return
        volts = raw / 10.0
        if volts < MIN_SANE_VOLTAGE:
            logger.warning(
                "Napiecie zasilania %.1f V - serwa odpowiadaja, ale przy tym napieciu "
                "nie ruszy. Podlacz zasilacz ramienia.",
                volts,
            )
        else:
            logger.info("Zasilanie ramienia: %.1f V", volts)

    def disconnect(self) -> None:
        if not self._connected:
            self.bus.close()
            return
        # Torque stays on unless `torque_off_on_exit` is set: a limp arm would drop
        # under its own weight. By default the arm stays where it stopped.
        if self.settings.torque_off_on_exit:
            for dev_id in self.ids.values():
                try:
                    self.bus.write(dev_id, ADDR_TORQUE_ENABLE, 0, 1)
                except Exception:  # pragma: no cover - disconnecting must not crash the application
                    logger.exception("Nie udalo sie wylaczyc momentu serwa %d", dev_id)
        self._connected = False
        self.bus.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ I/O
    def read_joints(self) -> dict[str, float]:
        """Joint positions; a joint without a fresh reply keeps its previous value.

        Does not raise on lost frames: one lost frame must not fake a joint jump
        or stop the loop. It counts CONSECUTIVE failed cycles instead, and
        `faults()` reports loudly when the link is down.
        """
        if not self._connected:
            raise RuntimeError("Ramie nie jest polaczone")
        fresh: dict[str, int] = {}
        sync_silent = False
        if self._sync_read_ok:
            got = self.bus.sync_read(ADDR_PRESENT_POSITION, 2, list(self.ids.values()))
            fresh = {name: got[dev_id] for name, dev_id in self.ids.items() if dev_id in got}
            sync_silent = not fresh
            if fresh:
                # One reply means the firmware knows SYNC READ; a missing rest is the link or a servo.
                self._sync_ever_ok, self._sync_misses = True, 0

        missing = [name for name in self.ids if name not in fresh]
        # Silence on a SYNC READ that has already worked this session is a stalled
        # link, not the firmware: single reads, with a 0.25 s timeout each over the
        # bridge, would add 0.5 s or more to the cycle and get nothing. Skip them then.
        if missing and not (sync_silent and self._sync_ever_ok):
            failures = 0
            for name in missing:
                ticks = self.bus.read(self.ids[name], ADDR_PRESENT_POSITION, 2)
                if ticks is None:
                    failures += 1
                    if failures >= 2:
                        break                   # second silence in the cycle: stop paying for timeouts
                    continue
                fresh[name] = ticks
            if sync_silent and not self._sync_ever_ok and fresh:
                # The servos answer single reads but not SYNC READ. This counts only until
                # SYNC READ first works (`_sync_ever_ok`), so network hiccups later in the
                # session cannot switch it off for good and drop the loop over the bridge to ~7 Hz.
                self._sync_misses += 1
                if self._sync_misses >= 3:
                    # Older firmware versions do not know SYNC READ: stay with single reads.
                    self._sync_read_ok = False
                    logger.info("Serwa nie odpowiadaja na SYNC READ - odczyt po kolei.")

        for name, ticks in fresh.items():
            self._positions[name] = self._to_units(name, ticks)
        self._stale = [name for name in self.ids if name not in fresh]
        self._link_silent = not fresh
        if self._stale:
            self._failed_cycles += 1
            if self._failed_cycles == LINK_LOSS_CYCLES:
                logger.warning("Serwa nie odpowiadaja (%s) od %d cykli odczytu.",
                               ", ".join(self._stale), self._failed_cycles)
        else:
            self._failed_cycles = 0
            self._last_full_read = time.monotonic()
        return dict(self._positions)

    def faults(self) -> list[str]:
        """Error bits from the servos' latest replies and link loss, as sentences for the operator."""
        try:
            out: list[str] = []
            for name, dev_id in self.ids.items():
                bits = self.bus.errors.get(dev_id, 0)
                if bits:
                    out.append(describe_servo_error(name, dev_id, bits))
            if self._connected and self._failed_cycles >= LINK_LOSS_CYCLES:
                ms = (time.monotonic() - self._last_full_read) * 1000.0
                which = "" if len(self._stale) == len(self.ids) else f" ({', '.join(self._stale)})"
                out.append(f"brak odpowiedzi serw od {ms:.0f} ms{which}")
            return out
        except Exception as exc:  # pragma: no cover - called every loop cycle, must not crash it
            logger.exception("Nie udalo sie sprawdzic stanu serw")
            return [f"nie udalo sie sprawdzic stanu serw: {exc}"]

    def gripper_range_q(self) -> tuple[float, float]:
        """(closed, open) jaw angle [rad from joint zero]: exactly the ticks `_to_ticks`/`_to_units` use."""
        rc = self.settings
        rad = 2.0 * math.pi / TICKS_PER_REV
        return ((rc.gripper_closed_ticks - rc.center_ticks) * rad,
                (rc.gripper_open_ticks - rc.center_ticks) * rad)

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        for name, (lo, hi) in limits.items():
            if name in self.limits:
                self.limits[name] = (float(lo), float(hi))

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """The servos' EEPROM angle limits in app units (only the joints that have them)."""
        out: dict[str, tuple[float, float]] = {}
        for name, (low, high) in self.servo_limits.items():
            a, b = self._to_units(name, low), self._to_units(name, high)
            out[name] = (min(a, b), max(a, b))
        return out

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Ramie nie jest polaczone")
        if self._link_silent:
            # The last read got NOT ONE reply: the link is down. Over the `socket://`
            # bridge a write would "succeed" anyway (it lands in the TCP buffer), and
            # when the network comes back the whole queue of goals reaches the servos at
            # once and the arm jumps to the last one at full speed, even after a STOP
            # pressed during the outage. Nothing is sent until a read confirms that the
            # servos hear again.
            return {}
        sent: dict[str, float] = {}
        ticks: dict[int, int] = {}
        for name in self.joints:
            if name not in targets:
                continue
            lo, hi = self.limits[name]
            value = min(max(float(targets[name]), lo), hi)
            tick = self._to_ticks(name, value)

            # The servo clamps the command to its range anyway. Doing it here lets
            # the caller see the value that ACTUALLY went out; otherwise the rest of
            # the application (speed limits, HUD, 3D view) would work from a position
            # the arm never reached.
            limits = self.servo_limits.get(name)
            if limits is not None:
                tick = min(max(tick, limits[0]), limits[1])
                value = self._to_units(name, tick)

            ticks[self.ids[name]] = tick
            sent[name] = value
        self.bus.sync_write(ADDR_GOAL_POSITION, 2, ticks)
        return sent
