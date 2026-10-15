"""COM <-> TCP bridge: makes the arm's serial port available to a machine over the network.

What it is for
--------------
The arm is not a "USB device" that has to be passed through whole: it is a
USB-serial adapter, and all that comes out of it is a byte stream. Redirecting
the whole USB device puts the network inside the serial port driver's loop:
every servo transaction then waits for several URB round trips. The bridge
carries only the byte stream over the network, so one transaction is one
round trip, not several.

On the remote machine any virtual serial port driver over TCP will do (HW VSP3,
com0com + com2tcp, socat). The application there sees a plain `COM3` or
`/dev/ttyS10` and does not know that the port is elsewhere. The `feetech` backend
needs no driver: it opens `socket://host:5555` itself.

A limitation to know about
--------------------------
The Feetech protocol is request-reply, and the servo driver has its own reply
timeout (about 34 ms at 1 Mbaud in feetech-servo-sdk). Locally one transaction
costs ~0.3 ms, so the margin is huge; over the network it costs one RTT. On a
link under ~20 ms it works; above that the driver starts reporting read errors.
(The `feetech` backend over `socket://` waits up to 0.25 s.) `--stats` shows
the actual traffic.

Running it
----------
    soarm-bridge --port COM5 --listen 0.0.0.0:5555 --allow 1.2.3.4
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger(__name__)

#: How long to wait for bytes from the port before the thread checks whether to stop.
POLL_S = 0.02
#: Upper bound of a single read, only so as not to allocate without limit.
CHUNK = 4096


class SerialLike(Protocol):
    """As much of `serial.Serial` as the bridge really uses (makes testing easier)."""

    in_waiting: int

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def close(self) -> None: ...


class Counters:
    """Byte counts in both directions, for `--stats` and for the tests."""

    def __init__(self) -> None:
        self.to_net = 0
        self.to_port = 0
        self._lock = threading.Lock()

    def add(self, *, to_net: int = 0, to_port: int = 0) -> None:
        with self._lock:
            self.to_net += to_net
            self.to_port += to_port

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.to_net, self.to_port


def _pump_port_to_net(
    link: SerialLike, conn: socket.socket, stop: threading.Event, counters: Counters
) -> None:
    """Port -> network. Reads one byte, blocking (up to POLL_S), then takes all the rest at once.

    This gives the lowest latency pyserial can deliver: there is no busy
    waiting, and a servo reply goes on the moment it arrives, without an
    artificial collection window.
    """
    while not stop.is_set():
        try:
            data = link.read(1)
            if not data:
                continue
            waiting = getattr(link, "in_waiting", 0)
            if waiting:
                data += link.read(min(waiting, CHUNK))
            conn.sendall(data)
            counters.add(to_net=len(data))
        except OSError as exc:
            if not stop.is_set():
                logger.debug("Koniec strumienia port->siec: %s", exc)
            break
    stop.set()


def _pump_net_to_port(
    link: SerialLike, conn: socket.socket, stop: threading.Event, counters: Counters
) -> None:
    """Network -> port."""
    while not stop.is_set():
        try:
            data = conn.recv(CHUNK)
            if not data:
                break
            link.write(data)
            counters.add(to_port=len(data))
        except OSError as exc:
            if not stop.is_set():
                logger.debug("Koniec strumienia siec->port: %s", exc)
            break
    stop.set()


def handle_client(
    link: SerialLike,
    conn: socket.socket,
    counters: Counters | None = None,
    stop: threading.Event | None = None,
) -> Counters:
    """Passes bytes both ways until the client disconnects. Blocks."""
    counters = counters or Counters()
    stop = stop or threading.Event()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    up = threading.Thread(
        target=_pump_port_to_net,
        args=(link, conn, stop, counters),
        name="port-do-sieci",
        daemon=True,
    )
    up.start()
    try:
        _pump_net_to_port(link, conn, stop, counters)
    finally:
        stop.set()
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        up.join(timeout=1.0)
    return counters


def serve(
    open_link: Callable[[], SerialLike],
    host: str,
    port: int,
    allow: tuple[str, ...] = (),
    stop: threading.Event | None = None,
    stats_every_s: float = 0.0,
    on_ready: Callable[[int], None] | None = None,
) -> None:
    """Listens and serves one client at a time.

    The serial port is opened only when a client connects and closed when it
    disconnects, so the bridge can wait in the background without holding the device.
    """
    stop = stop or threading.Event()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)
    bound = server.getsockname()[1]
    logger.info("Most nasluchuje na %s:%d", host, bound)
    if on_ready is not None:
        on_ready(bound)

    try:
        while not stop.is_set():
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            if allow and addr[0] not in allow:
                logger.warning("Odrzucam polaczenie z %s - spoza listy --allow", addr[0])
                conn.close()
                continue

            logger.info("Klient %s:%d - otwieram port", addr[0], addr[1])
            try:
                link = open_link()
            except Exception:
                logger.exception("Nie udalo sie otworzyc portu szeregowego")
                conn.close()
                continue

            counters = Counters()
            reporter = _start_reporter(counters, stats_every_s) if stats_every_s > 0 else None
            try:
                handle_client(link, conn, counters)
            finally:
                if reporter is not None:
                    reporter.set()
                try:
                    link.close()
                except Exception:  # pragma: no cover - closing must not crash the bridge
                    logger.exception("Blad przy zamykaniu portu")
                conn.close()
                to_net, to_port = counters.snapshot()
                logger.info(
                    "Klient %s rozlaczony (%d B z portu, %d B do portu)", addr[0], to_net, to_port
                )
    finally:
        server.close()


def _start_reporter(counters: Counters, every_s: float) -> threading.Event:
    done = threading.Event()

    def run() -> None:
        last = counters.snapshot()
        while not done.wait(every_s):
            now = counters.snapshot()
            logger.info(
                "ruch: %d B/s z portu, %d B/s do portu",
                int((now[0] - last[0]) / every_s),
                int((now[1] - last[1]) / every_s),
            )
            last = now

    threading.Thread(target=run, name="statystyki", daemon=True).start()
    return done


def split_listen(value: str) -> tuple[str, int]:
    """Parses `--listen`: `5555` (localhost only), `0.0.0.0:5555` or `127.0.0.1:5555`."""
    if ":" in value:
        host, _, port = value.rpartition(":")
        return host or "0.0.0.0", int(port)
    return "127.0.0.1", int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soarm-bridge",
        description="Udostepnia port szeregowy ramienia po TCP (dla wirtualnego portu COM "
        "na maszynie zdalnej).",
    )
    parser.add_argument("--port", required=True, help="port szeregowy ramienia, np. COM5")
    parser.add_argument(
        "--baud", type=int, default=1_000_000, help="predkosc portu (domyslnie 1 Mbaud)"
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:5555",
        help="gdzie nasluchiwac; `5555` to tylko lokalnie, `0.0.0.0:5555` na wszystkich kartach",
    )
    parser.add_argument(
        "--allow",
        default="",
        help="lista adresow IP, ktore moga sie polaczyc (po przecinku); pusta = kazdy",
    )
    parser.add_argument("--stats", type=float, default=0.0, help="co ile sekund wypisywac ruch")
    parser.add_argument("-v", "--verbose", action="store_true", help="szczegolowe logi")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import serial
    except ImportError:
        logger.error("Brak biblioteki pyserial. Zainstaluj:  pip install pyserial")
        return 2

    host, tcp_port = split_listen(args.listen)
    allow = tuple(a.strip() for a in args.allow.split(",") if a.strip())
    if host not in ("127.0.0.1", "localhost") and not allow:
        logger.warning(
            "Most slucha na %s bez --allow. Kazdy, kto dosiegnie tego portu, moze ruszac "
            "ramieniem - ogranicz dostep adresem albo zapora.",
            host,
        )

    def open_link() -> SerialLike:
        # `timeout` must be finite so that the port thread can notice the end of work;
        # `write_timeout` guards against hanging when the board stops receiving.
        link = serial.Serial(args.port, args.baud, timeout=POLL_S, write_timeout=1.0)
        link.reset_input_buffer()
        link.reset_output_buffer()
        return link

    logger.info("Port %s @ %d bd", args.port, args.baud)
    try:
        serve(open_link, host, tcp_port, allow=allow, stats_every_s=args.stats)
    except KeyboardInterrupt:
        logger.info("Zatrzymano.")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry
    raise SystemExit(main())
