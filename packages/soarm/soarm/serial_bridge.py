"""Most COM <-> TCP: udostepnia port szeregowy ramienia maszynie po sieci.

Po co to jest
-------------
Ramie nie jest "urzadzeniem USB", ktore trzeba przepuscic w calosci - jest
przejsciowka USB-serial, a jedyne, co z niej wychodzi, to strumien bajtow.
Przekierowanie calego USB wklada siec do srodka petli sterownika portu
szeregowego: kazda transakcja z serwem czeka wtedy na kilka przebiegow URB
tam i z powrotem. Most przenosi przez siec sam strumien bajtow, wiec jedna
transakcja to jeden przebieg, a nie kilka.

Po stronie maszyny zdalnej wystarczy dowolny sterownik wirtualnego portu
szeregowego po TCP (HW VSP3, com0com + com2tcp, socat). Aplikacja widzi tam
zwykly `COM3` czy `/dev/ttyS10` i nie wie, ze port jest gdzie indziej.

Ograniczenie, ktore trzeba znac
-------------------------------
Protokol Feetech to pytanie-odpowiedz, a sterownik ma wlasny limit czasu na
odpowiedz (w feetech-servo-sdk okolo 34 ms przy 1 Mbaud). Lokalnie jedna
transakcja kosztuje ~0,3 ms, wiec zapas jest ogromny; przez siec kosztuje
tyle, ile wynosi RTT. Przy laczu ponizej ~20 ms to dziala, powyzej sterownik
zaczyna zglaszac bledy odczytu. `--stats` pokazuje, ile faktycznie schodzi.

Uruchomienie
------------
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

#: Ile czekac na bajty z portu, zanim watek sprawdzi, czy ma sie zatrzymac.
POLL_S = 0.02
#: Gorny limit jednorazowego odczytu - tylko po to, zeby nie alokowac bez konca.
CHUNK = 4096


class SerialLike(Protocol):
    """Tyle z `serial.Serial`, ile most naprawde uzywa (ulatwia testy)."""

    in_waiting: int

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def close(self) -> None: ...


class Counters:
    """Licznik bajtow w obie strony - do `--stats` i do testow."""

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
    """Port -> siec. Czyta blokujaco jeden bajt, potem zabiera cala reszte naraz.

    Ten uklad daje najmniejsze opoznienie, jakie da sie wycisnac z pyserial:
    nie ma aktywnego czekania, a odpowiedz serwa idzie dalej w tej samej
    chwili, w ktorej przyszla - bez doklejania sztucznego okna zbierania.
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
    """Siec -> port."""
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
    """Przepuszcza bajty w obie strony az do rozlaczenia. Blokuje."""
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
    """Nasluchuje i obsluguje po jednym kliencie naraz.

    Port szeregowy otwieramy dopiero na polaczenie i zamykamy przy rozlaczeniu -
    dzieki temu most moze czekac w tle, nie trzymajac urzadzenia zajetego.
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
                except Exception:  # pragma: no cover - zamkniecie nie moze wysypac mostu
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
    """`5555`, `0.0.0.0:5555` albo `127.0.0.1:5555`."""
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
        # `timeout` musi byc skonczony, zeby watek portu mogl zauwazyc koniec pracy;
        # `write_timeout` chroni przed zawieszeniem, gdy plytka przestanie odbierac.
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


if __name__ == "__main__":  # pragma: no cover - wejscie z linii polecen
    raise SystemExit(main())
