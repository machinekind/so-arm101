"""Most COM <-> TCP: lista dostepu odcina obce adresy od magistrali serw."""

from __future__ import annotations

import socket
import threading
from collections import deque

from soarm.serial_bridge import serve

TIMEOUT = 5.0


class LoopbackLink:
    """Udaje port szeregowy: oddaje przy odczycie to, co dostal przy zapisie.

    Wystarczy do sprawdzenia, ze most przepuszcza bajty w obie strony - a przy
    okazji odwzorowuje ksztalt wymiany z serwem (pytanie -> odpowiedz).
    """

    def __init__(self) -> None:
        self._buf: deque[int] = deque()
        self._cv = threading.Condition()
        self.closed = False

    @property
    def in_waiting(self) -> int:
        with self._cv:
            return len(self._buf)

    def read(self, size: int = 1) -> bytes:
        with self._cv:
            # Skonczone czekanie, jak w prawdziwym porcie z `timeout=` - inaczej
            # watek mostu nie mialby jak zauwazyc, ze ma sie zatrzymac.
            self._cv.wait_for(lambda: self._buf or self.closed, timeout=0.05)
            out = bytes(self._buf.popleft() for _ in range(min(size, len(self._buf))))
        return out

    def write(self, data: bytes) -> int:
        with self._cv:
            self._buf.extend(data)
            self._cv.notify_all()
        return len(data)

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class RunningBridge:
    """Most wystartowany na wolnym porcie, gotowy do polaczenia w tescie."""

    def __init__(self, **kwargs: object) -> None:
        self.links: list[LoopbackLink] = []
        self.stop = threading.Event()
        self._ready = threading.Event()
        self.port = 0

        def open_link() -> LoopbackLink:
            link = LoopbackLink()
            self.links.append(link)
            return link

        def on_ready(port: int) -> None:
            self.port = port
            self._ready.set()

        self._thread = threading.Thread(
            target=serve,
            args=(open_link, "127.0.0.1", 0),
            kwargs={"stop": self.stop, "on_ready": on_ready, **kwargs},
            daemon=True,
        )

    def __enter__(self) -> RunningBridge:
        self._thread.start()
        assert self._ready.wait(TIMEOUT), "most nie wystartowal"
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self._thread.join(timeout=TIMEOUT)

    def connect(self) -> socket.socket:
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=TIMEOUT)
        conn.settimeout(TIMEOUT)
        return conn


def read_until_closed(conn: socket.socket) -> bytes:
    """Odczyt konczacy sie zamknieciem polaczenia, niezaleznie od systemu.

    Zamkniecie gniazda z nieodczytanymi danymi w buforze Windows kwituje
    RST-em, a nie czystym FIN - i to raz jako 10054 (reset), raz jako 10053
    (abort), zaleznie od tego, co zdazylo sie wyslac. Dla testu to ten sam
    wynik: rozmowy nie ma.
    """
    try:
        return conn.recv(16)
    except OSError:
        return b""


def test_bridge_rejects_addresses_outside_allow():
    """Most rusza fizycznym ramieniem, wiec lista dostepu ma faktycznie odcinac."""
    with RunningBridge(allow=("10.11.12.13",)) as bridge:
        with bridge.connect() as conn:
            conn.sendall(b"x")
            assert read_until_closed(conn) == b""
        assert bridge.links == []
