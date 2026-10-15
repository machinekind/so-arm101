"""COM <-> TCP bridge: the access list cuts foreign addresses off from the servo bus."""

from __future__ import annotations

import socket
import threading
from collections import deque

from soarm.serial_bridge import serve

TIMEOUT = 5.0


class LoopbackLink:
    """Stands in for a serial port: a read returns what a write put in.

    Enough to check that the bridge passes bytes both ways, and it also mirrors
    the shape of an exchange with a servo (request -> reply).
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
            # A bounded wait, as in a real port with `timeout=`; otherwise the
            # bridge thread would have no way to notice that it has to stop.
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
    """A bridge started on a free port, ready for a test to connect to."""

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
    """A read that ends with the connection closed, whatever the OS.

    Windows answers closing a socket with unread data in its buffer with an
    RST, not a clean FIN, sometimes as 10054 (reset), sometimes as 10053
    (abort), depending on what got sent. For the test it is the same result:
    there is no conversation.
    """
    try:
        return conn.recv(16)
    except OSError:
        return b""


def test_bridge_rejects_addresses_outside_allow():
    """The bridge moves a physical arm, so the access list must really cut off."""
    with RunningBridge(allow=("10.11.12.13",)) as bridge:
        with bridge.connect() as conn:
            conn.sendall(b"x")
            assert read_until_closed(conn) == b""
        assert bridge.links == []
