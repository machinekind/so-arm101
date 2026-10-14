"""Bezposrednie sterowanie serwami Feetech STS3215 - bez LeRobot i bez torcha.

Po co drugi backend do tego samego ramienia
-------------------------------------------
`lerobot_backend` daje pelna zgodnosc z ekosystemem LeRobot (kalibracja,
nagrywanie zbiorow, uczenie), ale ciagnie za soba `torch` i `torchvision` -
kilka gigabajtow na to, zeby wpisac szesc liczb do rejestrow serw. Do samej
teleoperacji to nie jest potrzebne: serwa gadaja protokolem, ktory miesci sie
na jednym ekranie, a caly ruch to jeden pakiet `SYNC WRITE` na cykl petli.

Protokol jest zgodny z Dynamixel 1.0:
    FF FF <ID> <LEN> <INST> [parametry...] <SUMA>
gdzie ``LEN`` to liczba parametrow + 2, a suma kontrolna to negacja sumy
wszystkiego po naglowku. Wartosci dwubajtowe ida najmlodszym bajtem naprzod.

Czego ten backend NIE robi
--------------------------
Nie ma tu kalibracji LeRobota, wiec przeliczenie tikow na stopnie zaklada, ze
mechaniczne zero stawu pokrywa sie ze srodkiem zakresu serwa (tik 2048) - tak
wychodzi przy standardowym montazu SO-101, w ktorym serwa skladasz wysrodkowane.
Jesli ktorys staw ma zero przesuniete, popraw go przez `joints.<nazwa>.offset`,
a zwrot przez `joints.<nazwa>.invert`.
"""

from __future__ import annotations

import logging
import math
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lerobot_mp_api.backend import BackendContext, RobotBackend, RobotInfo

from .settings import from_mapping

logger = logging.getLogger(__name__)

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

#: Wlasne limity kata serwa, zapisane w jego EEPROM-ie. Serwo przycina do nich
#: KAZDY rozkaz po cichu, wiec aplikacja, ktora ich nie zna, zadaje pozy
#: niewykonalne i nie ma jak sie o tym dowiedziec.
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


#: Ponizej tego napiecia serwo odpowiada (plytka zasila sie z USB), ale nie ma
#: z czego ruszyc. Warto powiedziec to wprost, zamiast pozwolic uzytkownikowi
#: szukac bledu w konfiguracji.
MIN_SANE_VOLTAGE = 9.0

#: Bity bajtu bledu w kazdej odpowiedzi serwa (jak ERRBIT_* w scservo_sdk).
#: Serwo w ochronie zwalnia moment, ale dalej odpowiada - bez tych bitow staw,
#: ktory opadl, wyglada na zdrowy (odtworzone: przeciazony bark 0x20, a fala
#: kalibracyjna jechala dalej z 63 stopniami roznicy rozkaz-pomiar).
SERVO_ERROR_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "napiecie zasilania poza zakresem"),
    (0x02, "blad czujnika kata / kat poza zakresem"),
    (0x04, "przegrzanie"),
    (0x08, "za duzy prad"),
    (0x20, "przeciazenie"),
)

#: Tyle kolejnych nieudanych odczytow ramienia = lacze zerwane (a nie jedna zgubiona ramka).
LINK_LOSS_CYCLES = 5

#: Najkrotsze okno ciszy [s], po ktorym lacze po zgubionej odpowiedzi uznajemy za puste.
#: Wlasciwe okno to ~2x najdluzszy ostatni przebieg tam i z powrotem (patrz `FeetechBus._quiet`).
QUIET_MIN_S = 0.02

#: O tyle tikow (~1 stopien) moga sie roznic dwa odczyty pozycji przy starcie.
#: Z tej pozycji robi sie cel serwa tuz przed zalaczeniem momentu - jeden
#: przeklamany odczyt oznaczalby skok ramienia z pelna predkoscia.
START_TOLERANCE_TICKS = 12


def checksum(payload: bytes) -> int:
    """Suma kontrolna pakietu: negacja sumy wszystkiego po naglowku."""
    return (~sum(payload)) & 0xFF


def describe_servo_error(name: str, dev_id: int, bits: int) -> str:
    """Bity bledu serwa jako zdanie dla operatora: "staw (serwo N): przeciazenie, ...".

    Wspolne dla obu backendow (bajt bledu odpowiedzi tu, rejestr Status w `lerobot`
    - te same bity), zeby panel i nadzor dostawaly to samo zdanie niezaleznie od backendu.
    """
    what = [text for bit, text in SERVO_ERROR_BITS if bits & bit]
    known = sum(bit for bit, _ in SERVO_ERROR_BITS)
    if bits & ~known:
        what.append(f"nieznany blad 0x{bits & ~known:02X}")
    return f"{name} (serwo {dev_id}): {', '.join(what)}"


class FeetechBus:
    """Warstwa protokolu na porcie szeregowym. Oddzielona, zeby dala sie testowac."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._link: Any = None
        #: Ostatni bajt bledu z poprawnej odpowiedzi kazdego serwa {id: bity}.
        self.errors: dict[int, int] = {}
        #: Odpowiedzi odrzucone przez zla sume kontrolna (diagnostyka).
        self.corrupt_replies = 0
        #: Jak skonczyl sie ostatni `_receive`: "ok", "lost" (brak/ucieta ramka,
        #: strumien moze byc przesuniety) albo "corrupt" (cala ramka, zla tresc).
        self._last_rx = "ok"
        #: Najdluzszy ostatni przebieg udanej transakcji [s] (maleje powoli) - z niego okno ciszy.
        self._rtt = 0.0
        #: Po zgubionej odpowiedzi strumien jest podejrzany: spozniona odpowiedz moze jeszcze
        #: przyjsc i trafic do nastepnej transakcji. Kasowane dopiero przez czysty odczyt.
        self._suspect = False
        #: Odczyty, w ktorych po odpowiedzi przyszla nowsza - wzieta ta nowsza (diagnostyka).
        self.late_replies = 0

    # ------------------------------------------------------------------ port
    def open(self, link: Any = None) -> None:
        """Otwiera port. `link` podstawia gotowy obiekt (testy) zamiast pyserial."""
        if link is not None:
            self._link = link
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - zalezne od srodowiska
            raise RuntimeError(
                "Backend `feetech` wymaga biblioteki pyserial. "
                'Zainstaluj:  pip install "soarm[feetech]"'
            ) from exc
        if "://" in self.port:
            # `socket://adres:5555` - most `soarm-bridge` na maszynie przy ramieniu.
            # Maszyna zdalna nie potrzebuje wtedy sterownika wirtualnego
            # portu COM. Limit odpowiedzi rosnie o czas przebiegu przez siec.
            self.timeout = max(self.timeout, 0.25)
            self._link = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=self.timeout,
                                               write_timeout=1.0)
            self._disable_nagle()
        else:
            self._link = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=1.0)
        self._link.reset_input_buffer()
        self._link.reset_output_buffer()

    def _disable_nagle(self) -> None:
        """Wylacza algorytm Nagle'a na gniezdzie `socket://` (pyserial go nie rusza).

        SYNC WRITE nie ma odpowiedzi, wiec most nie ma czym potwierdzic go od razu
        i czeka ze swoim ACK do konca opoznienia (200 ms na Windowsie, 40 ms na
        Linuksie). Z wlaczonym Nagle'em kazdy nastepny rozkaz i odczyt czeka na
        ten ACK - cele dochodza do serw paczkami ~5 razy na sekunde, a odczyty
        zjadaja caly limit 0,25 s. Most ustawia TCP_NODELAY tylko po swojej
        stronie, a to dotyczy odpowiedzi, nie rozkazow.
        """
        sock = getattr(self._link, "_socket", None)
        if sock is None:
            return
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # pragma: no cover - zalezne od systemu
            logger.warning("Nie udalo sie wylaczyc Nagle'a na %s - rozkazy moga isc paczkami.", self.port)

    def close(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None

    @property
    def is_open(self) -> bool:
        return self._link is not None

    # -------------------------------------------------------------- transakcje
    def _send(self, dev_id: int, instruction: int, params: bytes = b"") -> None:
        body = bytes([dev_id, len(params) + 2, instruction]) + params
        self._link.reset_input_buffer()
        self._link.write(b"\xff\xff" + body + bytes([checksum(body)]))
        self._link.flush()

    def _receive_frame(self) -> tuple[int, bytes] | None:
        """Czyta jedna ramke odpowiedzi: (id serwa, dane bez bajtu bledu i sumy).

        Ramka z zla suma kontrolna jest traktowana jak zgubiona. Na magistrali
        TTL miedzy adapterem a serwami to jedyna ochrona danych (TCP mostu
        chroni tylko odcinek sieciowy) - bez niej przeklamany bit w starszym
        bajcie pozycji dawal -180 stopni zamiast 0, a twin robil z tego
        pierwszy rozkaz i ramie skakalo o 105 stopni z pelna predkoscia.
        """
        self._last_rx = "lost"
        head = self._link.read(4)
        if len(head) < 4 or head[0] != 0xFF or head[1] != 0xFF:
            return None
        dev_id, length = head[2], head[3]
        if length < 2:
            return None                     # nie ma nawet bajtu bledu i sumy - to nie jest odpowiedz
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
        """Odbiera odpowiedz serwa `dev_id` i zwraca same dane (bez naglowka, dlugosci i sumy)."""
        frame = self._receive_frame()
        if frame is None:
            return None
        if frame[0] != dev_id:
            self._last_rx = "lost"          # cudza ramka - nie ta transakcja
            return None
        return frame[1]

    # ------------------------------------------------- spoznione odpowiedzi
    def _pending(self) -> bool:
        """Czy w laczu czekaja bajty (`socket://` pyserial mowi tylko 0/1, a nie ile)."""
        try:
            return bool(getattr(self._link, "in_waiting", 0))
        except Exception:  # pragma: no cover - zamkniete lacze
            return False

    def _quiet(self) -> float:
        """Okno ciszy: ~2x najdluzszy ostatni przebieg, nie krocej niz QUIET_MIN_S, nie dluzej niz timeout."""
        return min(max(2.0 * self._rtt + 0.005, QUIET_MIN_S), max(self.timeout, QUIET_MIN_S))

    def _note_rtt(self, t0: float) -> None:
        self._rtt = max(time.monotonic() - t0, self._rtt * 0.9)

    def _settle(self, ids: list[int], length: int) -> dict[int, int]:
        """Czyta, co jeszcze przyjdzie, az lacze ucichnie na `_quiet()`, potem czysci bufor.

        Zwraca {id: wartosc} z NAJNOWSZYCH pelnych ramek serw `ids` (pusto = nic nowego).
        Protokol nie numeruje transakcji, a `_send` czysci tylko to, co JUZ przyszlo.
        Zmierzone przez most: odpowiedz sprzed przestoju sieci (albo spozniona o 0,3 s
        retransmisja TCP) przychodzila w trakcie NASTEPNEGO odczytu i byla brana za jego
        odpowiedz - blizniak robil z pozy sprzed 0,3 s rozkaz i serwo cofalo sie o 6-7 st.
        (takze po STOP-ie). Wlasciwa odpowiedz przychodzi zaraz za spozniona, w kolejnosci
        TCP, wiec wygrywa ostatnia ramka kazdego serwa, ktora zdazy przed cisza.
        """
        quiet = self._quiet()
        last = time.monotonic()
        hard = last + self.timeout + quiet          # ciagly szum nie trzyma petli bez konca
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
                    break                           # ucieta ramka - strumien przesuniety, bufor do kosza
                continue
            dev_id, data = frame
            if dev_id in ids and len(data) >= length:
                newer[dev_id] = int.from_bytes(data[:length], "little")
            last = time.monotonic()
        self._link.reset_input_buffer()
        return newer

    def _finish(self, out: dict[int, int], ids: list[int], length: int, t0: float) -> dict[int, int]:
        """Konczy odczyt: po zgubionej odpowiedzi oproznia lacze, a pierwszy odczyt po niej sprawdza.

        Odczyt, ktory dostal odpowiedz, gdy strumien byl podejrzany, czeka jeszcze na cisze:
        jesli za odpowiedzia przyszla nowsza, tamta byla spozniona i liczy sie tylko nowsza.
        Po zgubionej odpowiedzi lacze jest oprozniane (czytane i wyrzucane do ciszy), zeby
        spozniona ramka nie trafila do nastepnego rozkazu.
        """
        lost = self._last_rx == "lost"
        if out and self._suspect:
            newer = self._settle(ids, length)
            if newer:
                self.late_replies += 1
                logger.debug("Spozniona odpowiedz serw %s - biore nowsza", sorted(out))
                out = newer                         # czego nie ma w nowszej, jest nieswieze
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
        """Jeden pakiet, odpowiedz kazdego serwa po kolei - {id: wartosc} dla tych, ktore odpowiedzialy.

        Przez most sieciowy to jeden przebieg tam i z powrotem zamiast szesciu:
        przy 20 ms RTT odczyt ramienia kosztuje 20 ms, a nie 120.
        """
        params = bytes([addr, length, *ids])
        t0 = time.monotonic()
        self._send(BROADCAST_ID, INST_SYNC_READ, params)
        out: dict[int, int] = {}
        # Serwa odpowiadaja po kolei, ale to, ktore milczy, nie moze zabrac ze
        # soba reszty: ramki dopasowujemy po ID, nie po pozycji w strumieniu.
        # Inaczej jedno gluche serwo nr 1 wygladalo jak zerwane lacze.
        for _ in ids:
            frame = self._receive_frame()
            if frame is None:
                if self._last_rx == "lost":
                    break                   # cisza albo ucieta ramka - dalej nic juz nie przyjdzie rowno
                continue                    # cala ramka przeczytana, strumien rowny - nastepne sa dobre
            dev_id, data = frame
            if dev_id in ids and dev_id not in out and len(data) >= length:
                out[dev_id] = int.from_bytes(data[:length], "little")
            if dev_id == ids[-1]:
                break                       # ostatnie serwo odpowiedzialo - nie czekamy na brakujace
        return self._finish(out, list(ids), length, t0)

    def sync_write(self, addr: int, length: int, values: dict[int, int]) -> None:
        """Jeden pakiet dla wszystkich serw naraz - bez odpowiedzi, wiec bez czekania.

        To jest powod, dla ktorego petla sterowania nie placi za liczbe stawow:
        szesc celow kosztuje tyle, co jeden.
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
    """Ramie SO-101 sterowane wprost przez port szeregowy."""

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
        #: Limity odczytane z serw [tiki]. Pusty wpis = serwo nie ogranicza.
        self.servo_limits: dict[str, tuple[int, int]] = {}
        self._sync_read_ok = True
        #: SYNC READ odpowiedzial choc raz w tej sesji - wtedy cisza na niego to
        #: lacze, a nie stary firmware, i nie wolno go wylaczyc na stale.
        self._sync_ever_ok = False
        #: KOLEJNE cykle, w ktorych serwa odpowiedzialy na pojedyncze odczyty,
        #: a na SYNC READ nie (liczone tylko, zanim SYNC READ raz zadzialal).
        self._sync_misses = 0
        #: Kolejne cykle odczytu, w ktorych choc jeden staw nie dostal swiezej pozycji.
        self._failed_cycles = 0
        self._last_full_read = 0.0
        self._stale: list[str] = []
        #: Ostatni cykl odczytu bez ANI JEDNEJ odpowiedzi - lacze stoi.
        self._link_silent = False
        self.info = RobotInfo(
            name="SO-101 (feetech)",
            description=f"port {self.port} @ {self.settings.baudrate} bd",
            simulated=False,
        )

    # ------------------------------------------------------------- przeliczenia
    def _to_ticks(self, name: str, value: float) -> int:
        rc = self.settings
        if name == self.gripper:
            # Chwytak jedzie w 0..100 (0 = zwarty), a nie w stopniach.
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            ticks = rc.gripper_closed_ticks + span * (value / 100.0)
        else:
            ticks = rc.center_ticks + value * TICKS_PER_REV / 360.0
        return int(round(min(max(ticks, TICK_MIN), TICK_MAX)))

    def _to_units(self, name: str, ticks: int) -> float:
        rc = self.settings
        if name == self.gripper:
            span = rc.gripper_open_ticks - rc.gripper_closed_ticks
            if abs(span) < 1e-6:  # pragma: no cover - konfiguracja bez zakresu
                return 0.0
            return (ticks - rc.gripper_closed_ticks) * 100.0 / span
        return (ticks - rc.center_ticks) * 360.0 / TICKS_PER_REV

    # ---------------------------------------------------------------- polaczenie
    def connect(self) -> None:
        # Magistrala moze byc juz otwarta - tak wchodzi podstawiona w testach
        # albo wspoldzielona z diagnostyka. Powtorne otwarcie zerwaloby ja.
        if not self.bus.is_open:
            self.bus.open()
        # Nowa sesja - bledy i liczniki poprzedniej nie maja tu nic do powiedzenia.
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

        # KOLEJNOSC MA ZNACZENIE. Rejestr celu pamieta wartosc z poprzedniej
        # sesji, wiec samo zalaczenie momentu szarpneloby ramie do tamtej pozy.
        # Najpierw wpisujemy cel = tam, gdzie ramie faktycznie stoi.
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
        """Pozycja startowa z DWOCH zgodnych odczytow kazdego stawu.

        Ta pozycja staje sie celem serw tuz przed zalaczeniem momentu, wiec
        jeden przeklamany odczyt (8-bitowa suma kontrolna nie lapie kazdego
        zaklocenia) oznaczalby skok ramienia z pelna predkoscia. Dwa niezalezne
        odczyty, ktore sie zgadzaja, praktycznie to wykluczaja.
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
        """Odczytuje wlasne limity serw i mowi glosno, gdy sa ciasniejsze niz konfiguracja.

        Bez tego aplikacja zadaje pozy, ktorych serwo nie wykona: przycina je
        po cichu do swojego zakresu i wraca w zupelnie inne miejsce. Widac to
        wtedy jako "staw nie slucha rozkazu", chociaz rozkaz doszedl poprawnie.
        """
        self.servo_limits = {}
        conflicts: list[str] = []
        for name, dev_id in self.ids.items():
            low = self.bus.read(dev_id, ADDR_MIN_ANGLE_LIMIT, 2)
            high = self.bus.read(dev_id, ADDR_MAX_ANGLE_LIMIT, 2)
            if low is None or high is None or low >= high:
                continue
            if low <= TICK_MIN and high >= TICK_MAX:
                continue  # pelny zakres = serwo niczego nie ogranicza
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
        """Ostrzega, gdy ramie odpowiada, ale nie ma z czego ruszyc."""
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
        # Momentu NIE wylaczamy: ramie zostaloby wtedy wiotkie i opadlo pod
        # wlasnym ciezarem. Zostaje tam, gdzie stanelo.
        if self.settings.torque_off_on_exit:
            for dev_id in self.ids.values():
                try:
                    self.bus.write(dev_id, ADDR_TORQUE_ENABLE, 0, 1)
                except Exception:  # pragma: no cover - rozlaczenie nie moze wysypac aplikacji
                    logger.exception("Nie udalo sie wylaczyc momentu serwa %d", dev_id)
        self._connected = False
        self.bus.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ wymiana
    def read_joints(self) -> dict[str, float]:
        """Pozycje stawow; staw bez swiezej odpowiedzi zostaje przy poprzedniej wartosci.

        Nie rzuca z powodu zgubionych ramek - jedna zgubiona ramka nie ma prawa
        udawac skoku stawu ani zatrzymywac petli. Za to liczy KOLEJNE nieudane
        cykle, a `faults()` mowi glosno, gdy lacze padlo.
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
                # Choc jedna odpowiedz = firmware zna SYNC READ; brak reszty to juz lacze albo serwo.
                self._sync_ever_ok, self._sync_misses = True, 0

        missing = [name for name in self.ids if name not in fresh]
        # Cisza na SYNC READ, ktory juz w tej sesji dzialal, to stojace lacze, nie
        # firmware: szesc kolejnych odczytow po 0,25 s przez most trzymaloby
        # petle 1,75 s na cykl i nic by nie dalo. Wtedy nie probujemy po kolei.
        if missing and not (sync_silent and self._sync_ever_ok):
            failures = 0
            for name in missing:
                ticks = self.bus.read(self.ids[name], ADDR_PRESENT_POSITION, 2)
                if ticks is None:
                    failures += 1
                    if failures >= 2:
                        break                   # drugie milczenie w cyklu - nie placimy dalej timeoutami
                    continue
                fresh[name] = ticks
            if sync_silent and not self._sync_ever_ok and fresh:
                # Serwa odpowiadaja po kolei, a na SYNC READ milcza. Liczone KOLEJNE
                # cykle - wczesniej trzy przypadkowe czkawki sieci w calej sesji
                # wylaczaly SYNC READ na zawsze i petla przez most spadala do ~7 Hz.
                self._sync_misses += 1
                if self._sync_misses >= 3:
                    # Starsze wersje firmware'u nie znaja SYNC READ - wtedy zostajemy przy odczytach po kolei.
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
        """Bity bledu z ostatnich odpowiedzi serw i utrata lacza - zdania dla operatora."""
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
        except Exception as exc:  # pragma: no cover - wolane co cykl petli, nie moze jej wysypac
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
        """Limity kata z EEPROM serw w jednostkach aplikacji (tylko stawy, ktore je maja)."""
        out: dict[str, tuple[float, float]] = {}
        for name, (low, high) in self.servo_limits.items():
            a, b = self._to_units(name, low), self._to_units(name, high)
            out[name] = (min(a, b), max(a, b))
        return out

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Ramie nie jest polaczone")
        if self._link_silent:
            # Ostatni odczyt nie dostal ANI JEDNEJ odpowiedzi - lacze stoi. Przez
            # most `socket://` zapis i tak by sie "udal" (laduje w buforze TCP),
            # a po powrocie sieci cala kolejka celow dochodzi do serw naraz i
            # ramie skacze do ostatniego z nich z pelna predkoscia - takze po
            # STOP-ie wcisnietym w trakcie przestoju. Nic nie wysylamy, dopoki
            # odczyt nie potwierdzi, ze serwa znow slysza.
            return {}
        sent: dict[str, float] = {}
        ticks: dict[int, int] = {}
        for name in self.joints:
            if name not in targets:
                continue
            lo, hi = self.limits[name]
            value = min(max(float(targets[name]), lo), hi)
            tick = self._to_ticks(name, value)

            # Serwo i tak przytnie rozkaz do swojego zakresu. Robiac to tutaj,
            # zwracamy wyzej wartosc, ktora NAPRAWDE pojechala - inaczej reszta
            # aplikacji (limity predkosci, HUD, podglad 3D) liczylaby na pozycji,
            # ktorej ramie nigdy nie osiagnelo.
            limits = self.servo_limits.get(name)
            if limits is not None:
                tick = min(max(tick, limits[0]), limits[1])
                value = self._to_units(name, tick)

            ticks[self.ids[name]] = tick
            sent[name] = value
        self.bus.sync_write(ADDR_GOAL_POSITION, 2, ticks)
        return sent
