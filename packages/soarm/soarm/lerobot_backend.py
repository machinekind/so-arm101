"""Sterowanie prawdziwym ramieniem SO-101 przez biblioteke LeRobot.

Uklad modulow w LeRobot zmienial sie miedzy wydaniami (0.4 scalilo SO-100 i
SO-101 w `lerobot.robots.so_follower`, wczesniej byl osobny `so101_follower`,
a jeszcze wczesniej caly pakiet siedzial pod `lerobot.common`). Zamiast zakladac
jedna wersje, szukamy klas po kolei we wszystkich znanych lokalizacjach.
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

from .feetech import LINK_LOSS_CYCLES, Info, describe_servo_error
from .settings import from_mapping

logger = logging.getLogger(__name__)

#: Co tyle odczytow pozycji odczyt rejestru Status serw (bity ochrony: przeciazenie,
#: przegrzanie, napiecie). `sync_read` LeRobota nie oddaje bajtu bledu odpowiedzi,
#: wiec bez tego serwo w ochronie (zwolniony moment, dalej grzecznie odpowiada)
#: wygladalo na zdrowe, a fala i identyfikacja jechaly dalej reszta ramienia.
#: Przy odczycie pozycji 10 Hz to jeden krotki SYNC READ co 0,3 s - petla tego nie czuje.
#: Liczone w odczytach, nie w sekundach: blizniak bez watku (testy) ma wlasny zegar,
#: a zegar sciany stal wtedy w miejscu i Status nie byl czytany wcale.
STATUS_EVERY_READS = 3

#: Tyle KOLEJNYCH nieudanych odczytow Status (~1,5 s przy 10 Hz) = ochrona serw niewidoczna,
#: co jest usterka sama w sobie. Pojedyncza czkawka magistrali nie zatrzymuje ramienia.
STATUS_FAIL_LIMIT = 5

#: Tiki na obrot STS3215, gdy magistrala LeRobota nie poda wlasnej tabeli rozdzielczosci.
DEFAULT_RESOLUTION = 4096

#: O tyle tikow (~1,8 st.) zero stawu LeRobota (srodek zakresu z kalibracji) moze
#: odbiegac od zera, ktore zaklada blizniak (`lerobot.center_ticks`), zanim ostrzezemy.
ZERO_TOLERANCE_TICKS = 20

#: Kolejne lokalizacje klas SO-follower - od najnowszej do najstarszej.
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


def create(ctx: Any) -> LeRobotArm:
    """Entry point of the `lerobot` backend plugin."""
    return LeRobotArm(ctx)


class LeRobotArm:
    """Adapter na `SO101Follower` / `SO100Follower` z LeRobot."""

    def __init__(self, ctx: Any):
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
        self.info = Info(
            name=f"{kind.upper()} (LeRobot)",
            description=f"port {self.port}",
            simulated=False,
        )

    # ------------------------------------------------------------- importy
    @staticmethod
    def _resolve_classes(kind: str) -> tuple[Any, Any, str]:
        """Znajduje (klasa_robota, klasa_konfiguracji, nazwa_modulu) w LeRobot."""
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
        """Sklada obiekt konfiguracji, pomijajac pola nieznane danej wersji."""
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
            # LeRobot domyslnie zdejmuje moment przy rozlaczeniu - ramie opada pod
            # wlasnym ciezarem. U nas decyduje o tym ta sama opcja co w `feetech`.
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

    # ------------------------------------------------------------ polaczenie
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
        """Odmawia polaczenia, zanim cokolwiek dotknie portu, gdy LeRobot nie ma pliku kalibracji.

        Bez pliku `SOFollower.connect()` wola `calibrate()`: zdejmuje moment ze
        wszystkich serw (ramie trzymane przez poprzednia sesje opada na stol)
        i czeka na `input()` w konsoli serwera - w panelu nic nie widac, a watek
        wisi. Dokonczona kalibracja nadpisalaby jeszcze Homing_Offset i limity
        w EEPROM, czyli zero i zakresy, na ktorych stoi backend `feetech`.
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
        """Laczy tak, zeby LeRobot nigdy nie zdjal momentu ani nie zapytal o nic w konsoli."""
        bus = getattr(robot, "bus", None)
        # `getattr_static`, nie `hasattr`: `is_calibrated` magistrali LeRobota to wlasciwosc,
        # ktora CZYTA serwa, a na niepolaczonej rzuca DeviceNotConnectedError (ConnectionError,
        # nie AttributeError). `hasattr` wywolywal ja przed `bus.connect()` i LeRobot 0.6.1
        # z plikiem kalibracji nie laczyl sie NIGDY (odtworzone na emulatorze magistrali).
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
                # `configure()` LeRobota robi swoje zapisy pod `torque_disabled()` - przy
                # ramieniu trzymajacym poze (np. zaraz po sesji `feetech`) to zdjety moment
                # na czas kilkudziesieciu transakcji, a przez most sekunda swobodnego spadku.
                # Ustawienia z poprzednich polaczen LeRobota i tak siedza w EEPROM serw.
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
            except Exception:  # pragma: no cover - rozlaczenie nie moze przykryc wlasciwego bledu
                logger.exception("Blad przy rozlaczaniu robota")
            raise RuntimeError(
                "Kalibracja LeRobota w pliku nie zgadza sie z zapisana w serwach. " + self._calibrate_hint()
            )

    @staticmethod
    def _torque_on(bus: Any) -> bool:
        """Czy ktores serwo trzyma moment. W razie watpliwosci - tak (bezpieczniej)."""
        try:
            return any(int(v) for v in bus.sync_read("Torque_Enable").values())
        except Exception:
            logger.debug("Nie udalo sie odczytac Torque_Enable - zakladam, ze serwa trzymaja", exc_info=True)
            return True

    @staticmethod
    def _close_bus_keeping_torque(bus: Any) -> None:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # pragma: no cover - rozlaczenie nie moze przykryc wlasciwego bledu
            logger.exception("Blad przy zamykaniu magistrali LeRobota")

    def motor_names(self) -> list[str]:
        """Nazwy stawow zgloszone przez LeRobot (kolejnosc jak w sterowniku)."""
        if self._robot is None:
            return list(self.joints)
        try:
            return [key.removesuffix(".pos") for key in self._robot.action_features]
        except Exception:  # pragma: no cover - zalezne od wersji LeRobot
            return list(self.joints)

    def disconnect(self) -> None:
        if self._robot is not None and self._connected:
            try:
                self._robot.disconnect()
            except Exception:  # pragma: no cover - rozlaczenie nie moze wysypac aplikacji
                logger.exception("Blad przy rozlaczaniu robota")
        self._connected = False
        self._robot = None
        self._reset_status()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def link_silent(self) -> bool:
        """Ostatni `read_joints` nie dostal ZADNEJ swiezej odpowiedzi - oddal stare pozycje."""
        return bool(getattr(self, "_link_silent", False))

    def __enter__(self) -> LeRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------- wymiana IO
    def read_joints(self) -> dict[str, float]:
        """Pozycje stawow; nieudany odczyt oddaje poprzednie i ustawia `link_silent` (jak `feetech`).

        LeRobot rzuca ConnectionError, gdy SYNC READ Present_Position nie dostal odpowiedzi
        w `num_read_retries + 1` probach - krotka seria zaklocen przy ruchu kilku stawow.
        Wyjatek konczyl petle blizniaka (rozlaczenie, ponowne laczenie w panelu), a `feetech`
        przezywa to samo bez mrugniecia. Kolejne nieudane cykle ida do `faults()`
        ("brak odpowiedzi serw ..."), ktore blizniak traktuje jak zerwane lacze.
        """
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        try:
            observation = self._robot.get_observation()
        except ConnectionError as exc:
            if not self._positions or not self._robot_connected():
                raise                               # nie ma czego oddac albo robot naprawde rozlaczony
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
        """`DeviceNotConnectedError` to tez ConnectionError - rozlaczony robot nie jest czkawka lacza."""
        try:
            return bool(getattr(self._robot, "is_connected", True))
        except Exception:  # pragma: no cover - zalezne od wersji LeRobot
            return False

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot nie jest polaczony")
        if self._link_silent:
            # Jak `feetech`: ostatni odczyt nie dostal odpowiedzi - nic nie wysylamy, dopoki
            # odczyt nie potwierdzi, ze serwa slysza i gdzie ramie naprawde jest.
            return {}
        known = set(self.motor_names())
        # W stopniach LeRobot nie przycina celu (przycina tylko skale -100..100 i 0..100),
        # a serwo robi to po cichu do Min/Max_Position_Limit z EEPROM. Przycinamy tutaj,
        # zeby wyzej wrocilo to, co NAPRAWDE pojechalo - inaczej nadzor, polityka
        # i identyfikacja liczyly na pozycji, ktorej ramie nigdy nie osiagnelo (jak w `feetech`).
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
            # Z `max_relative_target` LeRobot czyta Present_Position PRZED zapisem celu - ten
            # odczyt tez gubi ramki. Nic nie poszlo; wolajacy zostaje przy poprzednim rozkazie.
            if not self._robot_connected():
                raise
            logger.debug("Nieudany odczyt pozycji przed wyslaniem celu - nic nie wyslano", exc_info=True)
            return {}
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in (sent or action).items()
            if key.endswith(".pos")
        }

    # ------------------------------------------------------------ stan serw
    def _reset_status(self) -> None:
        #: Ostatnio odczytany rejestr Status kazdego serwa {staw: bity}.
        self._status: dict[str, int] = {}
        #: Odczyty pozycji do nastepnego odczytu Status (0 = przy najblizszym).
        self._reads_to_status = 0
        #: KOLEJNE nieudane odczyty Status i ostatni blad (do komunikatu).
        self._status_failures = 0
        self._status_error = ""
        #: Ostatnie dobre pozycje - oddawane, gdy odczyt nie dostal odpowiedzi.
        self._positions: dict[str, float] = {}
        #: Ostatni odczyt pozycji bez odpowiedzi (`link_silent`) i stawy bez swiezej wartosci.
        self._link_silent = False
        self._stale: list[str] = []
        #: KOLEJNE nieudane odczyty pozycji, ostatni blad i czas ostatniego dobrego odczytu.
        self._failed_cycles = 0
        self._link_error = ""
        self._last_full_read = time.monotonic()

    def _poll_status(self) -> None:
        """Czyta rejestr Status wszystkich serw co `STATUS_EVERY_READS` odczytow pozycji. Nigdy nie rzuca.

        Rejestr ma te same bity, co bajt bledu odpowiedzi (0x20 przeciazenie, 0x04
        przegrzanie...), a SYNC READ odpowiada nawet serwo w ochronie. Nieudany odczyt
        zostawia poprzedni stan - bity przeciazenia nie znikaja przez jedna zgubiona ramke.
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
        """Bity ochrony z rejestru Status serw i utrata ich odczytu - zdania dla operatora.

        Te same zdania co w `feetech` ("staw (serwo N): przeciazenie"), wiec nadzor
        blizniaka zatrzymuje ramie tak samo na obu backendach.
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
                # To samo zdanie co `feetech` - blizniak rozpoznaje po nim zerwane lacze.
                ms = (time.monotonic() - self._last_full_read) * 1000.0
                out.append(f"brak odpowiedzi serw od {ms:.0f} ms ({self._link_error})")
            return out
        except Exception as exc:  # pragma: no cover - wolane co cykl petli, nie moze jej wysypac
            logger.exception("Nie udalo sie sprawdzic stanu serw")
            return [f"nie udalo sie sprawdzic stanu serw: {exc}"]

    # ------------------------------------------------------------ kalibracja
    def _bus(self) -> Any:
        return getattr(self._robot, "bus", None) if self._robot is not None else None

    def _calibration(self) -> dict[str, Any]:
        """Kalibracja, ktora LeRobot naprawde stosuje (ta z magistrali), inaczej ta z pliku.

        Przy laczeniu `is_calibrated` sprawdzilo, ze zakresy z pliku sa rowne
        Min/Max_Position_Limit w EEPROM serw - wiec to jest tez to, do czego serwo przycina.
        """
        cal = getattr(self._bus(), "calibration", None) or getattr(self._robot, "calibration", None)
        return dict(cal) if cal else {}

    def _motors(self) -> dict[str, Any]:
        return dict(getattr(self._bus(), "motors", None) or {})

    def _motor_ids(self) -> dict[str, int]:
        return {str(name): int(getattr(m, "id", 0)) for name, m in self._motors().items()}

    def _norm_mode(self, name: str) -> str:
        """Tryb normalizacji stawu w LeRobot: DEGREES, RANGE_M100_100 albo RANGE_0_100."""
        mode = getattr(self._motors().get(name), "norm_mode", None)
        return str(getattr(mode, "name", mode) or "").upper()

    def _resolution(self, name: str) -> int:
        table = getattr(self._bus(), "model_resolution_table", None) or {}
        return int(table.get(getattr(self._motors().get(name), "model", None), DEFAULT_RESOLUTION))

    def _drive_inverted(self, cal: Any) -> bool:
        return bool(getattr(self._bus(), "apply_drive_mode", True)) and bool(getattr(cal, "drive_mode", 0))

    def _ticks_to_units(self, name: str, ticks: float) -> float | None:
        """Tiki serwa -> jednostki LeRobota, tym samym wzorem co `MotorsBus._normalize`
        (bez przycinania do zakresu). None = staw bez kalibracji albo w nieznanym trybie."""
        cal = self._calibration().get(name)
        if cal is None:
            return None
        lo, hi = float(cal.range_min), float(cal.range_max)
        if hi == lo:
            return None
        mode = self._norm_mode(name)
        if mode == "DEGREES":
            # Zero stawu w stopniach to SRODEK zakresu z kalibracji; drive_mode tu nie dziala.
            return (ticks - (lo + hi) / 2.0) * 360.0 / (self._resolution(name) - 1)
        if mode == "RANGE_M100_100":
            norm = (ticks - lo) / (hi - lo) * 200.0 - 100.0
            return -norm if self._drive_inverted(cal) else norm
        if mode == "RANGE_0_100":
            norm = (ticks - lo) / (hi - lo) * 100.0
            return 100.0 - norm if self._drive_inverted(cal) else norm
        return None

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Zakres z kalibracji LeRobota (= Min/Max_Position_Limit w EEPROM) w jednostkach aplikacji.

        Tylko stawy, ktore serwo naprawde ogranicza - pelny obrot (0..4095, zwykle
        wrist_roll) jest pomijany jak w `feetech`. Bez tego nadzor liczyl na cele
        spoza zakresu, ktore serwo po cichu przycinalo.
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
        """O ile stopni zero stawu LeRobota lezy od zera blizniaka: {staw: stopnie} (ponad tolerancje).

        W stopniach LeRobot liczy kat od SRODKA zakresu z kalibracji, a blizniak
        (kinematyka, polityki) od `lerobot.center_ticks` - tak jak backend `feetech`.
        Dla typowej kalibracji SO-101 to kilka stopni na stawie, dla zakresu nagranego
        tylko w jedna strone (limity ramienia nr 1: wrist_flex 2025..2858) ~35 st.
        Nie przeliczamy tego po cichu - zmienilyby sie jednostki zapisow zgodnych z LeRobotem.
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
        """Stawy ramienia (bez chwytaka), ktorych LeRobot NIE podaje w stopniach.

        `lerobot.use_degrees: false` przestawia je na -100..100 procent zakresu z kalibracji,
        a kinematyka, polityki i IK traktuja kazda liczbe jak stopnie - 100 "stopni" to wtedy
        koniec zakresu stawu. Wczesniej nic w panelu o tym nie mowilo.
        """
        return [str(name) for name in self._motors()
                if name != self.gripper and self._norm_mode(str(name)) != "DEGREES"]

    def calibration_warnings(self) -> list[str]:
        """Niefatalne zastrzezenia do kalibracji tego ramienia - zdania dla panelu. Nigdy nie rzuca."""
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
        except Exception as exc:  # pragma: no cover - sama diagnostyka nie moze zablokowac polaczenia
            logger.debug("Nie udalo sie porownac zera stawow z kalibracja LeRobota", exc_info=True)
            return [f"nie udalo sie sprawdzic kalibracji LeRobota: {exc}"]

    def _warn_about_joint_zero(self) -> None:
        for text in self.calibration_warnings():
            if text.startswith("UWAGA"):
                logger.error(text)
            else:
                logger.warning(text)

