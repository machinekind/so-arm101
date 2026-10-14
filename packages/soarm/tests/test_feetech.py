"""Backend `feetech`: protokol STS3215, przeliczenia jednostek i kolejnosc startu.

Testy chodza po *symulowanej magistrali* (`fake_bus.FakeBusLink`) - odpowiada na pakiety tak, jak
prawdziwe serwo, i zapisuje kolejnosc operacji. Dzieki temu da sie sprawdzic
rzeczy, ktorych na sprzecie sprawdzic nie sposob bez ryzyka: na przyklad ze
moment zalacza sie DOPIERO po nadpisaniu rejestru celu.
"""

from __future__ import annotations

import threading
import time

import pytest
from lerobot_mp_api.backend import BackendContext

from fake_bus import REST_TICKS, FakeBusLink
from soarm import serial_bridge
from soarm.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    INST_READ,
    INST_SYNC_READ,
    FeetechArm,
    FeetechBus,
    checksum,
)


@pytest.fixture
def arm_ctx(so101_context) -> BackendContext:
    """What the twin hands the `feetech` backend for the SO-101 on port COM_TEST."""
    return so101_context()


def connected(ctx, link: FakeBusLink | None = None) -> tuple[FeetechArm, FakeBusLink]:
    link = link or FakeBusLink()
    bus = FeetechBus("COM_TEST")
    bus.open(link=link)
    arm = FeetechArm(ctx, bus=bus)
    arm.connect()
    return arm, link


def test_connect_writes_the_goal_before_enabling_torque(arm_ctx):
    """Rejestr celu pamieta poprzednia sesje - moment zalaczony przed jego
    nadpisaniem szarpnalby ramieniem do tamtej pozy."""
    _, link = connected(arm_ctx)
    first_torque = next(
        i for i, entry in enumerate(link.log)
        if entry[0] == "write" and entry[2] == ADDR_TORQUE_ENABLE
    )
    goal_writes = [
        i for i, entry in enumerate(link.log)
        if entry[0] == "sync" and entry[2] == ADDR_GOAL_POSITION
    ]
    assert goal_writes, "cel nie zostal w ogole ustawiony"
    assert max(goal_writes) < first_torque


def test_connect_refuses_when_a_servo_is_silent(arm_ctx):
    link = FakeBusLink(ids=(1, 2, 3, 4, 5))  # brakuje chwytaka
    bus = FeetechBus("COM_TEST")
    bus.open(link=link)
    with pytest.raises(RuntimeError, match="gripper"):
        FeetechArm(arm_ctx, bus=bus).connect()


def test_gripper_range_is_the_one_the_backend_converts_with_and_matches_the_description(arm_ctx, robot_yaml):
    """The twin compares it with robot.yaml's `gripper_range_q`: it must be what the backend computes with."""
    arm, _ = connected(arm_ctx)
    rad = 2 * 3.141592653589793 / 4096
    closed, opened = arm.gripper_range_q()
    assert arm._to_ticks("gripper", 0.0) == round(2048 + closed / rad) == 1986
    assert arm._to_ticks("gripper", 100.0) == round(2048 + opened / rad) == 2670
    assert arm._to_units("shoulder_pan", 2048) == 0.0
    assert (closed, opened) == pytest.approx(robot_yaml["gripper_range_q"], abs=1e-6)


def test_send_joints_clamps_to_the_configured_limits(arm_ctx):
    arm, link = connected(arm_ctx)
    limit = arm_ctx.limits["shoulder_pan"][1]
    sent = arm.send_joints({"shoulder_pan": limit + 500.0})
    assert sent["shoulder_pan"] == pytest.approx(limit)
    assert link.registers[1][ADDR_GOAL_POSITION] == arm._to_ticks("shoulder_pan", limit)


def test_targets_never_leave_the_servo_tick_range(arm_ctx):
    """Nawet absurdalny cel nie moze przepelnic dwubajtowego rejestru."""
    arm, link = connected(arm_ctx)
    arm.send_joints({name: 1e6 for name in arm_ctx.joints})
    for registers in link.registers.values():
        assert 0 <= registers[ADDR_GOAL_POSITION] <= 4095


def test_a_reply_with_a_bad_checksum_is_rejected(arm_ctx):
    """Jeden przeklamany bit na linii TTL dawal -180 stopni zamiast 0 - i to szlo jako rozkaz."""
    arm, link = connected(arm_ctx)
    before = arm.read_joints()
    link.corrupt_next = {1}
    positions = arm.read_joints()
    assert positions["shoulder_pan"] == pytest.approx(before["shoulder_pan"])
    assert arm.bus.corrupt_replies == 1


def test_connect_takes_the_start_position_only_from_two_matching_reads(arm_ctx):
    """Pozycja startowa staje sie celem przed zalaczeniem momentu - jeden zly odczyt to skok ramienia."""
    link = FakeBusLink()
    real_write = link.write
    glitches = {"left": 1}

    def glitchy(data: bytes) -> int:
        n = real_write(data)
        # Pierwszy odczyt pozycji barku wraca z inna wartoscia (z poprawna suma) - raz.
        if data[4] == INST_READ and data[2] == 1 and data[5] == ADDR_PRESENT_POSITION and glitches["left"]:
            glitches["left"] -= 1
            link._out.clear()
            link._reply(1, (100).to_bytes(2, "little"))
        return n

    link.write = glitchy
    connected(arm_ctx, link)
    assert link.registers[1][ADDR_GOAL_POSITION] == REST_TICKS


def test_servo_protection_bits_are_reported_as_faults(arm_ctx):
    """Serwo w ochronie zwalnia moment, ale dalej odpowiada - bez tego wygladalo na zdrowe."""
    arm, link = connected(arm_ctx)
    arm.read_joints()
    assert arm.faults() == []
    link.status[2] = 0x20
    arm.read_joints()
    faults = arm.faults()
    assert len(faults) == 1
    assert "shoulder_lift" in faults[0] and "przeciazenie" in faults[0]
    link.status[2] = 0
    arm.read_joints()
    assert arm.faults() == []


def test_a_dead_link_is_reported_after_consecutive_failed_reads(arm_ctx):
    arm, link = connected(arm_ctx)
    arm.read_joints()
    link.silent = True
    for _ in range(4):
        arm.read_joints()
    assert arm.faults() == []                       # kilka zgubionych ramek to jeszcze nie awaria
    arm.read_joints()
    assert any("brak odpowiedzi serw od" in fault for fault in arm.faults())
    link.silent = False
    arm.read_joints()
    assert arm.faults() == []


def test_no_goals_are_queued_while_the_link_is_silent(arm_ctx):
    """Zapis przez `socket://` "udaje sie" do bufora TCP, a po powrocie sieci cala
    kolejka celow dochodzila naraz - ramie skakalo do rozkazu sprzed STOP-u."""
    arm, link = connected(arm_ctx)
    arm.read_joints()
    link.silent = True
    arm.read_joints()
    link.log.clear()
    assert arm.send_joints({"shoulder_pan": 30.0}) == {}
    assert link.ops("sync") == []
    link.silent = False
    arm.read_joints()
    assert arm.send_joints({"shoulder_pan": 30.0})["shoulder_pan"] == pytest.approx(30.0)
    assert link.ops("sync")


class DelayedServoPort:
    """Szesc serw za mostem `serial_bridge`, z odpowiedziami, ktore moga przyjsc za pozno.

    Odpowiedzi ida w kolejnosci (jak TCP): spozniona wstrzymuje wszystkie za nia.
    `delay_next` - dodatkowe opoznienie odpowiedzi na NASTEPNY SYNC READ (retransmisja TCP).
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pos = {dev_id: REST_TICKS for dev_id in range(1, 7)}
        #: (kiedy gotowe, id, wartosc albo None = pozycja w chwili, gdy zapytanie dotrze).
        self.queue: list[tuple[float, int, int | None]] = []
        self.delay_next = 0.0
        self.hold_until = 0.0
        self._capture_next = False
        self.closed = False

    def write(self, data: bytes) -> int:
        now = time.monotonic()
        with self.lock:
            i = 0
            while i + 5 < len(data):
                length, instruction, params = data[i + 3], data[i + 4], data[i + 5: i + 3 + data[i + 3]]
                i += 4 + length
                if instruction != INST_SYNC_READ:
                    continue
                extra, self.delay_next = self.delay_next, 0.0
                ready = max(now + extra, self.hold_until, self.queue[-1][0] if self.queue else 0.0)
                in_flight = now >= self.hold_until or self._capture_next
                self._capture_next = False
                for dev_id in params[2:]:
                    self.queue.append((ready, dev_id, self.pos[dev_id] if in_flight else None))
        return len(data)

    def _ready(self, size: int) -> bytes:
        out = bytearray()
        now = time.monotonic()
        while self.queue and self.queue[0][0] <= now and len(out) < size:
            _, dev_id, value = self.queue.pop(0)
            value = self.pos[dev_id] if value is None else value
            body = bytes([dev_id, 4, 0]) + value.to_bytes(2, "little")
            out += b"\xff\xff" + body + bytes([checksum(body)])
        return bytes(out)

    def read(self, size: int = 1) -> bytes:
        deadline = time.monotonic() + serial_bridge.POLL_S
        while not self.closed:
            with self.lock:
                data = self._ready(size)
            if data or time.monotonic() >= deadline:
                return data
            time.sleep(0.001)
        return b""

    @property
    def in_waiting(self) -> int:
        return 0                                     # `_ready` oddaje cale ramki od razu

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def bridged():
    """FeetechBus po `socket://` przez prawdziwy most do `DelayedServoPort`."""
    pytest.importorskip("serial")
    port, ready, stop = DelayedServoPort(), threading.Event(), threading.Event()
    bound: list[int] = []
    server = threading.Thread(target=serial_bridge.serve, args=(lambda: port, "127.0.0.1", 0),
                              kwargs=dict(stop=stop, on_ready=lambda p: (bound.append(p), ready.set())), daemon=True)
    server.start()
    assert ready.wait(5)
    bus = FeetechBus(f"socket://127.0.0.1:{bound[0]}")
    bus.open()
    try:
        yield bus, port
    finally:
        bus.close()
        stop.set()
        port.close()
        server.join(3)


IDS = [1, 2, 3, 4, 5, 6]


def first_answer(bus: FeetechBus, tries: int = 8) -> dict[int, int]:
    for _ in range(tries):
        got = bus.sync_read(ADDR_PRESENT_POSITION, 2, IDS)
        if got:
            return got
    raise AssertionError("serwa nie odpowiedzialy ani razu")


def test_a_late_reply_is_never_taken_for_the_next_read(bridged):
    """Odpowiedz spozniona o 0,4 s (limit 0,25 s) przychodzila w trakcie NASTEPNEGO odczytu
    i byla brana za jego odpowiedz: blizniak dostawal poze sprzed 0,4 s jako swieza."""
    bus, port = bridged
    assert first_answer(bus)[1] == REST_TICKS
    port.delay_next = 0.4
    assert bus.sync_read(ADDR_PRESENT_POSITION, 2, IDS) == {}          # limit minal - nic
    port.pos[1] = REST_TICKS + 500                                      # serwo pojechalo dalej
    assert first_answer(bus)[1] == REST_TICKS + 500


def test_disconnect_leaves_the_arm_holding_by_default(arm_ctx):
    """Wylaczony moment znaczy wiotkie ramie, ktore opada pod wlasnym ciezarem."""
    arm, link = connected(arm_ctx)
    arm.disconnect()
    assert all(regs[ADDR_TORQUE_ENABLE] == 1 for regs in link.registers.values())
    assert link.closed
