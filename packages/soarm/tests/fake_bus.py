"""Six STS3215 servos on a fake bus, for the tests of the `feetech` backend.

`FakeBusLink` stands in for the serial port (`FeetechBus.open(link=...)`): it answers packets as the
servos do and logs every operation. test_feetech.py and test_twin_contract.py use it.
"""

from __future__ import annotations

from soarm.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_MAX_ANGLE_LIMIT,
    ADDR_MIN_ANGLE_LIMIT,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_VOLTAGE,
    ADDR_TORQUE_ENABLE,
    BROADCAST_ID,
    INST_PING,
    INST_READ,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    checksum,
)

REST_TICKS = 2048


class FakeBusLink:
    """Szesc serw na jednej magistrali - tyle protokolu, ile backend uzywa."""

    #: Starszy firmware nie zna SYNC READ i po prostu milczy.
    knows_sync_read = True

    def __init__(self, ids=(1, 2, 3, 4, 5, 6), start_ticks: int = REST_TICKS):
        self.registers = {
            dev_id: {
                ADDR_TORQUE_ENABLE: 0,
                ADDR_GOAL_POSITION: 3000,  # celowo INNY niz pozycja biezaca
                ADDR_PRESENT_POSITION: start_ticks,
                ADDR_PRESENT_VOLTAGE: 119,
                ADDR_MIN_ANGLE_LIMIT: 0,
                ADDR_MAX_ANGLE_LIMIT: 4095,
            }
            for dev_id in ids
        }
        #: Slad operacji: ("ping"|"read"|"write"|"sync", id, addr, wartosc).
        self.log: list[tuple] = []
        #: Bajt bledu, ktory serwo wpisuje w kazda odpowiedz (0x20 = przeciazenie).
        self.status: dict[int, int] = {}
        #: Serwa, ktore przestaly odpowiadac (odlaczony przewod, zgubione ramki).
        self.mute: set[int] = set()
        #: Cala magistrala milczy - tak wyglada zerwany most sieciowy.
        self.silent = False
        #: Serwa, ktorych NASTEPNA odpowiedz z danymi dostanie przeklamany bit
        #: przy niezmienionej sumie kontrolnej - zaklocenie na linii TTL.
        self.corrupt_next: set[int] = set()
        self._out = bytearray()
        self.closed = False

    # ------------------------------------------------------------- pyserial
    def reset_input_buffer(self) -> None:
        self._out.clear()

    def reset_output_buffer(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def read(self, size: int = 1) -> bytes:
        out, self._out = bytes(self._out[:size]), self._out[size:]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self._out)

    # --------------------------------------------------------------- serwo
    def write(self, data: bytes) -> int:
        assert data[:2] == b"\xff\xff", "brak naglowka pakietu"
        dev_id, length, instruction = data[2], data[3], data[4]
        params = data[5:-1]
        assert checksum(data[2:-1]) == data[-1], "zla suma kontrolna"
        assert length == len(params) + 2, "dlugosc nie zgadza sie z parametrami"

        # Serwo, ktorego nie ma na magistrali, po prostu milczy - i tak wlasnie
        # wyglada brakujacy staw dla backendu.
        if dev_id != BROADCAST_ID and dev_id not in self.registers:
            self.log.append(("cisza", dev_id, None, None))
            return len(data)

        if instruction == INST_PING:
            self.log.append(("ping", dev_id, None, None))
            self._reply(dev_id, b"")
        elif instruction == INST_READ:
            addr, size = params[0], params[1]
            self.log.append(("read", dev_id, addr, None))
            value = self.registers[dev_id].get(addr, 0)
            self._reply(dev_id, value.to_bytes(size, "little"))
        elif instruction == INST_WRITE:
            addr, value = params[0], int.from_bytes(params[1:], "little")
            self.registers[dev_id][addr] = value
            self.log.append(("write", dev_id, addr, value))
            self._reply(dev_id, b"")
        elif instruction == INST_SYNC_READ:
            assert dev_id == BROADCAST_ID, "sync read musi isc rozgloszeniowo"
            addr, size = params[0], params[1]
            self.log.append(("syncread", None, addr, tuple(params[2:])))
            if self.knows_sync_read:
                for target in params[2:]:
                    if target in self.registers:
                        self._reply(target, self.registers[target].get(addr, 0).to_bytes(size, "little"))
        elif instruction == INST_SYNC_WRITE:
            assert dev_id == BROADCAST_ID, "sync write musi isc rozgloszeniowo"
            addr, size = params[0], params[1]
            body = params[2:]
            stride = size + 1
            for offset in range(0, len(body), stride):
                target = body[offset]
                value = int.from_bytes(body[offset + 1 : offset + stride], "little")
                self.registers[target][addr] = value
                self.log.append(("sync", target, addr, value))
        else:  # pragma: no cover - backend nie uzywa innych instrukcji
            raise AssertionError(f"nieznana instrukcja 0x{instruction:02X}")
        return len(data)

    def _reply(self, dev_id: int, payload: bytes) -> None:
        if self.silent or dev_id in self.mute:
            return
        body = bytes([dev_id, len(payload) + 2, self.status.get(dev_id, 0)]) + payload
        packet = bytearray(b"\xff\xff" + body + bytes([checksum(body)]))
        if dev_id in self.corrupt_next and payload:
            self.corrupt_next.discard(dev_id)
            packet[-2] ^= 0x08                       # bit 3 starszego bajtu: 2048 tikow -> 0
        self._out += packet

    # ------------------------------------------------------------- pomocnicze
    def ops(self, kind: str) -> list[tuple]:
        return [entry for entry in self.log if entry[0] == kind]
