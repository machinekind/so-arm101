from __future__ import annotations

import pytest

from soarm.backend import Backend
from soarm.bridge import BridgeArm
from soarm.feetech import FeetechArm
from soarm.lerobot import LeRobotArm

MEMBERS = {
    "connect", "disconnect", "read_joints", "send_joints",
    "is_connected", "link_silent", "faults", "joint_limits",
}


def test_protocol_has_the_twin_backend_shape() -> None:
    assert {name for name in vars(Backend) if not name.startswith("_")} == MEMBERS


@pytest.mark.parametrize("cls", [FeetechArm, BridgeArm, LeRobotArm])
def test_stub_has_every_backend_member(cls: type) -> None:
    assert all(hasattr(cls, name) for name in MEMBERS)


@pytest.mark.parametrize("cls", [FeetechArm, BridgeArm, LeRobotArm])
def test_stub_refuses_to_construct(cls: type) -> None:
    with pytest.raises(NotImplementedError, match="#145"):
        cls()
