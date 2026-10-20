"""`soarm-zero`: move an arm slowly to its zero pose (or `home`) and hold it there, or release it.

    uv run soarm-zero                                  # the one USB-serial arm, to the zero pose
    uv run soarm-zero --port /dev/ttyACM0 --port /dev/ttyACM1   # several arms, one after the other
    uv run soarm-zero --home                           # robot.yaml's `home` instead of the zero pose
    uv run soarm-zero --off                            # ... then switch torque off (support the arm!)

The zero pose is every joint at 0 degrees and the gripper at its zero tick (2048): LeRobot's middle
pose, upper arm vertical, forearm forward, wrist straight. Torque is enabled where the arm stands
(`FeetechArm.connect` sets the goal to the measured pose first, so nothing jerks), then every joint
ramps at `--speed` degrees per second. Torque stays on afterwards unless `--off` is given, and an
arm released at the zero pose falls: hold it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import ROBOT_DIR
from .feetech import ADDR_TORQUE_ENABLE, FeetechArm
from .panel import RobotDescription

logger = logging.getLogger(__name__)

HZ = 20.0
#: The gripper's zero: tick 2048, as every joint's.
GRIPPER_ZERO_TICK = 2048


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soarm-zero", description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", action="append", default=[],
                        help="serial port, repeatable (default: the one USB-serial device)")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--robot", type=Path, default=ROBOT_DIR, help="the folder with robot.yaml")
    parser.add_argument("--home", action="store_true",
                        help="go to robot.yaml's `home` instead of the zero pose")
    parser.add_argument("--speed", type=float, default=30.0, help="degrees per second (default 30)")
    parser.add_argument("--off", action="store_true", help="switch torque off once there (the arm goes limp)")
    return parser


def detect_port() -> str:
    from serial.tools.list_ports import comports

    devices = sorted(str(p.device) for p in comports() if getattr(p, "vid", None) is not None)
    if len(devices) != 1:
        raise SystemExit(f"soarm-zero: pass --port; USB-serial devices found: {', '.join(devices) or 'none'}")
    return devices[0]


def move(arm: FeetechArm, target: dict[str, float], speed: float) -> dict[str, float]:
    """Ramps every joint from where the arm is to `target` at `speed` deg/s; returns the measured pose."""
    cmd = dict(arm.read_joints())
    step = speed / HZ
    while max(abs(target[j] - cmd[j]) for j in target) > 0.05:
        for j in target:
            cmd[j] += max(-step, min(step, target[j] - cmd[j]))
        arm.send_joints(cmd)
        time.sleep(1 / HZ)
    time.sleep(0.5)
    return arm.read_joints()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    description = RobotDescription.load(args.robot)
    ports = args.port or [detect_port()]
    for port in ports:
        arm = FeetechArm(description.context(port, {"baudrate": args.baudrate}))
        arm.connect()
        try:
            target = {j: 0.0 for j in description.joints}
            if description.gripper:
                target[description.gripper] = arm._to_units(description.gripper, GRIPPER_ZERO_TICK)
            if args.home:
                target.update(description.home)
            start = arm.read_joints()
            print(f"{port}: from " + ", ".join(f"{j} {v:.1f}" for j, v in start.items()))
            end = move(arm, target, args.speed)
            print(f"{port}: now  " + ", ".join(f"{j} {v:.1f}" for j, v in end.items()))
            if args.off:
                for dev_id in arm.ids.values():
                    arm.bus.write(dev_id, ADDR_TORQUE_ENABLE, 0, 1)
                print(f"{port}: torque off")
        finally:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
