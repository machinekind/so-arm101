"""The `soarm` command: servo and arm tools that run without the twin.

Every subcommand is a placeholder for now. It prints "not implemented" and exits with code 2.
"""

from __future__ import annotations

import argparse
import sys

NOT_IMPLEMENTED = 2

COMMANDS: dict[str, str] = {
    "scan": "ping servo IDs on a port and list the ones that answer",
    "inspect": "read position, voltage, temperature and EEPROM limits of one servo",
    "limits": "read the EEPROM angle limits of every servo",
    "torque-off": "release torque on every servo (the arm drops)",
    "bridge": "expose the arm's serial port over TCP",
    "leader": "print the positions of a hand-guided leader arm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soarm", description="SO-101 servo and arm tools.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, help_text in COMMANDS.items():
        cmd = sub.add_parser(name, help=help_text, description=help_text)
        cmd.add_argument("--port", help="serial port, e.g. /dev/ttyACM0, /dev/cu.usbmodem*, COM5")
    sub.choices["inspect"].add_argument("--id", type=int, default=1, help="servo ID (default 1)")
    sub.choices["bridge"].add_argument("--listen", default="127.0.0.1:5555", help="host:port to listen on")
    sub.choices["bridge"].add_argument("--allow", default="", help="comma-separated client IPs; empty = any")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"soarm {args.command}: not implemented", file=sys.stderr)
    return NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
