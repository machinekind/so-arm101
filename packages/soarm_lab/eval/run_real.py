"""Evaluate a strategy on the real arms. Stub.

It will run a policy or planner through the twin's safety supervisor on hardware and write
results to eval/results/. It replaces the real-arm path of the twin's `twin/rl/runner.py` (#146).
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("run_real follows twin twin/rl/runner.py (#146)")


if __name__ == "__main__":
    raise SystemExit(main())
