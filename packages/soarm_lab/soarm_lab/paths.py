"""Where the lab keeps its data: the policies it trains.

Policies trained here go to `workspace/policies`. Run from a checkout of the so-arm101 repository,
that is relative to the repository root, so a command started in another folder finds the same
policies; installed as a package (no checkout), relative to the current folder. `SOARM_LAB_POLICIES`
names another folder. Paths the user passes (flags) are taken as given.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The repository root when the package runs from a checkout: `packages/soarm_lab/soarm_lab/paths.py`.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: Environment variable naming the folder of trained policies.
POLICIES_ENV = "SOARM_LAB_POLICIES"
POLICIES_REL = Path("workspace") / "policies"


def source_checkout() -> bool:
    """Does the package run from a checkout of the repository (`uv sync`, `pip install -e`)?"""
    return (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "packages" / "soarm_lab").is_dir()


def data_path(rel: str | Path) -> Path:
    """A default data path: relative to the repository root in a checkout, else to the current folder."""
    rel = Path(rel)
    if rel.is_absolute() or not source_checkout():
        return rel
    try:
        if Path.cwd().resolve() == REPO_ROOT:
            return rel
    except OSError:  # pragma: no cover - the current folder was removed
        pass
    return REPO_ROOT / rel


def policies_dir() -> Path:
    """The folder of trained policies: `SOARM_LAB_POLICIES`, else `workspace/policies` (`data_path`)."""
    value = os.environ.get(POLICIES_ENV, "").strip()
    return Path(value).expanduser() if value else data_path(POLICIES_REL)
