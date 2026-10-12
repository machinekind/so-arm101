from __future__ import annotations

import pytest

from soarm.cli import COMMANDS, NOT_IMPLEMENTED, main


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_is_not_implemented(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([command]) == NOT_IMPLEMENTED == 2
    assert "not implemented" in capsys.readouterr().err


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
