"""Verify .env wiring picks up keys from a local file without manual shell export."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_load_reads_keys_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A test-only env var set in a .env file should be loaded into os.environ."""
    from src import _env

    var_name = "_CITATION_REWARD_HACKING_TEST_VAR"
    monkeypatch.delenv(var_name, raising=False)

    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{var_name}=hello_from_dotenv\n", encoding="utf-8")

    assert var_name not in os.environ, "precondition: var must not be set in shell"

    found = _env.load(dotenv)

    assert found is True
    assert os.environ[var_name] == "hello_from_dotenv"


def test_load_does_not_override_shell_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shell-set vars take precedence over .env (override=False contract)."""
    from src import _env

    var_name = "_CITATION_REWARD_HACKING_TEST_VAR2"
    monkeypatch.setenv(var_name, "shell_value")

    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{var_name}=dotenv_value\n", encoding="utf-8")

    _env.load(dotenv)

    assert os.environ[var_name] == "shell_value"


def test_load_returns_false_when_no_dotenv(tmp_path: Path) -> None:
    """Absent .env file is not an error — just a no-op."""
    from src import _env

    missing = tmp_path / "nonexistent.env"
    assert _env.load(missing) is False
