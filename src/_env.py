"""Load environment variables from .env at repo root.

Entry-point scripts (elicit, extract_citations, pilot*) import this module
once near the top so ANTHROPIC_API_KEY / OPENALEX_MAILTO / COURTLISTENER_API_TOKEN
are picked up from .env without requiring callers to export them in the shell.

Shell-exported values take precedence (override=False) — a stale .env never
overwrites a deliberately-set env var.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = REPO_ROOT / ".env"


def load(path: Path = DOTENV_PATH, *, override: bool = False) -> bool:
    """Load env vars from `path`. Returns True if the file was found and parsed."""
    return load_dotenv(path, override=override)


load()
