"""Bootstrap: load .env from repo root on first import so entry-point scripts
and library modules alike see ANTHROPIC_API_KEY / OPENALEX_MAILTO /
COURTLISTENER_API_TOKEN / etc. without manual shell export.

Shell-exported values win (override=False); see src/_env.py.
"""
from . import _env  # noqa: F401
