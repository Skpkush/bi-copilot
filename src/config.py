"""Central configuration and path helpers for BI Copilot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "business.db"
EMBEDDING_MARKER = CHROMA_DIR / "embedding_backend.txt"
QUALITY_REPORT_PATH = DATA_DIR / "quality_report.json"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.jsonl"

# The sample warehouse ends on 2026-09-30, so 2026-Q3 is the latest complete
# quarter. Keeping this fixed makes the demo reproducible.
DATASET_END_DATE = "2026-09-30"
LATEST_QUARTER = "Q3 2026"
LATEST_QUARTER_START = "2026-07-01"
LATEST_QUARTER_END = "2026-09-30"

COLLECTION_NAME = "business_documents"
EMBEDDING_DIMENSIONS = 768
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_SQL_ROWS = 200
MAX_PREVIEW_ROWS = 50

load_dotenv(PROJECT_ROOT / ".env")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a permissive boolean flag from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def openai_api_key() -> str | None:
    """Return a usable API key, treating empty strings and placeholders as unset."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.lower().startswith(("your-", "sk-your", "<")):
        return None
    return key


def offline_mode() -> bool:
    """Offline demo mode is enabled explicitly, or implicitly when no key exists."""
    return env_flag("BI_COPILOT_OFFLINE") or openai_api_key() is None


def llm_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def embedding_model() -> str:
    return (
        os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def require_db() -> Path:
    """Fail with an actionable message when setup has not been run yet."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python -m src.generate_data "
            "&& python -m src.load_data"
        )
    return DB_PATH
