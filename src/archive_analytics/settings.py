"""Application configuration and logging setup.

The raw-data directory is resolved from the ``ARCHIVE_ANALYTICS_RAW_DIR``
environment variable.  If the variable is unset, a sensible project-relative
default (``<project_root>/data/raw``) is used instead of a hard-coded Windows
path.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels above this file: src/archive_analytics/settings.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level: int | str = logging.INFO) -> None:
    """Set up structured logging for the entire package.

    Call once at process start-up (e.g. in ``__main__.py`` or ``app.py``).
    """
    root = logging.getLogger("archive_analytics")
    if root.handlers:
        return  # already configured
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Default raw-data directory
# ---------------------------------------------------------------------------


def _default_raw_dir() -> Path:
    """Resolve the raw-data directory from the environment or project tree."""
    env = os.getenv("ARCHIVE_ANALYTICS_RAW_DIR")
    if env:
        return Path(env)
    # Fallback: look for a ``data/raw`` folder next to the project root
    candidate = PROJECT_ROOT / "data" / "raw"
    return candidate


# ---------------------------------------------------------------------------
# AppConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = field(default_factory=_default_raw_dir)
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    ui_mutations_enabled: bool = field(
        default_factory=lambda: _env_flag("ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS")
    )

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        for d in (self.processed_dir, self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Raise ``FileNotFoundError`` if the raw-data directory is missing."""
        if not self.raw_data_dir.is_dir():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}\n"
                "Set the ARCHIVE_ANALYTICS_RAW_DIR environment variable to "
                "the folder containing the archive parquet files."
            )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the global singleton ``AppConfig``, creating it on first call."""
    global _CONFIG  # noqa: PLW0603
    if _CONFIG is None:
        _CONFIG = AppConfig()
    _CONFIG.ensure_directories()
    return _CONFIG
