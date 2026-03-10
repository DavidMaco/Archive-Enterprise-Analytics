"""Build processed data tables from raw archive files.

Usage::

    python -m archive_analytics build          # preferred
    python scripts/build_data.py               # legacy
"""

from __future__ import annotations

from archive_analytics.data import build_processed_assets
from archive_analytics.settings import configure_logging


def main() -> None:
    configure_logging()
    paths = build_processed_assets(force=True)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
