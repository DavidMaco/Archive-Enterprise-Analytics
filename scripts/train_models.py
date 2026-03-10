"""Train all risk-scoring models.

Usage::

    python -m archive_analytics train          # preferred
    python scripts/train_models.py             # legacy
"""

from __future__ import annotations

import json

from archive_analytics.data import build_processed_assets
from archive_analytics.modeling import train_all_targets
from archive_analytics.settings import configure_logging


def main() -> None:
    configure_logging()
    build_processed_assets(force=False)
    metrics = train_all_targets(force=True)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
