"""CLI entry-point for archive-enterprise-analytics.

Usage::

    python -m archive_analytics build          # build processed tables
    python -m archive_analytics train          # train risk models
    python -m archive_analytics serve          # launch Streamlit dashboard
    python -m archive_analytics build --force  # force rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archive_analytics",
        description="Archive Enterprise Analytics pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Build processed data tables.")
    build.add_argument("--force", action="store_true", help="Force rebuild.")

    train = sub.add_parser("train", help="Train all risk models.")
    train.add_argument("--force", action="store_true", help="Force retrain.")

    sub.add_parser("serve", help="Launch the Streamlit dashboard.")

    return parser


def main(argv: list[str] | None = None) -> int:
    from .settings import configure_logging

    configure_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        from .data import build_processed_assets

        paths = build_processed_assets(force=args.force)
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0

    if args.command == "train":
        from .data import build_processed_assets
        from .modeling import train_all_targets

        build_processed_assets(force=False)
        metrics = train_all_targets(force=args.force)
        import json

        print(json.dumps(metrics, indent=2, default=str))
        return 0

    if args.command == "serve":
        import subprocess

        app_path = str(
            __import__("pathlib").Path(__file__).resolve().parents[2] / "app.py"
        )
        return subprocess.call([sys.executable, "-m", "streamlit", "run", app_path])

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
