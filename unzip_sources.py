from __future__ import annotations

import argparse

from shared.archive import ensure_cache_archives_extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached training sources from artifacts/*.zip")
    parser.add_argument(
        "--subdirs",
        nargs="+",
        help="Cache subdirectories to extract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_cache_archives_extracted(args.subdirs)


if __name__ == "__main__":
    main()
