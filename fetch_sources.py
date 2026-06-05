from __future__ import annotations

import argparse
from pathlib import Path

from shared.archive import (
    cache_has_archive,
    cache_has_tokenized_cache,
    ensure_cache_archive_extracted,
    zip_cache_subdir,
)
from shared.config import load_json_config
from shared.data_pipeline import build_tokenized_dataset_cache
from shared.paths import ARTIFACT_ROOT, CACHE_ROOT
from tqdm.auto import tqdm


def discover_cache_subdirs() -> list[str]:
    if not CACHE_ROOT.exists():
        return []

    subdirs = []
    for tokenized_dir in CACHE_ROOT.glob("*/tokenized"):
        if any(tokenized_dir.glob("*.parquet")):
            subdirs.append(tokenized_dir.parent.name)
    return sorted(set(subdirs))


def build_all_sources() -> list[Path]:
    subdirs = discover_cache_subdirs()
    if not subdirs:
        print("No tokenized caches found under .cache to archive.")
        return []

    results: list[Path] = []
    for subdir_name in tqdm(subdirs, desc="Archiving caches", unit="cache"):
        build_cache_subdir(subdir_name)
        results.append(ARTIFACT_ROOT / f"{subdir_name}_cache.zip")
    return results


def build_cache_subdir(subdir_name: str) -> None:
    config_path = Path("classification/.config.json")
    if not config_path.exists():
        config_path = Path("classification/config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            "No classification config found. Expected classification/.config.json or classification/config.json"
        )

    config = load_json_config(config_path)
    dataset_cfg = config.get("dataset", {})
    cache_dir = dataset_cfg.get("cache_dir")
    if cache_dir is None:
        raise ValueError("classification config does not define dataset.cache_dir")

    expected_subdir = Path(cache_dir).parent.name
    if expected_subdir != subdir_name:
        raise ValueError(
            f"Config cache_dir {cache_dir!r} does not match requested subdir {subdir_name!r}"
        )

    print(f"Building tokenized cache for {subdir_name} from {config_path} ...")
    build_tokenized_dataset_cache(config)


def validate_tokenized_cache(subdir_name: str) -> Path:
    tokenized_dir = CACHE_ROOT / subdir_name / "tokenized"
    if not tokenized_dir.exists():
        raise FileNotFoundError(f"Tokenized cache directory does not exist: {tokenized_dir}")
    return tokenized_dir


def reconcile_cache_subdir(subdir_name: str) -> Path:
    has_cache = cache_has_tokenized_cache(subdir_name)
    has_archive = cache_has_archive(subdir_name)

    if has_cache and has_archive:
        print(f"{subdir_name}: cache and archive already exist, skipping")
        return ARTIFACT_ROOT / f"{subdir_name}_cache.zip"

    if has_cache and not has_archive:
        return zip_cache_subdir(subdir_name)

    if has_archive and not has_cache:
        print(f"{subdir_name}: tokenized cache missing, restoring from archive")
        ensure_cache_archive_extracted(subdir_name)
        return ARTIFACT_ROOT / f"{subdir_name}_cache.zip"

    print(f"{subdir_name}: cache and archive both missing, rebuilding cache locally")
    build_cache_subdir(subdir_name)
    return ARTIFACT_ROOT / f"{subdir_name}_cache.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package cached tokenized datasets into zip archives.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build datasets before packaging them. Omit this to only zip existing tokenized caches.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        help="Cache subdirectories to package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        build_all_sources()
    subdirs = args.subdirs if args.subdirs is not None else discover_cache_subdirs()
    for subdir_name in tqdm(subdirs, desc="Archiving caches", unit="cache"):
        reconcile_cache_subdir(subdir_name)


if __name__ == "__main__":
    main()
