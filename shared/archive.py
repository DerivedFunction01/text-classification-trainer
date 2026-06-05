from __future__ import annotations

import os
import shutil
import subprocess
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from shared.paths import ARTIFACT_ROOT, CACHE_ROOT
ARCHIVE_NAME_TEMPLATE = "{subdir_name}_cache.zip"


def archive_path_for(subdir_name: str, artifact_root: Path = ARTIFACT_ROOT) -> Path:
    return artifact_root / ARCHIVE_NAME_TEMPLATE.format(subdir_name=subdir_name)


def _archive_extractor() -> tuple[list[str], str]:
    for candidate in ("7z", "7za", "unzip"):
        executable = shutil.which(candidate)
        if executable is not None:
            return [executable], candidate
    raise FileNotFoundError("No supported archive extractor found. Install 7z or unzip.")


def _extract_with_native_tool(archive_path: Path, destination_dir: Path) -> None:
    extractor, tool_name = _archive_extractor()
    destination_dir.mkdir(parents=True, exist_ok=True)

    if tool_name in {"7z", "7za"}:
        cmd = [*extractor, "x", "-y", f"-o{destination_dir}", str(archive_path)]
    else:
        cmd = [*extractor, "-o", str(archive_path), "-d", str(destination_dir)]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _has_tokenized_cache(tokenized_cache_dir: Path) -> bool:
    if not tokenized_cache_dir.exists():
        return False
    return any(tokenized_cache_dir.glob("*.parquet"))


def ensure_cache_archive_extracted(
    subdir_name: str,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    target_dir = cache_root / subdir_name
    tokenized_cache_dir = target_dir / "tokenized"
    if _has_tokenized_cache(tokenized_cache_dir):
        return target_dir

    archive_path = archive_path_for(subdir_name, artifact_root=artifact_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing cache archive: {archive_path}")

    cache_root = cache_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extracted_path = (cache_root / info.filename).resolve()
            if extracted_path != cache_root and cache_root not in extracted_path.parents:
                raise ValueError(f"Unsafe archive member path: {info.filename}")
        _extract_with_native_tool(archive_path, cache_root)

    if not _has_tokenized_cache(tokenized_cache_dir):
        raise RuntimeError(f"Archive extraction did not restore a usable cache: {tokenized_cache_dir}")
    return target_dir


def cache_has_archive(subdir_name: str, *, artifact_root: Path = ARTIFACT_ROOT) -> bool:
    return archive_path_for(subdir_name, artifact_root=artifact_root).exists()


def cache_has_tokenized_cache(subdir_name: str, *, cache_root: Path = CACHE_ROOT) -> bool:
    return _has_tokenized_cache(cache_root / subdir_name / "tokenized")


def zip_cache_subdir(
    subdir_name: str,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    tokenized_dir = cache_root / subdir_name / "tokenized"
    if not _has_tokenized_cache(tokenized_dir):
        raise FileNotFoundError(f"Tokenized cache directory does not exist or is empty: {tokenized_dir}")

    meta_path = tokenized_dir / "dataset.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tokenized cache metadata not found: {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        json.load(f)

    artifact_root.mkdir(parents=True, exist_ok=True)
    archive_base = artifact_root / f"{subdir_name}_cache"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=cache_root,
            base_dir=f"{subdir_name}/tokenized",
        )
    )
    return archive_path


def ensure_cache_archives_extracted(
    subdir_names: Iterable[str],
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
) -> list[Path]:
    subdir_names = list(subdir_names)
    if not subdir_names:
        return []

    max_workers = min(len(subdir_names), os.cpu_count() or 1)
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                ensure_cache_archive_extracted,
                subdir_name,
                artifact_root=artifact_root,
                cache_root=cache_root,
            ): subdir_name
            for subdir_name in subdir_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting caches", unit="cache"):
            results.append(future.result())
    return results
