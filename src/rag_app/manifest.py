from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


def file_hash(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def build_manifest(
    *,
    dataset_path: Path,
    embedding_model: str,
    embedding_dim: int,
    metric: str,
    chunking_strategy: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": file_hash(dataset_path),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "metric": metric,
        "chunking": {
            "strategy": chunking_strategy,
        },
    }


def save_manifest(storage_dir: Path, manifest: Dict[str, Any]) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / MANIFEST_FILENAME

    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_manifest(storage_dir: Path) -> Dict[str, Any]:
    path = storage_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError("Manifest file not found")

    return json.loads(path.read_text(encoding="utf-8"))


def is_compatible(
    *,
    stored: Dict[str, Any],
    expected: Dict[str, Any],
) -> bool:
    if stored.get("schema_version") != expected.get("schema_version"):
        return False

    return (
        stored.get("dataset_hash") == expected.get("dataset_hash")
        and stored.get("embedding_model") == expected.get("embedding_model")
        and stored.get("embedding_dim") == expected.get("embedding_dim")
        and stored.get("metric") == expected.get("metric")
        and stored.get("chunking") == expected.get("chunking")
    )
