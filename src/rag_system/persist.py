import json
import logging
from pathlib import Path

import faiss

from rag_system.index import FaissStore

logger = logging.getLogger(__name__)


def store_exists(storage_dir: Path) -> bool:
    return (storage_dir / "index.faiss").exists() and (
        storage_dir / "chunks.json"
    ).exists()


def save_store(store: FaissStore, storage_dir: Path) -> None:
    logger.info(f"Storing FAISS store to: {storage_dir}")

    storage_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(store.index, str(storage_dir / "index.faiss"))

    (storage_dir / "chunks.json").write_text(
        json.dumps(store.chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_store(storage_dir: Path) -> FaissStore:
    logger.info(f"Loading FAISS store from: {storage_dir}")

    index = faiss.read_index(str(storage_dir / "index.faiss"))

    chunks = json.loads((storage_dir / "chunks.json").read_text(encoding="utf-8"))

    logger.info("Successfully loaded FAISS store.")

    return FaissStore(index=index, chunks=chunks)
