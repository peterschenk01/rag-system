from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag_system.config import DATA_PATH, EMBEDDING_MODEL, STORAGE_DIR
from rag_system.generate import generate
from rag_system.index import FaissStore, build_faiss_store
from rag_system.ingest import load_dataset
from rag_system.logging_config import setup_logging
from rag_system.manifest import (
    build_manifest,
    is_compatible,
    load_manifest,
    save_manifest,
)
from rag_system.persist import load_store, save_store, store_exists
from rag_system.retrieve import retrieve

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_manifest(*, dataset_path: Path, embedding_dim: int) -> dict[str, Any]:
    return build_manifest(
        dataset_path=dataset_path,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=embedding_dim,
        metric="cosine",
        chunking_strategy="one-line",
    )


def get_expected_manifest() -> dict[str, Any]:
    return make_manifest(dataset_path=DATA_PATH, embedding_dim=768)


def build_and_persist_store(dataset: list[str]) -> FaissStore:
    store = build_faiss_store(dataset)

    manifest = make_manifest(dataset_path=DATA_PATH, embedding_dim=store.index.d)

    save_store(store, STORAGE_DIR)
    save_manifest(STORAGE_DIR, manifest)
    return store


def get_or_build_store(dataset: list[str]) -> FaissStore:
    if store_exists(STORAGE_DIR):
        try:
            stored = load_manifest(STORAGE_DIR)
        except FileNotFoundError:
            logger.info("Manifest missing! Rebuilding FAISS store.")
            return build_and_persist_store(dataset)

        expected = get_expected_manifest()

        if is_compatible(stored=stored, expected=expected):
            logger.info("Manifest matches! Using persisted FAISS store.")
            return load_store(STORAGE_DIR)

        logger.info("Persisted FAISS store incompatible! Rebuilding...")

    return build_and_persist_store(dataset)


def main():
    dataset = load_dataset()
    store = get_or_build_store(dataset)

    while True:
        input_query = input("Ask me a question (or 'quit'): ")

        if not input_query:
            continue
        if input_query.strip().lower() in {"exit", "quit", "q"}:
            break

        retrieved_knowledge = retrieve(store, input_query)

        for chunk, similarity in retrieved_knowledge:
            logger.info(f" - (similarity: {similarity:.2f}) {chunk}")

        generate(input_query=input_query, context=retrieved_knowledge)


if __name__ == "__main__":
    main()
