from __future__ import annotations

from rag_app.config import STORAGE_DIR, DATA_PATH, EMBEDDING_MODEL
from rag_app.ingest import load_dataset
from rag_app.index import build_faiss_store, search, FaissStore
from rag_app.persist import load_store, save_store, store_exists
from rag_app.manifest import build_manifest, save_manifest, load_manifest, is_compatible


def get_expected_manifest() -> dict:
    return build_manifest(
        dataset_path=DATA_PATH,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=768,
        metric="cosine",
        chunking_strategy="one-line",
    )


def build_and_persist_store(dataset: list[str]) -> FaissStore:
    store = build_faiss_store(dataset)

    manifest = build_manifest(
        dataset_path=DATA_PATH,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=store.index.d,
        metric="cosine",
        chunking_strategy="one-line",
    )

    save_store(store, STORAGE_DIR)
    save_manifest(STORAGE_DIR, manifest)
    return store


def get_or_build_store(dataset: list[str]) -> FaissStore:
    expected = get_expected_manifest()

    if store_exists(STORAGE_DIR):
        try:
            stored = load_manifest(STORAGE_DIR)
        except FileNotFoundError:
            print("Manifest missing! Rebuilding FAISS store.")
        else:
            if is_compatible(stored=stored, expected=expected):
                print("Manifest matches! Using persisted FAISS store.")
                return load_store(STORAGE_DIR)

            print("Persisted FAISS store incompatible! Rebuilding...")

    return build_and_persist_store(dataset)


def main():
    dataset = load_dataset()
    store = get_or_build_store(dataset)

    query = "How much do cats sleep?"
    hits = search(store, query, k=5)


if __name__ == "__main__":
    main()