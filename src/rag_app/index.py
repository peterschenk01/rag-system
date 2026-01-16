from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"


@dataclass
class FaissStore:
    index: faiss.Index
    chunks: List[str]


def embed_texts(texts: List[str]) -> np.ndarray:
    resp = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    embs = np.array(resp["embeddings"], dtype="float32")  # (N, D)
    return embs


def build_faiss_index(chunks: List[str]) -> FaissStore:
    print("Building FAISS index...")

    vectors = embed_texts(chunks)  # (N, D)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No embeddings returned.")

    dim = vectors.shape[1]

    # cosine similarity
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)

    index.add(vectors)
    print(f"Added {index.ntotal} vectors to FAISS index.")

    return FaissStore(index=index, chunks=chunks)


def search(store: FaissStore, query: str, k: int = 5) -> List[Tuple[str, float]]:
    print(f"Searching store with query: \"{query}\"")
    print(f"Retrieving top {k} results...")

    q = embed_texts([query])  # (1, D)
    faiss.normalize_L2(q)

    scores, ids = store.index.search(q, k)

    results: List[Tuple[str, float]] = []

    for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
        if idx == -1:
            continue
        results.append((store.chunks[idx], float(score)))

    print(f"Retrieved {len(results)} result(s).")

    for chunk, score in results:
        print(f"[{score}] {chunk}")
    
    return results
