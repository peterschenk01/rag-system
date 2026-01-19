from __future__ import annotations

import logging
from dataclasses import dataclass

import faiss
import numpy as np
import ollama

from rag_system.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@dataclass
class FaissStore:
    index: faiss.Index
    chunks: list[str]


def embed_texts(texts: list[str]) -> np.ndarray:
    resp = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    embs = np.array(resp["embeddings"], dtype="float32")  # (N, D)
    return embs


def build_faiss_store(chunks: list[str]) -> FaissStore:
    logger.info("Building FAISS store...")

    vectors = embed_texts(chunks)  # (N, D)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No embeddings returned.")

    dim = vectors.shape[1]

    # cosine similarity
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    return FaissStore(index=index, chunks=chunks)
