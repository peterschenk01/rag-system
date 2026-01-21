import numpy as np

import rag_system.index as index


def test_embed_texts_calls_ollama_and_returns_float32(monkeypatch):
    def fake_embed(model, input):
        return {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}

    monkeypatch.setattr(index.ollama, "embed", fake_embed)
    monkeypatch.setattr(index, "EMBEDDING_MODEL", "fake-model")

    embs = index.embed_texts(["a", "b"])
    assert embs.dtype == np.float32
    assert embs.shape == (2, 2)


def test_build_faiss_store_raises_on_empty_embeddings(monkeypatch):
    def fake_embed_texts(_texts):
        return np.array([], dtype="float32")

    monkeypatch.setattr(index, "embed_texts", fake_embed_texts)

    try:
        index.build_faiss_store(["x"])
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "No embeddings returned" in str(e)
