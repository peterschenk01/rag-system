import numpy as np
import pytest

import rag_system.retrieve as retrieve
from rag_system.index import FaissStore


class FakeIndex:
    def __init__(self, ids, scores):
        self._ids = np.array([ids], dtype="int64")
        self._scores = np.array([scores], dtype="float32")

    def search(self, q, k):
        return self._scores[:, :k], self._ids[:, :k]


def test_retrieve_maps_ids_to_chunks(monkeypatch):
    monkeypatch.setattr(retrieve, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))
    monkeypatch.setattr(retrieve.faiss, "normalize_L2", lambda x: None)

    store = FaissStore(index=FakeIndex(ids=[1, 0, -1], scores=[0.9, 0.5, 0.1]), chunks=["c0", "c1"])
    out = retrieve.retrieve(store, "q", k=3)

    assert out[0][0] == "c1"
    assert out[0][1] == pytest.approx(0.9, rel=1e-6)

    assert out[1][0] == "c0"
    assert out[1][1] == pytest.approx(0.5, rel=1e-6)

    assert len(out) == 2
