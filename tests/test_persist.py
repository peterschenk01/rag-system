import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag_system.persist as persist


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "faiss_store"


def test_store_exists_false_when_dir_missing(storage_dir: Path):
    assert persist.store_exists(storage_dir) is False


def test_store_exists_false_when_one_file_missing(storage_dir: Path):
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "index.faiss").write_bytes(b"dummy")
    assert persist.store_exists(storage_dir) is False

    (storage_dir / "index.faiss").unlink()
    (storage_dir / "chunks.json").write_text("[]", encoding="utf-8")
    assert persist.store_exists(storage_dir) is False


def test_save_store_writes_index_and_chunks(monkeypatch, storage_dir: Path):
    fake_index = object()
    chunks = [{"id": "c1", "text": "hello"}, {"id": "c2", "text": "world"}]

    store = SimpleNamespace(index=fake_index, chunks=chunks)

    write_calls = {}

    def fake_write_index(index, path):
        write_calls["index"] = index
        write_calls["path"] = path
        Path(path).write_bytes(b"FAISS_INDEX")

    monkeypatch.setattr(persist.faiss, "write_index", fake_write_index)

    persist.save_store(store, storage_dir)

    assert (storage_dir / "index.faiss").exists()
    assert (storage_dir / "chunks.json").exists()

    assert write_calls["index"] is fake_index
    assert write_calls["path"] == str(storage_dir / "index.faiss")

    saved_chunks = json.loads((storage_dir / "chunks.json").read_text(encoding="utf-8"))
    assert saved_chunks == chunks


def test_load_store_reads_index_and_chunks(monkeypatch, storage_dir: Path):
    storage_dir.mkdir(parents=True, exist_ok=True)

    chunks = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
    (storage_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (storage_dir / "index.faiss").write_bytes(b"FAISS_INDEX")

    fake_index = object()

    def fake_read_index(path):
        assert path == str(storage_dir / "index.faiss")
        return fake_index

    monkeypatch.setattr(persist.faiss, "read_index", fake_read_index)

    created = {}

    def fake_faiss_store(*, index, chunks):
        created["index"] = index
        created["chunks"] = chunks
        return SimpleNamespace(index=index, chunks=chunks)

    monkeypatch.setattr(persist, "FaissStore", fake_faiss_store)

    loaded = persist.load_store(storage_dir)

    assert loaded.index is fake_index
    assert loaded.chunks == chunks
    assert created["index"] is fake_index
    assert created["chunks"] == chunks


def test_roundtrip_save_then_load_with_patched_faiss(monkeypatch, storage_dir: Path):
    fake_index = object()
    chunks = [{"id": "x", "text": "αβγ"}]

    store = SimpleNamespace(index=fake_index, chunks=chunks)

    def fake_write_index(index, path):
        Path(path).write_text("sentinel", encoding="utf-8")

    def fake_read_index(path):
        assert Path(path).read_text(encoding="utf-8") == "sentinel"
        return fake_index

    monkeypatch.setattr(persist.faiss, "write_index", fake_write_index)
    monkeypatch.setattr(persist.faiss, "read_index", fake_read_index)

    monkeypatch.setattr(persist, "FaissStore", lambda *, index, chunks: SimpleNamespace(index=index, chunks=chunks))

    persist.save_store(store, storage_dir)
    loaded = persist.load_store(storage_dir)

    assert loaded.index is fake_index
    assert loaded.chunks == chunks
