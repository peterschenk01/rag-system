from pathlib import Path

import rag_system.ingest as ingest


def test_chunk_dataset_one_line_per_chunk():
    data = ["a", "b", "c"]
    assert ingest.chunk_dataset(data) == ["a", "b", "c"]


def test_ensure_data_exists_skips_download_if_present(monkeypatch, tmp_path: Path):
    fake_path = tmp_path / "cat-facts.txt"
    fake_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(ingest, "DATA_PATH", fake_path)

    called = {"n": 0}

    def fake_urlretrieve(url, path):
        called["n"] += 1

    monkeypatch.setattr(ingest.urllib.request, "urlretrieve", fake_urlretrieve)

    ingest.ensure_data_exists()
    assert called["n"] == 0


def test_load_dataset_strips_empty_lines(monkeypatch, tmp_path: Path):
    fake_path = tmp_path / "cat-facts.txt"
    fake_path.write_text(" a \n\nb\n  \n", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_PATH", fake_path)

    monkeypatch.setattr(ingest, "ensure_data_exists", lambda: None)

    chunks = ingest.load_dataset()
    assert chunks == ["a", "b"]
