from pathlib import Path

from rag_system.manifest import build_manifest, file_hash, is_compatible


def test_file_hash_is_stable(tmp_path: Path):
    p = tmp_path / "data.txt"
    p.write_text("abc", encoding="utf-8")
    h1 = file_hash(p)
    h2 = file_hash(p)
    assert h1 == h2


def test_is_compatible_true_when_equal(tmp_path: Path):
    p = tmp_path / "data.txt"
    p.write_text("abc", encoding="utf-8")

    m1 = build_manifest(
        dataset_path=p,
        embedding_model="model",
        embedding_dim=3,
        metric="cosine",
        chunking_strategy="one-line",
    )
    m2 = dict(m1)
    assert is_compatible(stored=m1, expected=m2) is True


def test_is_compatible_false_on_dim_mismatch(tmp_path: Path):
    p = tmp_path / "data.txt"
    p.write_text("abc", encoding="utf-8")

    stored = build_manifest(
        dataset_path=p,
        embedding_model="model",
        embedding_dim=3,
        metric="cosine",
        chunking_strategy="one-line",
    )
    expected = dict(stored)
    expected["embedding_dim"] = 4
    assert is_compatible(stored=stored, expected=expected) is False
