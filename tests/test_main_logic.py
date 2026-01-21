import rag_system.main as main
from rag_system.index import FaissStore


class DummyIndex:
    d = 768


def test_get_or_build_store_uses_persisted_when_manifest_matches(monkeypatch):
    dummy_store = FaissStore(index=DummyIndex(), chunks=["a"])

    monkeypatch.setattr(main, "store_exists", lambda _dir: True)
    monkeypatch.setattr(main, "load_manifest", lambda _dir: {"ok": True})
    monkeypatch.setattr(main, "get_expected_manifest", lambda: {"ok": True})
    monkeypatch.setattr(main, "is_compatible", lambda stored, expected: True)
    monkeypatch.setattr(main, "load_store", lambda _dir: dummy_store)

    def fail(*_args, **_kwargs):
        raise AssertionError

    monkeypatch.setattr(main, "build_and_persist_store", fail)

    store = main.get_or_build_store(["x"])
    assert store is dummy_store


def test_get_or_build_store_rebuilds_when_missing_manifest(monkeypatch):
    dummy_store = FaissStore(index=DummyIndex(), chunks=["a"])

    monkeypatch.setattr(main, "store_exists", lambda _dir: True)

    def raise_fn(_dir):
        raise FileNotFoundError

    monkeypatch.setattr(main, "load_manifest", raise_fn)

    monkeypatch.setattr(main, "build_and_persist_store", lambda ds: dummy_store)
    store = main.get_or_build_store(["x"])
    assert store is dummy_store
