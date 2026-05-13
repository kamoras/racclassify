"""Tests for the SQLite learning store."""

import pytest

from racclassify._store import LearningStore


@pytest.fixture
def store(tmp_path) -> LearningStore:
    return LearningStore(tmp_path / "test.db")


def test_lookup_miss(store):
    assert store.lookup("nonexistent") is None


def test_record_and_lookup(store):
    store.record("doc-1", label="HEALTHCARE", confidence=0.9, text="Medicare bill")
    result = store.lookup("doc-1")
    assert result is not None
    assert result[0] == "HEALTHCARE"
    assert result[1] == pytest.approx(0.9)


def test_record_update(store):
    store.record("doc-1", label="DEFENSE", confidence=0.7)
    store.record("doc-1", label="TAXES", confidence=1.0)
    result = store.lookup("doc-1")
    assert result[0] == "TAXES"
    assert result[1] == pytest.approx(1.0)


def test_delete(store):
    store.record("doc-2", label="ENVIRONMENT", confidence=0.8)
    store.delete("doc-2")
    assert store.lookup("doc-2") is None


def test_delete_nonexistent_does_not_raise(store):
    store.delete("does-not-exist")


def test_stats_empty(store):
    info = store.stats()
    assert info["total"] == 0
    assert info["distribution"] == {}


def test_stats_with_records(store):
    store.record("a", label="HEALTHCARE", confidence=0.9)
    store.record("b", label="HEALTHCARE", confidence=0.8)
    store.record("c", label="DEFENSE", confidence=0.7)
    info = store.stats()
    assert info["total"] == 3
    assert info["distribution"]["HEALTHCARE"] == 2
    assert info["distribution"]["DEFENSE"] == 1


def test_namespace_isolation(tmp_path):
    store_a = LearningStore(tmp_path / "shared.db", namespace="a")
    store_b = LearningStore(tmp_path / "shared.db", namespace="b")
    store_a.record("doc-1", label="HEALTHCARE", confidence=0.9)
    assert store_b.lookup("doc-1") is None
    assert store_a.lookup("doc-1") is not None


def test_concurrent_writes(tmp_path):
    """LearningStore must not raise under concurrent access."""
    import threading

    store = LearningStore(tmp_path / "concurrent.db")
    errors: list[Exception] = []

    def write(i: int) -> None:
        try:
            store.record(f"doc-{i}", label="HEALTHCARE", confidence=0.9)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.stats()["total"] == 20
