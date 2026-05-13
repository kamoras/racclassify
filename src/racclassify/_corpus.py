"""ChromaDB reference corpus for kNN classification.

Optional tier between prototype-matching and the learning store.
As documents are classified and embedded, they accumulate in ChromaDB
and subsequent documents are classified by analogy to the k nearest
neighbors rather than by raw prototype similarity.

This is the retrieval-augmented classification pattern from Lewis et al.
(2020), adapted for supervised (not generative) retrieval.

Install the optional dependency to enable this tier:
    pip install racclassify[chromadb]
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_COLLECTION_PREFIX = "racclassify_"


class ReferenceCorpus:
    """ChromaDB-backed kNN reference corpus.

    Args:
        persist_path: Directory where ChromaDB stores its data.
            Pass ``None`` to use an in-memory ephemeral client.
        namespace: Collection name suffix, allowing multiple classifiers
            to share the same ChromaDB instance.
        k: Number of nearest neighbours to consult per query.
        min_similarity: Neighbours below this cosine similarity are ignored.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        namespace: str = "default",
        k: int = 7,
        min_similarity: float = 0.30,
    ) -> None:
        self._k = k
        self._min_sim = min_similarity
        self._collection_name = f"{_COLLECTION_PREFIX}{namespace}"
        self._client = self._make_client(persist_path)
        self._col = self._get_or_create_collection()

    @staticmethod
    def _make_client(persist_path: str | None) -> object:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is required for ReferenceCorpus. "
                "Install it with: pip install racclassify[chromadb]"
            ) from e
        if persist_path:
            return chromadb.PersistentClient(path=persist_path)
        return chromadb.EphemeralClient()

    def _get_or_create_collection(self) -> object:

        return self._client.get_or_create_collection(  # type: ignore[attr-defined]
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def classify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (label, confidence) from kNN vote, or (None, 0.0) if corpus is thin."""
        result = self._col.query(  # type: ignore[attr-defined]
            query_embeddings=[embedding.tolist()],
            n_results=min(self._k, max(1, self._col.count())),  # type: ignore[attr-defined]
            include=["metadatas", "distances"],
        )
        if not result or not result["ids"] or not result["ids"][0]:
            return None, 0.0

        votes: Counter[str] = Counter()
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            # ChromaDB cosine distance = 1 − similarity
            sim = max(0.0, 1.0 - dist)
            if sim >= self._min_sim:
                label = meta.get("label", "")
                if label:
                    votes[label] += sim

        if not votes:
            return None, 0.0

        best, best_weight = votes.most_common(1)[0]
        confidence = best_weight / sum(votes.values())
        return best, confidence

    def add(self, doc_id: str, embedding: np.ndarray, label: str) -> None:
        """Add a classified document to the corpus."""
        self._col.upsert(  # type: ignore[attr-defined]
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[{"label": label}],
        )

    def remove(self, doc_id: str) -> None:
        """Remove a document from the corpus."""
        try:
            self._col.delete(ids=[doc_id])  # type: ignore[attr-defined]
        except Exception:
            pass

    def size(self) -> int:
        return self._col.count()  # type: ignore[attr-defined]

    def label_distribution(self) -> dict[str, int]:
        result = self._col.get(include=["metadatas"])  # type: ignore[attr-defined]
        if not result or not result["metadatas"]:
            return {}
        return dict(Counter(m.get("label", "") for m in result["metadatas"]))
