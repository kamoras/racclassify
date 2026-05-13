"""Core Classifier implementation.

Classification tiers (highest priority first):
  1. Exact-match learning store  — previously seen doc_id, instant
  2. kNN reference corpus        — ChromaDB, optional, grows over time
  3. Nearest-centroid prototype  — zero-shot, always available
  4. Augmented re-embed          — retries with context prefix for low-confidence cases

Tier 1 requires a store_path. Tier 2 requires chromadb installed and a
corpus_path. Tiers 3 and 4 always run.

Academic grounding
------------------
Nearest-centroid (tier 3) follows Rocchio (1971) and Manning, Raghavan
& Schütze (2008, Ch. 14). kNN in embedding space (tier 2) follows Cover
& Hart (1967) with the distance-weighted vote variant from Dudani (1976).
Experience replay via the learning store (tier 1) follows Lin (1992).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._embeddings import EncoderFn, embed_batch, embed_one

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_AUGMENT_THRESHOLD = 0.18
DEFAULT_MULTI_SECONDARY_THRESHOLD = 0.20
DEFAULT_MULTI_GAP_RATIO = 0.70
DEFAULT_KNN_CONFIDENCE = 0.45


@dataclass
class ClassificationResult:
    """Single-label classification result."""
    label: str
    confidence: float


class Classifier:
    """Zero-shot text classifier using semantic prototypes with adaptive learning.

    Describe each category in plain English — no labeled training data
    required to start. Accuracy improves as you call ``record()`` to
    accumulate real examples that seed the kNN reference corpus.

    Args:
        categories: Mapping of ``{label: natural_language_description}``.
            Descriptions should capture the semantic space of each category,
            not just a one-word synonym. More specific descriptions
            discriminate better among similar categories.
        model: HuggingFace model ID for sentence-transformers. Defaults to
            ``"Snowflake/snowflake-arctic-embed-xs"`` (~90 MB, fast on CPU).
            Any sentence-transformers-compatible model works.
        encoder: Optional custom encoding function ``(texts: list[str]) ->
            np.ndarray``. When provided, ``model`` is ignored. Useful for
            testing or using a pre-loaded model.
        store_path: Path to a SQLite file for the learning store.
            Enables tier-1 exact-match lookups. Created automatically.
        corpus_path: Directory for a ChromaDB persistent store. Enables
            tier-2 kNN classification. Requires ``pip install racclassify[chromadb]``.
        confidence_threshold: Minimum cosine similarity for tier-3 prototype
            classification. Below this, tier-4 augmented re-embed is tried.
        augment_threshold: Minimum similarity required after augmented
            re-embed. Below this, ``default_label`` is returned.
        knn_confidence_threshold: Minimum confidence from the kNN tier to
            accept its result over the prototype tier.
        default_label: Label to return when all tiers fail. Defaults to the
            first key in ``categories``.
        namespace: Isolates the learning store and corpus when multiple
            Classifier instances share the same files.

    Example::

        clf = Classifier(
            categories={
                "HEALTHCARE": "Medical insurance, hospitals, Medicare, prescription drugs...",
                "DEFENSE": "Military, national security, veterans' affairs, weapons...",
                "ENVIRONMENT": "Climate change, pollution, EPA, conservation...",
            },
            store_path="my_classifier.db",
        )

        label, confidence = clf.classify("A bill to expand Medicare prescription coverage")
        # → ("HEALTHCARE", 0.71)

        results = clf.classify_multi("Infrastructure bill covering roads and clean energy")
        # → [{"label": "ENVIRONMENT", "confidence": 0.62}, {"label": "TAXES", ...}]

        clf.record("doc-123", label="HEALTHCARE", text="A bill to expand Medicare...")
    """

    def __init__(
        self,
        categories: dict[str, str],
        *,
        model: str = DEFAULT_MODEL,
        encoder: EncoderFn | None = None,
        store_path: str | None = None,
        corpus_path: str | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        augment_threshold: float = DEFAULT_AUGMENT_THRESHOLD,
        knn_confidence_threshold: float = DEFAULT_KNN_CONFIDENCE,
        default_label: str | None = None,
        namespace: str = "default",
    ) -> None:
        if not categories:
            raise ValueError("categories must not be empty")

        self._categories = dict(categories)
        self._labels = list(categories.keys())
        self._model = model
        self._encoder = encoder
        self._threshold = confidence_threshold
        self._augment_threshold = augment_threshold
        self._knn_threshold = knn_confidence_threshold
        self._default = default_label or self._labels[0]
        self._namespace = namespace

        if self._default not in self._categories:
            raise ValueError(f"default_label {self._default!r} not in categories")

        self._proto_cache: np.ndarray | None = None
        self._proto_lock = threading.Lock()

        self._store = None
        if store_path:
            from ._store import LearningStore
            self._store = LearningStore(store_path, namespace=namespace)

        self._corpus = None
        if corpus_path:
            try:
                from ._corpus import ReferenceCorpus
                self._corpus = ReferenceCorpus(
                    persist_path=corpus_path, namespace=namespace
                )
                logger.info(
                    "Reference corpus loaded (%d examples)", self._corpus.size()
                )
            except ImportError:
                logger.warning(
                    "corpus_path given but chromadb is not installed. "
                    "Install with: pip install racclassify[chromadb]"
                )

    # ── Public API ───────────────────────────────────────────────────────

    def classify(self, text: str, doc_id: str | None = None) -> ClassificationResult:
        """Classify text into a single category.

        Returns the best-matching label and its cosine similarity score.
        The score reflects how similar the text is to that category's
        prototype description — it is not a calibrated probability.
        """
        if not text or not text.strip():
            return ClassificationResult(label=self._default, confidence=0.0)

        # Tier 1: exact-match learning store
        if doc_id and self._store:
            stored = self._store.lookup(doc_id)
            if stored:
                return ClassificationResult(label=stored[0], confidence=stored[1])

        query_emb = self._embed_one(text)

        # Tier 2: kNN reference corpus
        if self._corpus and self._corpus.size() >= 5:
            knn_label, knn_conf = self._corpus.classify(query_emb)
            if knn_label and knn_conf >= self._knn_threshold:
                return ClassificationResult(label=knn_label, confidence=knn_conf)

        # Tier 3: nearest-centroid prototype
        proto_embs = self._prototype_embeddings()
        scores = proto_embs @ query_emb
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self._threshold:
            return ClassificationResult(label=self._labels[best_idx], confidence=best_score)

        # Tier 4: augmented re-embed
        augmented = self._embed_one(f"This text is about: {text}")
        aug_scores = proto_embs @ augmented
        aug_idx = int(np.argmax(aug_scores))
        aug_score = float(aug_scores[aug_idx])

        if aug_score >= self._augment_threshold:
            return ClassificationResult(label=self._labels[aug_idx], confidence=aug_score)

        return ClassificationResult(label=self._default, confidence=best_score)

    def classify_multi(
        self,
        text: str,
        doc_id: str | None = None,
        max_labels: int = 4,
    ) -> list[ClassificationResult]:
        """Classify text into multiple categories.

        Returns a ranked list of all categories whose cosine similarity
        exceeds a secondary threshold, up to ``max_labels``. The first
        element is identical to what ``classify()`` would return.

        The gap-ratio filter prevents low-confidence noise from inflating
        the label count — a secondary label must score at least 70% of
        the primary label's score to be included.

        Grounded in Adler & Wilkerson (2012), who show that most
        legislation spans 2-4 policy domains.
        """
        primary = self.classify(text, doc_id=doc_id)

        if not text or not text.strip():
            return [primary]

        query_emb = self._embed_one(text)
        proto_embs = self._prototype_embeddings()
        scores = proto_embs @ query_emb

        pairs = sorted(
            zip(self._labels, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        top_score = pairs[0][1] if pairs else 0.0
        gap_threshold = max(
            DEFAULT_MULTI_SECONDARY_THRESHOLD,
            top_score * DEFAULT_MULTI_GAP_RATIO,
        )

        results: list[ClassificationResult] = []
        for label, score in pairs[:max_labels]:
            if score >= gap_threshold or label == primary.label:
                results.append(ClassificationResult(label=label, confidence=round(score, 4)))

        if not any(r.label == primary.label for r in results):
            results.insert(0, primary)

        return results

    def record(
        self,
        doc_id: str,
        label: str,
        text: str | None = None,
        confidence: float = 0.9,
    ) -> None:
        """Record a classification for adaptive learning.

        Stores the result in the learning store (for future exact-match
        lookup) and adds the document's embedding to the reference corpus
        (for future kNN classification).

        Args:
            doc_id: Stable identifier for the document.
            label: The correct label for this document.
            text: The document text. Required when ``corpus_path`` was set,
                so the embedding can be computed and stored.
            confidence: Confidence level to store. Use lower values for
                auto-generated labels, higher for human-verified ones.
        """
        if label not in self._categories:
            raise ValueError(f"label {label!r} not in categories")

        if self._store:
            self._store.record(doc_id, label=label, confidence=confidence, text=text)

        if self._corpus and text:
            emb = self._embed_one(text)
            self._corpus.add(doc_id, emb, label=label)

    def clear_prototype_cache(self) -> None:
        """Clear cached prototype embeddings.

        Call this if you modify ``categories`` after construction — though
        modifying categories after construction is not recommended.
        """
        with self._proto_lock:
            self._proto_cache = None

    def stats(self) -> dict[str, object]:
        """Return diagnostic information about the classifier's state."""
        info: dict[str, object] = {
            "categories": self._labels,
            "model": self._model if self._encoder is None else "<custom encoder>",
            "has_store": self._store is not None,
            "has_corpus": self._corpus is not None,
        }
        if self._store:
            info["store"] = self._store.stats()
        if self._corpus:
            info["corpus_size"] = self._corpus.size()
            info["corpus_distribution"] = self._corpus.label_distribution()
        return info

    # ── Internals ────────────────────────────────────────────────────────

    def _prototype_embeddings(self) -> np.ndarray:
        """Compute and cache the prototype embeddings for all categories."""
        with self._proto_lock:
            if self._proto_cache is None:
                descriptions = list(self._categories.values())
                self._proto_cache = embed_batch(
                    descriptions, model_name=self._model, encoder=self._encoder
                )
            return self._proto_cache

    def _embed_one(self, text: str) -> np.ndarray:
        return embed_one(text, model_name=self._model, encoder=self._encoder)
