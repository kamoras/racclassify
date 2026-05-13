"""Embedding model abstraction.

Wraps sentence-transformers with in-process caching and optional
injection of a custom encoder — useful for testing without downloading
model weights.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

_lock = threading.Lock()
_models: dict[str, object] = {}

EncoderFn = Callable[[list[str]], np.ndarray]


def _load_model(model_name: str) -> object:
    with _lock:
        if model_name not in _models:
            from sentence_transformers import SentenceTransformer
            _models[model_name] = SentenceTransformer(model_name)
        return _models[model_name]


def embed_batch(
    texts: list[str],
    *,
    model_name: str,
    encoder: EncoderFn | None = None,
    max_chars: int = 500,
) -> np.ndarray:
    """Embed a list of texts, returning a normalised (N, D) matrix."""
    if not texts:
        return np.empty((0, 0))

    clipped = [t[:max_chars] for t in texts]

    if encoder is not None:
        embs = encoder(clipped)
    else:
        model = _load_model(model_name)
        embs = model.encode(  # type: ignore[attr-defined]
            clipped,
            show_progress_bar=False,
            batch_size=min(64, len(clipped)),
        )

    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def embed_one(
    text: str,
    *,
    model_name: str,
    encoder: EncoderFn | None = None,
    max_chars: int = 500,
) -> np.ndarray:
    """Embed a single text and return a normalised 1-D vector."""
    return embed_batch([text], model_name=model_name, encoder=encoder, max_chars=max_chars)[0]
