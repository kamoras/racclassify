"""Shared test fixtures.

All tests use a deterministic mock encoder so the test suite runs
without downloading any model weights.  The mock encoder assigns each
text a unit vector whose direction is determined by a hash of the text —
semantically meaningless, but stable and consistent within a test run.
"""

import hashlib

import numpy as np
import pytest

from racclassify import Classifier


CATEGORIES = {
    "HEALTHCARE": "Medical insurance, hospitals, Medicare, prescription drugs, public health.",
    "DEFENSE": "Military, armed forces, national security, weapons, veterans.",
    "ENVIRONMENT": "Climate change, pollution, EPA, conservation, clean energy.",
    "TAXES": "Federal budget, tax reform, IRS, fiscal policy, spending.",
    "PROCEDURAL": "Procedural motions, commemorations, naming buildings.",
}


def _deterministic_encoder(texts: list[str]) -> np.ndarray:
    """Encode texts as deterministic unit vectors from SHA-256 hashes."""
    dim = 32
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        out.append(vec[:dim])
    return np.array(out)


@pytest.fixture
def clf() -> Classifier:
    """A Classifier with mock encoder and no persistence."""
    return Classifier(categories=CATEGORIES, encoder=_deterministic_encoder)


@pytest.fixture
def clf_with_store(tmp_path) -> Classifier:
    """A Classifier with mock encoder and a real SQLite learning store."""
    return Classifier(
        categories=CATEGORIES,
        encoder=_deterministic_encoder,
        store_path=str(tmp_path / "test.db"),
    )
