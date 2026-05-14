"""Integration tests using a real embedding model.

Skipped unless RACCLASSIFY_RUN_INTEGRATION=1 is set — the model
download (~17 MB for paraphrase-MiniLM-L3-v2) makes these too slow
for the normal unit-test pass.

These tests verify that classification is semantically meaningful with
real embeddings, not just that the API contract is correct.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RACCLASSIFY_RUN_INTEGRATION") != "1",
    reason="set RACCLASSIFY_RUN_INTEGRATION=1 to run",
)

MODEL = "paraphrase-MiniLM-L3-v2"

CATEGORIES = {
    "HEALTHCARE": (
        "Medical insurance, hospitals, Medicare, Medicaid, prescription drugs, "
        "public health, clinical care."
    ),
    "DEFENSE": (
        "Military, armed forces, national security, Pentagon, weapons, veterans, war, troops."
    ),
    "ENVIRONMENT": (
        "Climate change, pollution, EPA, conservation, clean energy, "
        "carbon emissions, endangered species."
    ),
    "TAXES": (
        "Federal budget, tax reform, IRS, fiscal policy, government spending, "
        "appropriations, revenue."
    ),
    "IMMIGRATION": (
        "Border security, asylum seekers, deportation, visas, citizenship, undocumented immigrants."
    ),
}


@pytest.fixture(scope="module")
def clf():
    from racclassify import Classifier

    return Classifier(categories=CATEGORIES, model=MODEL)


# ── Single-label accuracy ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A bill to expand Medicare prescription drug coverage", "HEALTHCARE"),
        ("Increase Pentagon funding for new fighter jets", "DEFENSE"),
        ("New EPA rules limiting carbon emissions from power plants", "ENVIRONMENT"),
        ("Cut the corporate tax rate and reform the IRS", "TAXES"),
        ("Stricter asylum rules and increased border patrol funding", "IMMIGRATION"),
    ],
)
def test_classifies_correctly(clf, text, expected):
    result = clf.classify(text)
    assert result.label == expected, (
        f"Expected {expected!r}, got {result.label!r} "
        f"(confidence {result.confidence:.3f}) for: {text!r}"
    )
    assert result.confidence > 0.20


# ── Multi-label ───────────────────────────────────────────────────────────


def test_multi_label_primary_matches_single(clf):
    text = "Proposed cuts to the defense budget would reduce Pentagon spending"
    single = clf.classify(text)
    multi = clf.classify_multi(text)
    assert multi[0].label == single.label


def test_multi_label_all_valid_categories(clf):
    results = clf.classify_multi("A healthcare bill with tax implications")
    assert all(r.label in CATEGORIES for r in results)
    assert len(results) >= 1


# ── Learning store overrides prototype ───────────────────────────────────


def test_store_overrides_prototype(tmp_path):
    from racclassify import Classifier

    clf = Classifier(categories=CATEGORIES, model=MODEL, store_path=str(tmp_path / "s.db"))
    clf.record("doc-1", label="TAXES", text="Medicare expansion")
    result = clf.classify("ignored text", doc_id="doc-1")
    assert result.label == "TAXES"


# ── Confidence ordering ───────────────────────────────────────────────────


def test_confidence_in_valid_range(clf):
    results = clf.classify_multi("Medicare expansion for elderly patients", max_labels=5)
    for r in results:
        assert 0.0 <= r.confidence <= 1.5


def test_empty_text_returns_default(clf):
    from racclassify import ClassificationResult

    result = clf.classify("")
    assert isinstance(result, ClassificationResult)
    assert result.label in CATEGORIES
    assert result.confidence == 0.0
