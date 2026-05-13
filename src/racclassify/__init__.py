"""racclassify — zero-shot text classification using semantic prototypes
with adaptive kNN learning.

Basic usage::

    from racclassify import Classifier

    clf = Classifier(
        categories={
            "BILLING": "invoices, payments, charges, refunds, subscriptions",
            "TECHNICAL": "bugs, crashes, errors, login issues, performance",
            "GENERAL": "questions, feedback, feature requests, other",
        },
        store_path="classifier.db",   # optional: persist learning across runs
    )

    result = clf.classify("My payment was charged twice last month")
    print(result.label, result.confidence)  # BILLING 0.68

    # record correct label to improve future kNN classification
    clf.record("ticket-42", label="BILLING", text="My payment was charged twice")
"""

from ._classifier import ClassificationResult, Classifier

__all__ = ["Classifier", "ClassificationResult"]
__version__ = "0.1.0"
