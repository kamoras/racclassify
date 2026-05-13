# racclassify

Zero-shot text classification using semantic prototypes with adaptive kNN learning.

Describe each category in plain English. No labeled training data required to start — the classifier improves over time as you record correct classifications that seed a growing reference corpus.

```python
from racclassify import Classifier

clf = Classifier(
    categories={
        "BILLING":   "invoices, payments, charges, refunds, subscriptions, pricing",
        "TECHNICAL": "bugs, crashes, errors, login issues, performance, API failures",
        "GENERAL":   "questions, feedback, feature requests, account management",
    },
    store_path="classifier.db",   # optional: persist learning across runs
)

result = clf.classify("My payment was charged twice last month")
print(result.label, result.confidence)  # BILLING  0.68

# Multi-label: returns all plausible categories, ranked
for r in clf.classify_multi("Login error causing payment to fail"):
    print(r.label, r.confidence)

# Record correct label to improve future classification
clf.record("ticket-42", label="BILLING", text="My payment was charged twice")
```

## How it works

Classification uses a four-tier strategy, falling through to the next tier when confidence is low:

| Tier | Method | Speed | Notes |
|------|--------|-------|-------|
| 1 | Exact-match learning store | Instant | Previously seen `doc_id` — returns stored label at confidence 1.0 |
| 2 | kNN reference corpus | Fast | Similarity-weighted vote over past examples (optional, requires ChromaDB) |
| 3 | Nearest-centroid prototype | Fast | Cosine similarity to category descriptions — always available |
| 4 | Augmented re-embed | Fast | Retries with `"This text is about: {text}"` prefix for low-confidence cases |

The system gets smarter over time. Every `record()` call adds an example to the reference corpus so future similar texts classify by analogy rather than raw prototype similarity.

## Installation

```bash
pip install racclassify
```

With ChromaDB for kNN learning (recommended for production):

```bash
pip install racclassify[chromadb]
```

## Usage

### Basic classification

```python
from racclassify import Classifier

clf = Classifier(
    categories={
        "HEALTHCARE": "Medical insurance, hospitals, Medicare, prescription drugs, public health.",
        "DEFENSE":    "Military, armed forces, national security, weapons procurement, veterans.",
        "ENVIRONMENT":"Climate change, pollution, EPA regulations, conservation, clean energy.",
    }
)

label, confidence = clf.classify("A bill to cut EPA emissions standards")
# label = "ENVIRONMENT", confidence ≈ 0.62
```

### Multi-label classification

Real documents often span multiple categories. `classify_multi` returns all plausible labels:

```python
results = clf.classify_multi("Infrastructure bill funding roads and clean energy")
# [ClassificationResult(label="ENVIRONMENT", confidence=0.61),
#  ClassificationResult(label="TAXES", confidence=0.44)]
```

### Adaptive learning with persistence

```python
clf = Classifier(
    categories={...},
    store_path="./my_classifier.db",       # SQLite, for exact-match lookup
    corpus_path="./my_classifier_corpus/", # ChromaDB, for kNN (optional)
)

# After classifying, record the result to improve future accuracy
result = clf.classify("some text", doc_id="doc-123")
clf.record("doc-123", label=result.label, text="some text")

# Next run: doc-123 returns instantly from the learning store
# Similar documents will now use kNN instead of raw prototype similarity
```

### Custom encoder

Inject your own embedding function — useful for testing or using a pre-loaded model:

```python
import numpy as np

def my_encoder(texts: list[str]) -> np.ndarray:
    # return (N, D) float32 array of embeddings
    ...

clf = Classifier(categories={...}, encoder=my_encoder)
```

## Choosing category descriptions

Better descriptions → better accuracy. A few guidelines:

- **Be specific.** `"Medical insurance, hospitals, Medicare, prescription drugs"` beats `"Healthcare"`.
- **Cover synonyms.** If users might say "copay" or "deductible", include those terms.
- **Distinguish overlapping categories.** If TECH and TELECOM often get confused, add distinguishing phrases to both descriptions.
- **Length is fine.** Sentence-transformers handle up to ~512 tokens; 2-3 sentences per category is ideal.

## Embedding models

The default model is `Snowflake/snowflake-arctic-embed-xs` (~90 MB, fast on CPU/ARM). You can use any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) compatible model:

```python
clf = Classifier(categories={...}, model="all-MiniLM-L6-v2")   # popular alternative
clf = Classifier(categories={...}, model="all-mpnet-base-v2")   # slower, higher accuracy
```

## Academic grounding

| Design decision | Reference |
|-----------------|-----------|
| Nearest-centroid prototype classification | Rocchio (1971); Manning, Raghavan & Schütze (2008) Ch. 14 |
| kNN with distance-weighted vote | Cover & Hart (1967); Dudani (1976) |
| Experience replay via learning store | Lin (1992) |
| Multi-label gap-ratio filter | Adler & Wilkerson (2012) |
| Dense embeddings over keyword matching | Reimers & Gurevych (2019, Sentence-BERT) |

## Development

```bash
git clone https://github.com/kamoras/racclassify
cd racclassify
pip install -e ".[dev]"
pytest
```

## License

MIT
