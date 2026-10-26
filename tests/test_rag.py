
import sys
from pathlib import Path
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import retrieve


@pytest.fixture
def sample_documents(monkeypatch):
    """Provide predictable documents for retrieval tests."""

    documents = [
        "Our company develops AI software for businesses",
        "Customer support teams use the software to answer questions",
        "The system uses Python to process documents",
    ]

    monkeypatch.setattr("rag.load_documents", lambda: documents)

    return documents


def test_retrieve_returns_results(sample_documents):
    results = retrieve("What does the company develop?")

    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_results_have_required_fields(sample_documents):
    results = retrieve("What does the company develop?")

    for item in results:
        assert set(item.keys()) == {"text", "score"}
        assert isinstance(item["text"], str)
        assert isinstance(item["score"], float)


def test_retrieve_respects_top_k(sample_documents):
    results = retrieve(
        "What does the company develop?",
        top_k=2,
    )

    assert len(results) <= 2


def test_empty_question_returns_empty_list(sample_documents):
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_retrieved_scores_are_between_zero_and_one(sample_documents):
    results = retrieve("What does the company develop?")

    for item in results:
        assert 0.0 <= item["score"] <= 1.0


def test_most_relevant_document_is_retrieved(sample_documents):
    results = retrieve(
        "What does the company develop?",
        top_k=1,
    )

    assert len(results) == 1
    assert "AI software" in results[0]["text"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_error(sample_documents, top_k):
    with pytest.raises(ValueError):
        retrieve(
            "What does the company develop?",
            top_k=top_k,
        )
