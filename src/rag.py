
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_DIR / "data" / "documents" / "company_guide.txt"


def load_documents():
    """Load the project document and split it into sentences."""
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    text = DOCUMENT_PATH.read_text(encoding="utf-8")

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    return sentences



def retrieve(question, top_k=3):
    """Retrieve the most relevant sentences for a question."""

    if not question.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    sentences = load_documents()

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(sentences + [question])

    similarities = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "text": sentences[index],
            "score": round(float(similarities[index]), 3)
        })

    return results

if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    results = retrieve(question)

    print("\nRetrieved sentences:")

    for item in results:
        print(f"\nScore: {item['score']:.3f}")
        print(item["text"])
