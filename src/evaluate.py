import json
from pathlib import Path

from rag import retrieve


PROJECT_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_DIR / "data" / "evaluation" / "questions.json"
)

RESULTS_DIR = PROJECT_DIR / "results"


def load_questions():
    """Load evaluation questions from the JSON file."""
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(top_k=3):
    """Evaluate retrieval using expected keywords."""
    questions = load_questions()
    results = []

    for item in questions:
        question = item["question"]
        expected_keywords = [
            keyword.lower()
            for keyword in item["expected_keywords"]
        ]

        retrieved_documents = retrieve(question, top_k=top_k)

        retrieved_text = " ".join(
            document["text"].lower()
            for document in retrieved_documents
        )

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword in retrieved_text
        ]

        keyword_recall = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords
            else 0.0
        )

        results.append({
            "question": question,
            "retrieved_documents": retrieved_documents,
            "matched_keywords": matched_keywords,
            "keyword_recall": round(keyword_recall, 3),
        })

    average_recall = (
        sum(item["keyword_recall"] for item in results) / len(results)
        if results
        else 0.0
    )

    return {
        "number_of_questions": len(results),
        "average_keyword_recall": round(average_recall, 3),
        "details": results,
    }


def save_results(evaluation_results):
    """Save evaluation results as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "retrieval_evaluation.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_results,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    evaluation_results = evaluate_retrieval()
    save_results(evaluation_results)
    print("Evaluation code finished.")
