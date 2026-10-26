
def build_prompt(question, retrieved_documents):
    """Build a prompt using the question and retrieved text."""

    context = "\n".join(
        item["text"] for item in retrieved_documents
    )

    prompt = f"""
You are a helpful assistant.
Answer the question using only the provided context.
If the context does not contain the answer, say you do not know.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()


def generate_answer(question, retrieved_documents):
    """Prepare the prompt for a future language model."""

    prompt = build_prompt(question, retrieved_documents)

    # The language model is not connected yet.
    # This function returns the prompt for now.
    return {
        "answer": None,
        "prompt": prompt,
        "model_status": "not_connected"
    }
