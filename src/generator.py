
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)


def build_prompt(question, retrieved_documents):
    """Build a prompt using retrieved documents."""

    context = "\n\n".join(
        item["text"] for item in retrieved_documents
    )

    prompt = f"""
You are a document-based question answering assistant.

Answer the question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer is not in the context, clearly say so.
- Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()


def generate_answer(question, retrieved_documents):
    """Generate an answer using OpenRouter LLM API."""

    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    prompt = build_prompt(question, retrieved_documents)

    if not retrieved_documents:
        return {
            "answer": "I could not find relevant information in the documents.",
            "prompt": prompt,
            "model_status": "no_context",
            "model": OPENROUTER_MODEL,
        }

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=2,
    )

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document-based question answering "
                        "assistant. Answer strictly using the provided "
                        "context. Never invent information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )

        answer = response.choices[0].message.content

        return {
            "answer": (
                answer.strip()
                if answer and answer.strip()
                else "The model returned an empty response."
            ),
            "prompt": prompt,
            "model_status": "connected",
            "model": OPENROUTER_MODEL,
        }

    except APITimeoutError as error:
        raise RuntimeError(
            "OpenRouter request timed out."
        ) from error

    except APIConnectionError as error:
        raise RuntimeError(
            "Could not connect to OpenRouter."
        ) from error

    except APIError as error:
        raise RuntimeError(
            f"OpenRouter API request failed: {error}"
        ) from error
