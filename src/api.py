from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import retrieve
from generator import generate_answer

app = FastAPI(
    title="RAG Document Assistant",
    description="An API for document-based question answering",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG Document Assistant API"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    documents = retrieve(question)
    result = generate_answer(question, documents)

    return {
        "question": question,
        "retrieved_documents": documents,
        "answer": result["answer"],
        "prompt": result["prompt"],
        "model_status": result["model_status"],
    }
