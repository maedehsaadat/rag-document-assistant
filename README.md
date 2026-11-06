# RAG Document Assistant

A document-based question-answering application built with Python, TF-IDF, FastAPI, and Large Language Models (LLMs).

The project implements a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant information from local documents and uses an LLM to generate context-grounded answers.

## Overview

RAG Document Assistant allows users to ask questions about a document and receive answers based on its content.

The system combines traditional information retrieval techniques with LLM-based answer generation, providing a lightweight and modular approach to document question answering.

## Features

- Document loading and text preprocessing
- TF-IDF-based document retrieval
- Cosine similarity for ranking relevant passages
- Context-aware prompt construction
- LLM integration through OpenRouter API
- Context-grounded answer generation
- FastAPI endpoint for question answering
- Retrieval evaluation using keyword recall
- Answer evaluation with predefined questions
- Automated tests for retrieval functionality

## Architecture

The application follows a modular RAG pipeline:

```text
User Question
      |
      v
  FastAPI Endpoint
      |
      v
 Document Retrieval
 (TF-IDF + Cosine Similarity)
      |
      v
 Retrieved Passages
      |
      v
 Prompt Construction
      |
      v
 OpenRouter LLM API
      |
      v
 Generated Answer
      |
      v
   API Response
```

## Project Structure

```text
rag-document-assistant/
│
├── data/
│   ├── documents/
│   │   └── company_guide.txt
│   └── evaluation/
│       └── questions.json
│
├── results/
│   ├── answer_evaluation.json
│   └── retrieval_evaluation.json
│
├── src/
│   ├── api.py
│   ├── rag.py
│   ├── generator.py
│   ├── evaluate.py
│   └── evaluate_answers.py
│
├── tests/
│   └── test_rag.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies

- Python
- Scikit-learn
- TF-IDF
- Cosine Similarity
- FastAPI
- Pydantic
- OpenAI Python SDK
- OpenRouter API
- Pytest
- JSON

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/maedehsaadat/rag-document-assistant.git
cd rag-document-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root using `.env.example` as a template.

Configure your OpenRouter credentials:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace the placeholder with your own API key.

## Running the Application

Start the FastAPI server:

```bash
py -m uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Usage

**Endpoint:** `POST /ask`

Example request:

```json
{
  "question": "What programming language is recommended?"
}
```

Example response:

```json
{
  "question": "What programming language is recommended?",
  "answer": "Python",
  "retrieved_documents": [
    {
      "text": "Python is the programming language used by the system for document processing and information retrieval",
      "score": 0.273
    }
  ],
  "model_status": "connected",
  "model": "openrouter/free"
}
```

## Evaluation

The project includes separate evaluation scripts for retrieval and answer generation.

**Retrieval Evaluation**

- Uses predefined questions and expected keywords.
- Measures keyword recall across retrieved passages.
- Saves evaluation results to `results/retrieval_evaluation.json`.

**Answer Evaluation**

- Evaluates generated answers against predefined evaluation questions.
- Saves evaluation results to `results/answer_evaluation.json`.

## Project Status

The application has been executed locally and its main question-answering API has been successfully tested.

The implementation demonstrates an end-to-end RAG workflow, from document retrieval to LLM-based answer generation through OpenRouter.

