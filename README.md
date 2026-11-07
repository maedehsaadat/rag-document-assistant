# RAG Document Assistant

**An AI-powered document question-answering application built with Python, TF-IDF, Streamlit, FastAPI, and Large Language Models (LLMs).**

🚀 **Live Demo:** [Try RAG Document Assistant](https://maedehsaadat-rag-document-assistant-app-c6ytq0.streamlit.app/)

The project implements a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant information from documents using TF-IDF and uses an LLM through OpenRouter to generate context-grounded answers.

## Overview

RAG Document Assistant allows users to ask questions about documents and receive answers based on their content.

The system combines traditional information retrieval techniques with LLM-based answer generation, providing a lightweight and modular approach to document question answering.

The application provides both an interactive Streamlit web interface and a FastAPI endpoint.

## Features

- Document loading and text preprocessing
- TF-IDF-based document retrieval
- Cosine similarity for ranking relevant passages
- Context-aware prompt construction
- LLM integration through OpenRouter API
- Context-grounded answer generation
- Interactive Streamlit web interface
- Retrieved passage visualization and similarity scores
- FastAPI endpoint for question answering
- Retrieval evaluation using keyword recall
- Answer evaluation with predefined questions
- Automated tests for retrieval functionality

## Live Demo

Try the application online:

**[Open RAG Document Assistant](https://maedehsaadat-rag-document-assistant-app-c6ytq0.streamlit.app/)**

The web interface allows users to submit questions and receive AI-generated answers based on retrieved document passages.

## Architecture

The application follows a modular RAG pipeline:

```text
User Question
      |
      v
Streamlit Web Interface / FastAPI Endpoint
      |
      v
Document Retrieval
(TF-IDF + Cosine Similarity)
      |
      v
Retrieved Passages
      |
      v
Context-Aware Prompt Construction
      |
      v
OpenRouter LLM API
      |
      v
Generated Answer
      |
      v
Displayed Answer / API Response
```

## Project Structure

```text
rag-document-assistant/
│
├── app.py
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
- Streamlit
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

### Streamlit Web Interface

Run the interactive web application:

```bash
py -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### FastAPI Server

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

The application has been successfully implemented and tested locally, including its RAG pipeline and OpenRouter integration.

A live Streamlit deployment is available for interactive testing.

The project demonstrates an end-to-end RAG workflow, combining document retrieval, similarity-based ranking, and LLM-powered answer generation.

