# RAG Document Assistant

A Python-based Retrieval-Augmented Generation (RAG) project designed to answer questions using information retrieved from documents.

The project demonstrates a document retrieval pipeline, prompt construction for a language model, a FastAPI interface, and a basic retrieval evaluation workflow.

## Overview

The system takes a user's question, searches a local document for relevant information, and prepares the retrieved content as context for question answering.

The project is organized into modular components to make it easier to extend, evaluate, and integrate with a language model in the future.

## Features

- **Document loading and preprocessing** from local text files
- **TF-IDF-based retrieval** to identify relevant text passages
- **Cosine similarity scoring** to rank retrieved passages
- **Prompt construction** using the question and retrieved context
- **FastAPI endpoints** for a question-answering interface
- **Retrieval evaluation** using predefined questions and expected keywords
- **Automated tests** for retrieval behavior and input validation

## Project Architecture

```text
rag-document-assistant/
├── data/
│   ├── documents/
│   │   └── company_guide.txt
│   └── evaluation/
│       └── questions.json
├── src/
│   ├── rag.py
│   ├── generator.py
│   ├── api.py
│   └── evaluate.py
├── tests/
│   └── test_rag.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- Python
- Scikit-learn
- TF-IDF and cosine similarity
- FastAPI
- Pydantic
- Pytest
- JSON

## How It Works

1. The document loader reads the source text.
2. The text is split into smaller passages.
3. TF-IDF converts the passages and user question into numerical representations.
4. Cosine similarity is used to rank passages by relevance.
5. The selected passages are combined with the question to construct a prompt.
6. The API layer exposes an endpoint for submitting questions and returning retrieved context.

## Retrieval Example

**Example question:**

`What does the company develop?`

The retrieval component searches the sample company guide and returns ranked passages with similarity scores. This demonstrates the basic retrieval workflow before language-model integration.

The current retrieval method is a lightweight baseline intended for experimentation and further improvement.

## Language Model Integration

The project includes a separate generation module that constructs a prompt from the retrieved passages and the user's question.

The language model is not currently connected. The generation module therefore prepares the prompt but does not generate a final natural-language answer.

## Evaluation and Testing

The repository includes an evaluation script with predefined questions and expected keywords, as well as automated tests for retrieval behavior, result structure, and input validation.

These components provide a starting point for measuring retrieval quality and maintaining the code as the project evolves.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the retrieval component from the project root:

```bash
python src/rag.py
```

To use the API, the FastAPI application can be launched with:

```bash
uvicorn api:app --app-dir src --reload
```

The API command is provided for development and has not been verified in the current project environment.

## Current Status

The TF-IDF retrieval component has been run locally with the sample document.

The repository also contains the prompt-generation module, API implementation, retrieval evaluation script, and automated tests. These additional components have been prepared in code but have not yet been fully executed and verified.

A language model has not been connected, so end-to-end question answering is not yet available.

## Future Improvements

- Integrate a local or hosted language model
- Improve document chunking and retrieval quality
- Add semantic search using embedding models
- Evaluate retrieval with additional questions and documents
- Verify API behavior and automated tests
- Add support for multiple documents and file formats

## Project Goal

This project explores the core components of a document question-answering application and provides a modular foundation for further development of a complete RAG pipeline.
