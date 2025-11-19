# AmbedkarGPT-Intern-Task

A simple command-line Q&A system that ingests a short speech (speech.txt) and answers questions based only on that content.

## Tech stack
- Python 3.11
- LangChain
- ChromaDB (local vector store)
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Ollama local LLM using Mistral 7B (local, free)

## Repository contents
- `main.py` — main script (ingest, embed, store, retrieve, answer)
- `speech.txt` — provided speech excerpt (use this exact text)
- `requirements.txt` — Python dependencies
- `.gitignore` — recommended ignores
- `chroma_db/` — Chroma persistence folder (created at runtime)

## Pre-requisites

### 1) Python environment
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
