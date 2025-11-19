# AmbedkarGPT-Intern-Task

A simple command-line Q&A system that ingests a short speech (speech.txt) and answers questions based only on that content.

## Tech stack
- Python 3.11
- LangChain
- ChromaDB (local vector store)
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Ollama local LLM using Mistral 7B (local, free)

## Project Folder Structure
```wasm
AmbedkarGPT-Intern-Task/
├─ README.md
├─ requirements.txt
├─ speech.txt
├─ main.py
├─ .gitignore
├─ Sample_Q&A_SS.png
└─ chroma_db/                # created by Chroma when running (do NOT commit large DB)

```


## Repository contents
- `main.py` — main script (ingest, embed, store, retrieve, answer)
- `speech.txt` — provided speech excerpt (use this exact text)
- `requirements.txt` — Python dependencies
- `.gitignore` — recommended ignores
- `chroma_db/` — Chroma persistence folder (created at runtime)

## Pre-requisites


### Install Python 3.11

https://www.python.org/downloads/

### Install Git

https://git-scm.com/downloads

### Install Ollama & Run The Project

Download from the official installer:

https://ollama.com/download

Ollama will automatically run as a background service on:
```arduino
http://localhost:11434
```
To Test
```nginx
ollama server
```
If it starts correctly, you will see something like:
```csharp
Ollama is running on port 11434
```
Ensure your model is downloaded
```nginx
ollama pull mistral
```
Now run your project
```css
python main.py
```
